"""
Unified predict_mask(audio_path) -> 10ms speech/no-speech mask for all five
systems, replacing the original benchmark's mismatched evaluation (NOVA-VAD
scored as a whole-file classifier; WebRTC converted to a file label via a
custom "40% of frames" rule; Silero/Pyannote/SpeechBrain not exercised at
their native frame/segment resolution at all).

Each adapter returns a Python list of 0/1 ints, one per 10ms frame, covering
[0, duration_ms) — directly comparable to a scene's frame_labels_10ms.
"""
import numpy as np
import soundfile as sf

FRAME_MS = 10


def _n_frames(duration_ms):
    return duration_ms // FRAME_MS


def _mask_from_intervals_ms(duration_ms, intervals_ms):
    n = _n_frames(duration_ms)
    mask = [0] * n
    for start_ms, end_ms in intervals_ms:
        start_frame = max(0, int(start_ms // FRAME_MS))
        end_frame = min(n, -(-int(end_ms) // FRAME_MS))  # ceil
        for i in range(start_frame, end_frame):
            mask[i] = 1
    return mask


# ── NOVA-VAD: sliding-window classification (coarse frame-level mode) ──────
def predict_mask_nova(audio_path, rf, gbt, scaler, duration_ms,
                       window_s=1.0, hop_s=0.25, threshold=0.5):
    from src.classifier import extract_features_from_array
    import librosa

    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    window_samples = int(window_s * sr)
    hop_samples = int(hop_s * sr)
    n = _n_frames(duration_ms)
    mask = [0] * n

    t = 0
    while t < len(audio):
        window = audio[t:t + window_samples]
        if len(window) < int(0.2 * sr):  # too short a tail to trust
            break
        feats = extract_features_from_array(window, sr)
        X_scaled = scaler.transform([feats])
        rf_prob = rf.predict_proba(X_scaled)[0][1]
        gbt_prob = gbt.predict_proba(X_scaled)[0][1]
        avg_prob = (rf_prob + gbt_prob) / 2
        is_speech = avg_prob > threshold

        # this window's verdict "votes" for the hop-sized chunk it centers on
        chunk_start_ms = round(t / sr * 1000)
        chunk_end_ms = round((t + hop_samples) / sr * 1000)
        start_frame = max(0, chunk_start_ms // FRAME_MS)
        end_frame = min(n, chunk_end_ms // FRAME_MS)
        for i in range(start_frame, end_frame):
            mask[i] = 1 if is_speech else 0

        t += hop_samples

    return mask


# ── WebRTC: native 30ms frames, each maps to exactly three 10ms frames ─────
def predict_mask_webrtc(audio_path, duration_ms, aggressiveness=3):
    import webrtcvad
    from src.vad import read_wav

    vad = webrtcvad.Vad(aggressiveness)
    pcm, sr = read_wav(audio_path)

    frame_ms = 30
    frame_size = int(sr * frame_ms / 1000) * 2  # bytes, *2 for 16-bit

    n = _n_frames(duration_ms)
    mask = [0] * n
    frame_idx_10ms = 0
    for i in range(0, len(pcm) - frame_size + 1, frame_size):
        frame = pcm[i:i + frame_size]
        is_speech = len(frame) == frame_size and vad.is_speech(frame, sr)
        for _ in range(3):  # 30ms = three 10ms sub-frames
            if frame_idx_10ms < n:
                mask[frame_idx_10ms] = 1 if is_speech else 0
                frame_idx_10ms += 1

    return mask


# ── Silero: native segment timestamps -> mask ───────────────────────────────
def predict_mask_silero(audio_path, model, duration_ms, threshold=0.5):
    from silero_vad import get_speech_timestamps
    import torch
    import librosa

    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    wav = torch.FloatTensor(audio)
    speeches = get_speech_timestamps(wav, model, sampling_rate=16000, threshold=threshold)
    intervals_ms = [(s["start"] / sr * 1000, s["end"] / sr * 1000) for s in speeches]
    return _mask_from_intervals_ms(duration_ms, intervals_ms)


# ── Pyannote: native timeline -> mask ───────────────────────────────────────
def predict_mask_pyannote(audio_path, pipeline, duration_ms):
    import torch
    import librosa

    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    waveform = torch.FloatTensor(audio).unsqueeze(0)
    output = pipeline({"waveform": waveform, "sample_rate": sr})
    intervals_ms = [(seg.start * 1000, seg.end * 1000) for seg in output.get_timeline()]
    return _mask_from_intervals_ms(duration_ms, intervals_ms)


def build_pyannote_pipeline(min_duration_on=0.0, min_duration_off=0.0):
    """onset/offset are NOT exposed as tunable pipeline parameters for
    pyannote/segmentation-3.0 -- confirmed by direct experimentation
    (pipeline.instantiate({'onset': ...}) raises
    "ValueError: parameter 'onset' does not exist"). This is because
    segmentation-3.0 is a powerset model, and pyannote's own source
    (pyannote/audio/pipelines/voice_activity_detection.py) hardcodes
    `self.onset = self.offset = 0.5` for powerset models instead of
    exposing them as Uniform() tunable hyperparameters, which only
    happens for non-powerset models. Only min_duration_on/min_duration_off
    (post-processing durations) are genuinely tunable here -- see
    reports/decision_v7.md Item 3 for the full documented finding."""
    import os
    from pyannote.audio import Model
    from pyannote.audio.pipelines import VoiceActivityDetection

    token = os.environ.get("HF_TOKEN")
    pmodel = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=token)
    pipeline = VoiceActivityDetection(segmentation=pmodel)
    pipeline.instantiate({"min_duration_on": min_duration_on, "min_duration_off": min_duration_off})
    return pipeline


# ── SpeechBrain: native boundaries -> mask ──────────────────────────────────
def predict_mask_speechbrain(audio_path, vad_model, duration_ms, tmp_path="data/tmp_sb_frame.wav",
                              activation_th=0.5, deactivation_th=0.25):
    import os
    import librosa

    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    sf.write(tmp_path, audio, sr)
    try:
        boundaries = vad_model.get_speech_segments(
            tmp_path, activation_th=activation_th, deactivation_th=deactivation_th)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    intervals_ms = [(float(b[0]) * 1000, float(b[1]) * 1000) for b in boundaries]
    return _mask_from_intervals_ms(duration_ms, intervals_ms)
