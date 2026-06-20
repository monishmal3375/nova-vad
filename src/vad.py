import webrtcvad
import soundfile as sf
import numpy as np

def read_wav(path: str):
    """
    Reads a .wav file and returns (pcm_data, sample_rate).
    Forces mono and 16kHz which WebRTC VAD requires.
    """
    audio, sr = sf.read(path, dtype='int16')

    # force mono if stereo
    if audio.ndim > 1:
        audio = audio[:, 0]

    # resample to 16kHz if needed
    if sr != 16000:
        import librosa
        audio_float = audio.astype(np.float32) / 32768.0
        audio_float = librosa.resample(audio_float, orig_sr=sr, target_sr=16000)
        audio = (audio_float * 32768).astype(np.int16)
        sr = 16000

    return audio.tobytes(), sr

def detect_speech(audio_path: str, aggressiveness: int = 3) -> dict:
    """
    Runs WebRTC VAD on an audio file.
    aggressiveness: 0 (least aggressive) to 3 (most aggressive)
    Returns a dict with file info and speech decision.
    """
    vad = webrtcvad.Vad(aggressiveness)
    pcm, sr = read_wav(audio_path)

    # WebRTC VAD works on 10, 20, or 30ms frames
    frame_duration_ms = 30
    frame_size = int(sr * frame_duration_ms / 1000) * 2  # *2 for 16-bit

    frames = [
        pcm[i:i+frame_size]
        for i in range(0, len(pcm) - frame_size + 1, frame_size)
    ]

    speech_frames = 0
    total_frames  = len(frames)

    for frame in frames:
        if len(frame) == frame_size:
            if vad.is_speech(frame, sr):
                speech_frames += 1

    speech_ratio = speech_frames / total_frames if total_frames > 0 else 0

    # if more than 20% of frames are speech → label as SPEECH
    is_speech = speech_ratio > 0.40

    return {
        "file":          audio_path,
        "total_frames":  total_frames,
        "speech_frames": speech_frames,
        "speech_ratio":  round(speech_ratio, 3),
        "prediction":    1 if is_speech else 0,
        "label":         "SPEECH" if is_speech else "NO SPEECH"
    }