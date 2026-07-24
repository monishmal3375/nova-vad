"""
Codec encode/decode round-trip degradation (plan Section 7.3's telephony
matrix — simulated codec, not actual RTC/VoIP transmission; that's a
separate, larger gap explicitly left for a future round, not conflated
with this one). Uses PyAV (bundles its own ffmpeg/libopus, no system
install required) — installed this round after confirming no system
ffmpeg/opus was available.

G.711 (A-law and mu-law): downsampled to 8kHz (G.711's native telephony
rate) before companding, then upsampled back to 16kHz — this matches real
G.711 behavior, which is specifically an 8kHz-band codec, not a 16kHz one.

Opus: encoded directly at 16kHz (a natively supported Opus rate — no
forced resampling to 48kHz, which would add an unrelated resampling
artifact), at 24kbps — a realistic moderate VoIP bitrate, not Opus's
highest-quality setting, so the codec actually introduces audible
degradation rather than being near-transparent.
"""
import io

import av
import librosa
import numpy as np


def _resample(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def apply_g711(audio_16k: np.ndarray, variant: str = "alaw") -> np.ndarray:
    """variant: 'alaw' or 'mulaw'. Input/output both 16kHz float32."""
    codec_name = f"pcm_{variant}"
    audio_8k = _resample(audio_16k, 16000, 8000)

    buf = io.BytesIO()
    output = av.open(buf, mode="w", format="wav")
    stream = output.add_stream(codec_name, rate=8000, layout="mono")
    frame = av.AudioFrame.from_ndarray(audio_8k.reshape(1, -1), format="flt", layout="mono")
    frame.sample_rate = 8000
    frame.pts = 0
    for packet in stream.encode(frame):
        output.mux(packet)
    for packet in stream.encode(None):
        output.mux(packet)
    output.close()
    buf.seek(0)

    input_ = av.open(buf, mode="r", format="wav")
    decoded = []
    for frame in input_.decode(audio=0):
        decoded.append(frame.to_ndarray().astype(np.float32) / 32768.0)
    input_.close()
    decoded_8k = np.concatenate(decoded, axis=-1).flatten()

    return _resample(decoded_8k, 8000, 16000).astype(np.float32)


def apply_opus(audio_16k: np.ndarray, bitrate: int = 24000) -> np.ndarray:
    """Input/output both 16kHz float32. bitrate in bits/sec (24000 = 24kbps,
    a realistic moderate VoIP setting)."""
    buf = io.BytesIO()
    output = av.open(buf, mode="w", format="ogg")
    stream = output.add_stream("libopus", rate=16000, layout="mono")
    stream.bit_rate = bitrate
    frame = av.AudioFrame.from_ndarray(audio_16k.reshape(1, -1).astype(np.float32), format="flt", layout="mono")
    frame.sample_rate = 16000
    frame.pts = 0
    for packet in stream.encode(frame):
        output.mux(packet)
    for packet in stream.encode(None):
        output.mux(packet)
    output.close()
    buf.seek(0)

    input_ = av.open(buf, mode="r", format="ogg")
    decode_rate = input_.streams.audio[0].rate  # Opus always decodes at 48kHz internally,
    # regardless of the encoder's requested rate — this is correct Opus behavior (the
    # codec's internal representation is fixed at 48kHz), not a bug to work around by
    # requesting a different rate. Must resample the decoded output back to 16kHz.
    decoded = []
    for frame in input_.decode(audio=0):
        decoded.append(frame.to_ndarray().astype(np.float32))
    input_.close()
    decoded_native = np.concatenate(decoded, axis=-1).flatten()
    decoded_16k = _resample(decoded_native, decode_rate, 16000)

    # Opus's encoder look-ahead adds a small fixed pre-skip; trim or pad to match
    # input length so ground-truth timing (which codec degradation doesn't
    # semantically shift) stays aligned for frame-level scoring.
    if len(decoded_16k) > len(audio_16k):
        decoded_16k = decoded_16k[:len(audio_16k)]
    elif len(decoded_16k) < len(audio_16k):
        decoded_16k = np.pad(decoded_16k, (0, len(audio_16k) - len(decoded_16k)))
    return decoded_16k
