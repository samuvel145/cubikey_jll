"""
audio_smoother.py  — AUDIO-SMOOTH-v1
PCM16 fade-in / fade-out helpers to eliminate click/pop at TTS chunk boundaries.

Revert keyword: AUDIO-SMOOTH-v1
  - Delete this file
  - Remove AudioSmootherProcessor from pipeline/processors.py
  - Remove AudioSmootherProcessor from imports and both pipelines in pipeline/agent.py

Assumptions:
  - Sample format : PCM 16-bit signed little-endian
  - Sample rate   : 16 000 Hz (pipeline default — see config.py SAMPLE_RATE)
  - FADE_SAMPLES  : 128 = 8 ms at 16 kHz (inaudible gap, eliminates click)
"""

import struct

FADE_SAMPLES = 128  # 8 ms at 16 kHz


def fade_in(pcm: bytes, n: int = FADE_SAMPLES) -> bytes:
    """Linear fade-in on the first n samples of a PCM16 buffer."""
    if len(pcm) < n * 2:
        return pcm
    samples = list(struct.unpack(f'<{len(pcm) // 2}h', pcm))
    for i in range(min(n, len(samples))):
        samples[i] = int(samples[i] * i / n)
    return struct.pack(f'<{len(samples)}h', *samples)


def fade_out(pcm: bytes, n: int = FADE_SAMPLES) -> bytes:
    """Linear fade-out on the last n samples of a PCM16 buffer."""
    if len(pcm) < n * 2:
        return pcm
    samples = list(struct.unpack(f'<{len(pcm) // 2}h', pcm))
    total = len(samples)
    for i in range(n):
        idx = total - n + i
        samples[idx] = int(samples[idx] * (n - i - 1) / n)
    return struct.pack(f'<{total}h', *samples)


def silence_pad(ms: int, sample_rate: int = 16000) -> bytes:
    """Generate N milliseconds of PCM16 silence."""
    return b'\x00\x00' * int(sample_rate * ms / 1000)
