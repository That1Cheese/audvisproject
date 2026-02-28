"""Microphone recording with energy-based silence detection."""

import numpy as np
import sounddevice as sd
import soundfile as sf

from pronun.config import (
    CHANNELS,
    MAX_RECORD_SECONDS,
    SAMPLE_RATE,
    SILENCE_DURATION,
    SILENCE_THRESHOLD,
)


def record_audio(
    output_path: str,
    sample_rate: int = SAMPLE_RATE,
    silence_threshold: float = SILENCE_THRESHOLD,
    silence_duration: float = SILENCE_DURATION,
    max_seconds: float = MAX_RECORD_SECONDS,
) -> np.ndarray:
    """Record audio from microphone until silence or max duration.

    Returns the recorded audio as a 1D numpy array.
    """
    block_size = int(sample_rate * 0.1)  # 100ms blocks
    silence_blocks = int(silence_duration / 0.1)
    max_blocks = int(max_seconds / 0.1)

    chunks: list[np.ndarray] = []
    silent_count = 0
    started = False

    def callback(indata, frames, time_info, status):
        nonlocal silent_count, started
        chunk = indata[:, 0].copy()
        energy = np.sqrt(np.mean(chunk ** 2))

        if energy > silence_threshold:
            started = True
            silent_count = 0
        elif started:
            silent_count += 1

        if started:
            chunks.append(chunk)

    with sd.InputStream(
        samplerate=sample_rate,
        channels=CHANNELS,
        blocksize=block_size,
        callback=callback,
    ):
        while True:
            sd.sleep(100)
            if started and silent_count >= silence_blocks:
                break
            if len(chunks) >= max_blocks:
                break

    if not chunks:
        audio = np.zeros(sample_rate, dtype=np.float32)
    else:
        audio = np.concatenate(chunks)

    sf.write(output_path, audio, sample_rate)
    return audio
