"""Synchronized audio + video capture."""

import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

from pronun.config import CHANNELS, MAX_RECORD_SECONDS, SAMPLE_RATE, SILENCE_DURATION, SILENCE_THRESHOLD
from pronun.workflow.camera import Camera


class SyncRecorder:
    """Records audio and video simultaneously.

    Audio is captured via sounddevice, video via OpenCV camera.
    Both streams run in parallel threads with shared stop signal.
    """

    def __init__(
        self,
        camera: Camera | None = None,
        sample_rate: int = SAMPLE_RATE,
        max_seconds: float = MAX_RECORD_SECONDS,
        silence_threshold: float = SILENCE_THRESHOLD,
        silence_duration: float = SILENCE_DURATION,
    ):
        self.camera = camera
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration

    def record(self, audio_path: str) -> tuple[np.ndarray, list[np.ndarray]]:
        """Record audio and video simultaneously.

        Args:
            audio_path: Path to save recorded WAV file.

        Returns:
            Tuple of (audio_array, list_of_rgb_frames).
        """
        stop_event = threading.Event()
        audio_chunks: list[np.ndarray] = []
        video_frames: list[np.ndarray] = []

        block_size = int(self.sample_rate * 0.1)
        silence_blocks = int(self.silence_duration / 0.1)
        silent_count = 0
        started = False

        def audio_callback(indata, frames, time_info, status):
            nonlocal silent_count, started
            chunk = indata[:, 0].copy()
            energy = np.sqrt(np.mean(chunk ** 2))

            if energy > self.silence_threshold:
                started = True
                silent_count = 0
            elif started:
                silent_count += 1

            if started:
                audio_chunks.append(chunk)

            if started and silent_count >= silence_blocks:
                stop_event.set()

        def video_loop():
            if self.camera is None:
                return
            while not stop_event.is_set():
                frame = self.camera.read_frame()
                if frame is not None:
                    video_frames.append(frame)
                time.sleep(1.0 / 30)  # ~30 fps

        video_thread = threading.Thread(target=video_loop, daemon=True)
        video_thread.start()

        max_blocks = int(self.max_seconds / 0.1)

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            blocksize=block_size,
            callback=audio_callback,
        ):
            start_time = time.time()
            while not stop_event.is_set():
                time.sleep(0.05)
                if time.time() - start_time > self.max_seconds:
                    stop_event.set()

        stop_event.set()
        video_thread.join(timeout=2.0)

        if audio_chunks:
            audio = np.concatenate(audio_chunks)
        else:
            audio = np.zeros(self.sample_rate, dtype=np.float32)

        sf.write(audio_path, audio, self.sample_rate)
        return audio, video_frames
