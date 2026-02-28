"""Webcam capture via OpenCV."""

import cv2
import numpy as np

from pronun.config import CAMERA_FPS, CAMERA_INDEX


class Camera:
    """Webcam capture wrapper."""

    def __init__(self, index: int = CAMERA_INDEX, fps: int = CAMERA_FPS):
        self.index = index
        self.fps = fps
        self._cap = None

    def open(self):
        """Open the camera."""
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.index}")
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

    def read_frame(self) -> np.ndarray | None:
        """Read a single frame as RGB array.

        Returns None if frame cannot be read.
        """
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
