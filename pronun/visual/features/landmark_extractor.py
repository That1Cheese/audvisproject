"""MediaPipe FaceLandmarker Tasks API → 3D mouth landmark extraction."""

import urllib.request

import numpy as np
import mediapipe as mp
import mediapipe.tasks

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
BaseOptions = mp.tasks.BaseOptions

from pronun.config import (
    ALL_LIP_INDICES,
    FACE_LANDMARKER_MODEL_URL,
    FACE_LANDMARKER_MODEL_PATH,
)


def _ensure_model():
    """Download the FaceLandmarker model if it doesn't exist locally."""
    if FACE_LANDMARKER_MODEL_PATH.exists():
        return FACE_LANDMARKER_MODEL_PATH
    FACE_LANDMARKER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, FACE_LANDMARKER_MODEL_PATH)
    return FACE_LANDMARKER_MODEL_PATH


class LandmarkExtractor:
    """Extracts 3D mouth landmarks from video frames using MediaPipe FaceLandmarker."""

    def __init__(self):
        model_path = _ensure_model()
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._frame_ts = 0

    def extract(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        """Extract 3D mouth landmarks from a single RGB frame.

        Args:
            frame_rgb: RGB image as numpy array (H, W, 3).

        Returns:
            Mouth landmarks as array of shape (N, 3) where N = len(ALL_LIP_INDICES),
            or None if no face detected.
        """
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(image, self._frame_ts)
        self._frame_ts += 33  # ~30fps increment

        if not result.face_landmarks:
            return None

        face = result.face_landmarks[0]
        h, w = frame_rgb.shape[:2]

        landmarks = np.zeros((len(ALL_LIP_INDICES), 3), dtype=np.float64)
        for i, idx in enumerate(ALL_LIP_INDICES):
            lm = face[idx]
            landmarks[i] = [lm.x * w, lm.y * h, lm.z * w]

        return landmarks

    def extract_sequence(self, frames: list[np.ndarray]) -> list[np.ndarray | None]:
        """Extract landmarks from a sequence of frames."""
        return [self.extract(f) for f in frames]

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
