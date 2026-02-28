"""3D landmark normalization: centering + scale normalization."""

import numpy as np

from pronun.config import ALL_LIP_INDICES, LEFT_LIP_CORNER, RIGHT_LIP_CORNER

# Indices within the extracted landmark array (not MediaPipe global indices)
_LEFT_IDX = ALL_LIP_INDICES.index(LEFT_LIP_CORNER)
_RIGHT_IDX = ALL_LIP_INDICES.index(RIGHT_LIP_CORNER)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Normalize 3D mouth landmarks by centering and scaling.

    Step 1 — Center: subtract centroid (mean of all landmark positions).
    Step 2 — Scale: divide by 3D mouth width (distance between lip corners).

    After normalization, mouth width = 1.0 for all samples.

    Args:
        landmarks: Array of shape (N, 3) with raw 3D landmarks.

    Returns:
        Normalized landmarks of shape (N, 3).
    """
    # Step 1: Centering
    centroid = landmarks.mean(axis=0)  # (3,)
    centered = landmarks - centroid

    # Step 2: Scale by 3D mouth width
    left = centered[_LEFT_IDX]
    right = centered[_RIGHT_IDX]
    width = np.linalg.norm(left - right)

    if width < 1e-8:
        return centered

    normalized = centered / width
    return normalized


def get_mouth_width(landmarks: np.ndarray) -> float:
    """Compute 3D mouth width from raw landmarks."""
    left = landmarks[_LEFT_IDX]
    right = landmarks[_RIGHT_IDX]
    return float(np.linalg.norm(left - right))


def normalize_sequence(landmark_seq: list[np.ndarray | None]) -> list[np.ndarray | None]:
    """Normalize a sequence of landmark arrays."""
    return [normalize_landmarks(lm) if lm is not None else None for lm in landmark_seq]
