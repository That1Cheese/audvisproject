"""Unified feature vector construction from normalized 3D mouth landmarks."""

import numpy as np
from scipy.spatial import ConvexHull

from pronun.config import (
    ALL_LIP_INDICES,
    LEFT_LIP_CORNER,
    LOWER_LIP_MID,
    OUTER_LIP_INDICES,
    RIGHT_LIP_CORNER,
    UPPER_LIP_MID,
)

# Indices within the extracted landmark array
_LEFT_IDX = ALL_LIP_INDICES.index(LEFT_LIP_CORNER)
_RIGHT_IDX = ALL_LIP_INDICES.index(RIGHT_LIP_CORNER)
_UPPER_IDX = ALL_LIP_INDICES.index(UPPER_LIP_MID)
_LOWER_IDX = ALL_LIP_INDICES.index(LOWER_LIP_MID)

# Upper lip indices (within outer lip, first half roughly)
_UPPER_LIP_OUTER = [ALL_LIP_INDICES.index(i) for i in OUTER_LIP_INDICES[:10]]
_LOWER_LIP_OUTER = [ALL_LIP_INDICES.index(i) for i in OUTER_LIP_INDICES[10:]]


def build_feature(normalized: np.ndarray) -> np.ndarray:
    """Build a unified feature vector from normalized 3D landmarks.

    Features:
        1. Mouth height H (3D distance upper↔lower lip midpoints)
        2. Opening ratio R = H / W (W=1 after normalization)
        3. Lip protrusion P (avg z-difference upper vs lower lip)
        4. 3D mouth area (convex hull area)
        5. Flattened normalized 3D landmarks [x1,y1,z1,...,xN,yN,zN]

    Args:
        normalized: Normalized landmarks of shape (N, 3).

    Returns:
        Feature vector as 1D array.
    """
    upper = normalized[_UPPER_IDX]
    lower = normalized[_LOWER_IDX]

    # 1. Mouth height
    height = float(np.linalg.norm(upper - lower))

    # 2. Opening ratio (width = 1 after normalization)
    ratio = height

    # 3. Lip protrusion (z-axis difference between upper and lower lip)
    upper_z = normalized[_UPPER_LIP_OUTER, 2].mean()
    lower_z = normalized[_LOWER_LIP_OUTER, 2].mean()
    protrusion = float(upper_z - lower_z)

    # 4. 3D mouth area via convex hull
    try:
        hull = ConvexHull(normalized[:, :2])  # 2D projection for area
        area = float(hull.volume)  # In 2D, volume = area
    except Exception:
        area = 0.0

    # 5. Flattened landmarks
    flat = normalized.flatten()

    # Concatenate all features
    geometric = np.array([height, ratio, protrusion, area], dtype=np.float64)
    return np.concatenate([geometric, flat])


def build_feature_sequence(
    normalized_seq: list[np.ndarray | None],
) -> list[np.ndarray]:
    """Build feature vectors with delta features for a frame sequence.

    For each frame t:
        f_t = build_feature(normalized_t)
        Δf_t = f_t - f_{t-1}  (zero for first frame)
        output_t = concatenate(f_t, Δf_t)

    Frames where landmarks are None are skipped.

    Returns:
        List of feature vectors (with deltas appended).
    """
    static_features = []
    valid_indices = []

    for i, norm in enumerate(normalized_seq):
        if norm is not None:
            static_features.append(build_feature(norm))
            valid_indices.append(i)

    if not static_features:
        return []

    result = []
    for i, feat in enumerate(static_features):
        if i == 0:
            delta = np.zeros_like(feat)
        else:
            delta = feat - static_features[i - 1]
        result.append(np.concatenate([feat, delta]))

    return result


def feature_dim(num_landmarks: int) -> int:
    """Compute expected feature dimension.

    4 geometric + 3*N landmark coords, all doubled for deltas.
    """
    static = 4 + 3 * num_landmarks
    return static * 2
