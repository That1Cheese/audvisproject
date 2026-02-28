"""Data augmentation for normalized 3D mouth landmarks.

Augmentations are applied after normalization (centering + mouth-width scaling)
and before feature computation. This broadens HMM emission distributions so the
model generalizes across input domains (e.g. GRID corpus → live webcam).

Three transforms, each applied independently with random parameters:
  1. Gaussian jitter — simulates landmark detection noise
  2. Small 3D rotation — simulates head pose variation
  3. Scale perturbation — simulates residual size differences
"""

import numpy as np

from pronun.config import AUG_JITTER_STD, AUG_ROTATION_DEG, AUG_SCALE_STD


def _rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Build a 3x3 rotation matrix from yaw/pitch/roll in radians."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rp = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rr = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])

    return Ry @ Rp @ Rr


def augment_landmarks(
    landmarks: np.ndarray,
    rng: np.random.Generator,
    jitter_std: float = AUG_JITTER_STD,
    rotation_deg: float = AUG_ROTATION_DEG,
    scale_std: float = AUG_SCALE_STD,
    rotation_matrix: np.ndarray | None = None,
    scale_factor: float | None = None,
) -> np.ndarray:
    """Apply random jitter + rotation + scale to normalized landmarks.

    Args:
        landmarks: Array of shape (N, 3) — normalized mouth landmarks.
        rng: NumPy random generator for reproducibility.
        jitter_std: Std of per-coordinate Gaussian noise.
        rotation_deg: Max rotation per axis in degrees.
        scale_std: Std of scale perturbation around 1.0.
        rotation_matrix: Pre-computed rotation matrix (for sequence consistency).
        scale_factor: Pre-computed scale factor (for sequence consistency).

    Returns:
        Augmented landmarks of the same shape (N, 3).
    """
    result = landmarks.copy()

    # 1. Gaussian jitter (independent per coordinate per frame)
    result += rng.normal(0.0, jitter_std, size=result.shape)

    # 2. 3D rotation
    if rotation_matrix is None:
        max_rad = np.radians(rotation_deg)
        yaw = rng.uniform(-max_rad, max_rad)
        pitch = rng.uniform(-max_rad, max_rad)
        roll = rng.uniform(-max_rad, max_rad)
        rotation_matrix = _rotation_matrix(yaw, pitch, roll)
    result = result @ rotation_matrix.T

    # 3. Scale perturbation
    if scale_factor is None:
        scale_factor = rng.normal(1.0, scale_std)
    result *= scale_factor

    return result


def augment_sequence(
    landmark_seq: list[np.ndarray | None],
    n_aug: int,
    rng: np.random.Generator,
    jitter_std: float = AUG_JITTER_STD,
    rotation_deg: float = AUG_ROTATION_DEG,
    scale_std: float = AUG_SCALE_STD,
) -> list[list[np.ndarray | None]]:
    """Return n_aug augmented copies of a full landmark sequence.

    Each augmented copy uses the same rotation and scale across all frames
    (temporal coherence), but independent jitter per frame.

    Args:
        landmark_seq: List of (N, 3) arrays or None per frame.
        n_aug: Number of augmented copies to generate.
        rng: NumPy random generator.
        jitter_std: Std of per-coordinate Gaussian noise.
        rotation_deg: Max rotation per axis in degrees.
        scale_std: Std of scale perturbation around 1.0.

    Returns:
        List of n_aug augmented sequences, each with same length as input.
    """
    augmented = []
    for _ in range(n_aug):
        # Sample rotation and scale once per sequence
        max_rad = np.radians(rotation_deg)
        yaw = rng.uniform(-max_rad, max_rad)
        pitch = rng.uniform(-max_rad, max_rad)
        roll = rng.uniform(-max_rad, max_rad)
        rot_mat = _rotation_matrix(yaw, pitch, roll)
        scale = rng.normal(1.0, scale_std)

        seq_copy = []
        for lm in landmark_seq:
            if lm is None:
                seq_copy.append(None)
            else:
                seq_copy.append(augment_landmarks(
                    lm, rng,
                    jitter_std=jitter_std,
                    rotation_deg=rotation_deg,
                    scale_std=scale_std,
                    rotation_matrix=rot_mat,
                    scale_factor=scale,
                ))
        augmented.append(seq_copy)

    return augmented
