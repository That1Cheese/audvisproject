"""Tests for 3D landmark normalization."""

import numpy as np
import pytest
from pronun.visual.features.normalizer import normalize_landmarks, get_mouth_width
from pronun.config import ALL_LIP_INDICES, LEFT_LIP_CORNER, RIGHT_LIP_CORNER

_LEFT_IDX = ALL_LIP_INDICES.index(LEFT_LIP_CORNER)
_RIGHT_IDX = ALL_LIP_INDICES.index(RIGHT_LIP_CORNER)
N = len(ALL_LIP_INDICES)


def _random_landmarks(mouth_width=100.0, seed=42):
    """Create random landmarks with a specific mouth width."""
    rng = np.random.RandomState(seed)
    lm = rng.randn(N, 3) * 10

    # Set left/right corners to define width
    lm[_LEFT_IDX] = [-mouth_width / 2, 0, 0]
    lm[_RIGHT_IDX] = [mouth_width / 2, 0, 0]
    return lm


def test_normalized_mouth_width_is_one():
    lm = _random_landmarks(mouth_width=150.0)
    norm = normalize_landmarks(lm)
    left = norm[_LEFT_IDX]
    right = norm[_RIGHT_IDX]
    width = np.linalg.norm(left - right)
    assert abs(width - 1.0) < 1e-6


def test_normalized_centroid_is_zero():
    lm = _random_landmarks(mouth_width=100.0)
    # Shift all landmarks
    lm += [500, 300, 50]
    norm = normalize_landmarks(lm)
    centroid = norm.mean(axis=0)
    np.testing.assert_allclose(centroid, [0, 0, 0], atol=1e-6)


def test_different_scales_produce_same_width():
    lm1 = _random_landmarks(mouth_width=50.0, seed=1)
    lm2 = _random_landmarks(mouth_width=200.0, seed=1)

    # Scale lm2 landmarks (except corners) to match structure
    norm1 = normalize_landmarks(lm1)
    norm2 = normalize_landmarks(lm2)

    w1 = np.linalg.norm(norm1[_LEFT_IDX] - norm1[_RIGHT_IDX])
    w2 = np.linalg.norm(norm2[_LEFT_IDX] - norm2[_RIGHT_IDX])

    assert abs(w1 - 1.0) < 1e-6
    assert abs(w2 - 1.0) < 1e-6


def test_get_mouth_width():
    lm = _random_landmarks(mouth_width=120.0)
    # After setting corners, actual 3D distance should be close to 120
    # (centroid subtraction may shift, but raw width should be ~120)
    w = get_mouth_width(lm)
    assert abs(w - 120.0) < 1e-6
