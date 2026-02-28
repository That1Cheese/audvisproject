"""Tests for feature vector construction."""

import numpy as np
import pytest
from pronun.visual.features.feature_builder import (
    build_feature,
    build_feature_sequence,
    feature_dim,
)
from pronun.config import ALL_LIP_INDICES

N = len(ALL_LIP_INDICES)


def _mock_normalized(seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(N, 3) * 0.1


def test_build_feature_shape():
    norm = _mock_normalized()
    feat = build_feature(norm)
    expected_static = 4 + 3 * N  # 4 geometric + 3N coordinates
    assert feat.shape == (expected_static,)


def test_build_feature_sequence_with_deltas():
    norms = [_mock_normalized(seed=i) for i in range(5)]
    seq = build_feature_sequence(norms)
    assert len(seq) == 5

    static_dim = 4 + 3 * N
    expected_dim = static_dim * 2  # static + delta
    for feat in seq:
        assert feat.shape == (expected_dim,)


def test_first_frame_delta_is_zero():
    norms = [_mock_normalized(seed=0), _mock_normalized(seed=1)]
    seq = build_feature_sequence(norms)
    static_dim = 4 + 3 * N
    delta = seq[0][static_dim:]
    np.testing.assert_allclose(delta, 0.0)


def test_none_frames_skipped():
    norms = [_mock_normalized(seed=0), None, _mock_normalized(seed=1)]
    seq = build_feature_sequence(norms)
    assert len(seq) == 2  # None frame skipped


def test_feature_dim_function():
    expected = (4 + 3 * N) * 2
    assert feature_dim(N) == expected


def test_empty_sequence():
    assert build_feature_sequence([]) == []
    assert build_feature_sequence([None, None]) == []
