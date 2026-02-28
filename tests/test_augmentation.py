"""Tests for landmark data augmentation."""

import numpy as np
import pytest

from pronun.visual.features.augmentation import (
    _rotation_matrix,
    augment_landmarks,
    augment_sequence,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sample_landmarks(rng):
    """Normalized landmarks: 40 points × 3 coords."""
    return rng.standard_normal((40, 3))


class TestAugmentLandmarks:
    def test_output_shape(self, sample_landmarks, rng):
        result = augment_landmarks(sample_landmarks, rng)
        assert result.shape == sample_landmarks.shape

    def test_differs_from_input(self, sample_landmarks, rng):
        result = augment_landmarks(sample_landmarks, rng)
        assert not np.allclose(result, sample_landmarks)

    def test_deterministic_with_same_seed(self, sample_landmarks):
        r1 = augment_landmarks(sample_landmarks, np.random.default_rng(0))
        r2 = augment_landmarks(sample_landmarks, np.random.default_rng(0))
        np.testing.assert_array_equal(r1, r2)

    def test_does_not_modify_input(self, sample_landmarks, rng):
        original = sample_landmarks.copy()
        augment_landmarks(sample_landmarks, rng)
        np.testing.assert_array_equal(sample_landmarks, original)

    def test_zero_jitter_and_scale_only_rotates(self, sample_landmarks, rng):
        result = augment_landmarks(
            sample_landmarks, rng,
            jitter_std=0.0, rotation_deg=5.0, scale_std=0.0,
            scale_factor=1.0,
        )
        # Pure rotation preserves inter-landmark distances
        orig_dists = np.linalg.norm(
            sample_landmarks[1:] - sample_landmarks[:-1], axis=1
        )
        aug_dists = np.linalg.norm(result[1:] - result[:-1], axis=1)
        np.testing.assert_allclose(orig_dists, aug_dists, atol=1e-10)

    def test_scale_only(self, sample_landmarks, rng):
        scale = 1.5
        result = augment_landmarks(
            sample_landmarks, rng,
            jitter_std=0.0, rotation_deg=0.0, scale_std=0.0,
            rotation_matrix=np.eye(3), scale_factor=scale,
        )
        # No jitter, identity rotation → result = input * scale
        # (jitter is still applied first but with std=0)
        np.testing.assert_allclose(result, sample_landmarks * scale, atol=1e-10)


class TestRotationMatrix:
    def test_identity_at_zero(self):
        R = _rotation_matrix(0.0, 0.0, 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-15)

    def test_orthogonal(self, rng):
        yaw, pitch, roll = rng.uniform(-0.5, 0.5, 3)
        R = _rotation_matrix(yaw, pitch, roll)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)

    def test_det_one(self, rng):
        yaw, pitch, roll = rng.uniform(-0.5, 0.5, 3)
        R = _rotation_matrix(yaw, pitch, roll)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12

    def test_preserves_distances(self, sample_landmarks, rng):
        R = _rotation_matrix(0.1, -0.05, 0.08)
        rotated = sample_landmarks @ R.T
        orig_dists = np.linalg.norm(
            sample_landmarks[1:] - sample_landmarks[:-1], axis=1
        )
        rot_dists = np.linalg.norm(rotated[1:] - rotated[:-1], axis=1)
        np.testing.assert_allclose(orig_dists, rot_dists, atol=1e-12)


class TestAugmentSequence:
    def test_correct_number_of_copies(self, sample_landmarks, rng):
        seq = [sample_landmarks, sample_landmarks.copy()]
        result = augment_sequence(seq, n_aug=5, rng=rng)
        assert len(result) == 5

    def test_each_copy_same_length(self, sample_landmarks, rng):
        seq = [sample_landmarks, None, sample_landmarks.copy()]
        result = augment_sequence(seq, n_aug=3, rng=rng)
        for copy in result:
            assert len(copy) == len(seq)

    def test_none_frames_preserved(self, sample_landmarks, rng):
        seq = [None, sample_landmarks, None, sample_landmarks.copy(), None]
        result = augment_sequence(seq, n_aug=2, rng=rng)
        for copy in result:
            assert copy[0] is None
            assert copy[2] is None
            assert copy[4] is None
            assert copy[1] is not None
            assert copy[3] is not None

    def test_augmented_frames_have_correct_shape(self, sample_landmarks, rng):
        seq = [sample_landmarks, sample_landmarks.copy()]
        result = augment_sequence(seq, n_aug=2, rng=rng)
        for copy in result:
            for frame in copy:
                assert frame.shape == sample_landmarks.shape

    def test_copies_differ(self, sample_landmarks, rng):
        seq = [sample_landmarks]
        result = augment_sequence(seq, n_aug=3, rng=rng)
        # At least two copies should differ (extremely unlikely to be identical)
        frames = [copy[0] for copy in result]
        assert not np.allclose(frames[0], frames[1])

    def test_zero_augmentations(self, sample_landmarks, rng):
        seq = [sample_landmarks]
        result = augment_sequence(seq, n_aug=0, rng=rng)
        assert result == []
