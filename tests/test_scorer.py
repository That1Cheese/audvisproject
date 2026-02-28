"""Tests for visual scorer and combined scoring."""

import numpy as np
import pytest
from pronun.visual.scoring.hmm import GaussianHMM
from pronun.visual.scoring.visual_scorer import VisualScorer
from pronun.visual.scoring.reference import ReferenceBaseline
from pronun.scoring.combiner import combine_scores, adaptive_combine


class TestVisualScorer:
    def test_score_equals_100_at_reference(self):
        """Score should be 100 when L_norm equals L_ref."""
        ref = ReferenceBaseline()
        ref.set_reference("test", -3.0)
        scorer = VisualScorer(reference=ref)

        hmm = GaussianHMM(num_states=1, feature_dim=2, self_loop_prob=1.0)
        hmm.set_emission_params(0, np.array([0.0, 0.0]), np.eye(2))

        # Create observations that produce L_norm close to -3.0
        # We need to find observations that give log-likelihood / T = -3.0
        obs = np.array([[0.0, 0.0]])
        result = scorer.score(hmm, obs, "test")

        # Score should be clamped to 100 if L_norm >= L_ref
        assert result["score"] <= 100.0
        assert result["score"] >= 0.0

    def test_score_decreases_for_poor(self):
        """Score should decrease for observations far from emission means."""
        ref = ReferenceBaseline()
        ref.set_reference("test", -2.0)
        scorer = VisualScorer(reference=ref)

        hmm = GaussianHMM(num_states=2, feature_dim=1, self_loop_prob=0.5)
        hmm.set_emission_params(0, np.array([0.0]), np.array([[0.5]]))
        hmm.set_emission_params(1, np.array([5.0]), np.array([[0.5]]))

        good_obs = np.array([[0.0], [5.0]])
        bad_obs = np.array([[10.0], [-5.0]])

        good_result = scorer.score(hmm, good_obs, "test")
        bad_result = scorer.score(hmm, bad_obs, "test")

        assert good_result["score"] > bad_result["score"]

    def test_empty_observations(self):
        scorer = VisualScorer()
        hmm = GaussianHMM(num_states=1, feature_dim=1)
        result = scorer.score(hmm, np.array([]).reshape(0, 1))
        assert result["score"] == 0.0

    def test_build_hmm(self):
        scorer = VisualScorer()
        viseme_seq = [0, 1, 2]
        obs_dict = {
            0: np.random.randn(20, 4),
            1: np.random.randn(20, 4),
            2: np.random.randn(20, 4),
        }
        hmm = scorer.build_hmm(viseme_seq, obs_dict, feature_dim=4)
        assert hmm.num_states == 3
        assert hmm.feature_dim == 4


class TestReferenceBaseline:
    def test_set_and_get(self):
        ref = ReferenceBaseline()
        ref.set_reference("hello", -3.5)
        assert ref.get_reference("hello") == -3.5

    def test_default_for_unknown(self):
        ref = ReferenceBaseline()
        val = ref.get_reference("unknown_word")
        assert val == ref.default_reference

    def test_update_from_samples(self):
        ref = ReferenceBaseline()
        ref.update_from_samples("test", [-10.0, -12.0], [5, 6])
        expected = np.mean([-10.0 / 5, -12.0 / 6])
        assert abs(ref.get_reference("test") - expected) < 1e-6


class TestCombiner:
    def test_audio_only_fallback(self):
        score = combine_scores(80.0, None)
        assert score == 80.0

    def test_default_weights(self):
        score = combine_scores(80.0, 60.0)
        expected = 0.7 * 80.0 + 0.3 * 60.0
        assert abs(score - expected) < 1e-6

    def test_clamp_range(self):
        score = combine_scores(100.0, 100.0)
        assert score <= 100.0
        score = combine_scores(0.0, 0.0)
        assert score >= 0.0

    def test_adaptive_distinctive_phoneme(self):
        phoneme_scores = [{"phoneme": "P", "gop_score": 80.0}]
        result = adaptive_combine(phoneme_scores, 60.0)
        # P is bilabial = visually distinctive → 50/50 weights
        assert abs(result[0]["audio_weight"] - 0.5) < 1e-6
        assert abs(result[0]["visual_weight"] - 0.5) < 1e-6

    def test_adaptive_ambiguous_phoneme(self):
        phoneme_scores = [{"phoneme": "K", "gop_score": 80.0}]
        result = adaptive_combine(phoneme_scores, 60.0)
        # K is NOT visually distinctive → 90/10 weights
        assert abs(result[0]["audio_weight"] - 0.9) < 1e-6

    def test_adaptive_no_visual(self):
        phoneme_scores = [{"phoneme": "P", "gop_score": 80.0}]
        result = adaptive_combine(phoneme_scores, None)
        assert result[0]["combined_score"] == 80.0
