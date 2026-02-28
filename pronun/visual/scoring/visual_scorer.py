"""Visual pronunciation scorer using HMM log-likelihood."""

import numpy as np

from pronun.visual.scoring.hmm import GaussianHMM
from pronun.visual.scoring.reference import ReferenceBaseline


class VisualScorer:
    """Converts HMM log-likelihood into a 0-100 pronunciation score.

    Score pipeline:
        1. L = log P(V | HMM) via Forward Algorithm
        2. L_norm = L / T (time normalization)
        3. Score_raw = exp(L_norm - L_ref)
        4. Score = 100 * min(1, Score_raw)
    """

    def __init__(self, reference: ReferenceBaseline | None = None):
        self.reference = reference or ReferenceBaseline()

    def score(
        self,
        hmm: GaussianHMM,
        observations: np.ndarray,
        word: str = "",
    ) -> dict:
        """Score an observation sequence against an HMM.

        Args:
            hmm: Trained HMM for the target viseme sequence.
            observations: Feature vectors, shape (T, feature_dim).
            word: Word being scored (for reference lookup).

        Returns:
            Dict with 'log_likelihood', 'log_likelihood_norm',
            'reference', 'score_raw', 'score'.
        """
        T = len(observations)
        if T == 0:
            return {
                "log_likelihood": -np.inf,
                "log_likelihood_norm": -np.inf,
                "reference": self.reference.get_reference(word),
                "score_raw": 0.0,
                "score": 0.0,
            }

        log_likelihood = hmm.forward(observations)
        log_likelihood_norm = log_likelihood / T

        l_ref = self.reference.get_reference(word)
        diff = log_likelihood_norm - l_ref
        if diff >= 0:
            score_raw = 1.0  # at or above native speaker level
        else:
            score_raw = float(np.exp(diff / self.reference.bandwidth))
        score = 100.0 * min(1.0, score_raw)

        return {
            "log_likelihood": log_likelihood,
            "log_likelihood_norm": log_likelihood_norm,
            "reference": l_ref,
            "score_raw": score_raw,
            "score": score,
        }

    def build_hmm(
        self,
        viseme_sequence: list[int],
        viseme_observations: dict[int, np.ndarray],
        feature_dim: int,
    ) -> GaussianHMM:
        """Build an HMM for a specific viseme sequence.

        Args:
            viseme_sequence: Target sequence of viseme IDs.
            viseme_observations: Dict mapping viseme_id → training observations
                                 of shape (num_obs, feature_dim).
            feature_dim: Dimensionality of feature vectors.

        Returns:
            Trained GaussianHMM ready for scoring.
        """
        num_states = len(viseme_sequence)
        hmm = GaussianHMM(num_states, feature_dim)

        for state_idx, viseme_id in enumerate(viseme_sequence):
            if viseme_id in viseme_observations:
                value = viseme_observations[viseme_id]
                if isinstance(value, tuple) and len(value) == 2:
                    hmm.set_emission_params(state_idx, value[0], value[1])
                else:
                    hmm.train_emissions(state_idx, value)

        return hmm
