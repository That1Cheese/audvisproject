"""Left-to-right HMM with Gaussian emissions and Forward Algorithm."""

import numpy as np
from scipy.stats import multivariate_normal

from pronun.config import HMM_COV_REGULARIZATION


class GaussianHMM:
    """Left-to-right HMM with Gaussian emission probabilities.

    States correspond to visemes in a sequence. Transitions only allow
    self-loops or advancing to the next state (left-to-right constraint).
    """

    def __init__(self, num_states: int, feature_dim: int, self_loop_prob: float = 0.5):
        """Initialize HMM.

        Args:
            num_states: Number of states (one per viseme in sequence).
            feature_dim: Dimensionality of observation feature vectors.
            self_loop_prob: Probability of staying in the same state.
        """
        self.num_states = num_states
        self.feature_dim = feature_dim
        self.self_loop_prob = self_loop_prob

        # Transition matrix: left-to-right
        self.log_trans = self._build_transition_matrix()

        # Initial state distribution: start in state 0
        self.log_pi = np.full(num_states, -np.inf)
        self.log_pi[0] = 0.0

        # Emission parameters (Gaussian): means and diagonal variances per state
        self.means = np.zeros((num_states, feature_dim))
        self.covs = np.array([
            np.ones(feature_dim) for _ in range(num_states)
        ])

    def _build_transition_matrix(self) -> np.ndarray:
        """Build left-to-right log-transition matrix."""
        trans = np.full((self.num_states, self.num_states), -np.inf)
        for i in range(self.num_states):
            trans[i][i] = np.log(self.self_loop_prob)
            if i + 1 < self.num_states:
                trans[i][i + 1] = np.log(1.0 - self.self_loop_prob)
            else:
                trans[i][i] = 0.0  # last state must self-loop
        return trans

    def set_emission_params(self, state: int, mean: np.ndarray, cov: np.ndarray):
        """Set Gaussian emission parameters for a state.

        Args:
            state: State index.
            mean: Mean vector of shape (feature_dim,).
            cov: Variance vector of shape (feature_dim,) or covariance matrix
                 of shape (feature_dim, feature_dim) — if 2-D, the diagonal
                 is extracted.
        """
        self.means[state] = mean
        if cov.ndim == 2:
            cov = np.diag(cov)
        self.covs[state] = cov + HMM_COV_REGULARIZATION

    def train_emissions(self, state: int, observations: np.ndarray):
        """Estimate Gaussian parameters from labeled observations for a state.

        Args:
            state: State index.
            observations: Array of shape (num_obs, feature_dim).
        """
        if len(observations) < 2:
            mean = observations[0] if len(observations) == 1 else np.zeros(self.feature_dim)
            cov = np.ones(self.feature_dim)
        else:
            mean = observations.mean(axis=0)
            cov = np.var(observations, axis=0)

        self.set_emission_params(state, mean, cov)

    def log_emission_prob(self, state: int, observation: np.ndarray) -> float:
        """Compute log P(observation | state) using Gaussian emission.

        Args:
            state: State index.
            observation: Feature vector of shape (feature_dim,).

        Returns:
            Log probability.
        """
        try:
            return float(multivariate_normal.logpdf(
                observation, mean=self.means[state], cov=self.covs[state],
            ))
        except (np.linalg.LinAlgError, ValueError):
            return -1e10

    def forward(self, observations: np.ndarray) -> float:
        """Forward Algorithm — compute log P(observations | HMM).

        Args:
            observations: Array of shape (T, feature_dim).

        Returns:
            Total log-likelihood of the observation sequence.
        """
        T = len(observations)
        if T == 0:
            return -np.inf

        N = self.num_states

        # Initialize: alpha[j] = log pi[j] + log P(o_0 | j)
        log_alpha = np.full(N, -np.inf)
        for j in range(N):
            emit = self.log_emission_prob(j, observations[0])
            log_alpha[j] = self.log_pi[j] + emit

        # Induction
        for t in range(1, T):
            new_log_alpha = np.full(N, -np.inf)
            for j in range(N):
                # Sum over previous states (in log-space)
                log_sum = np.full(N, -np.inf)
                for i in range(N):
                    log_sum[i] = log_alpha[i] + self.log_trans[i][j]
                max_val = np.max(log_sum)
                if max_val == -np.inf:
                    transition_sum = -np.inf
                else:
                    transition_sum = max_val + np.log(np.sum(np.exp(log_sum - max_val)))

                emit = self.log_emission_prob(j, observations[t])
                new_log_alpha[j] = transition_sum + emit

            log_alpha = new_log_alpha

        # Termination: log P(O | lambda) = log sum_j alpha_T(j)
        max_val = np.max(log_alpha)
        if max_val == -np.inf:
            return -np.inf
        return float(max_val + np.log(np.sum(np.exp(log_alpha - max_val))))
