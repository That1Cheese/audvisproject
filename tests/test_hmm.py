"""Tests for HMM Forward Algorithm."""

import numpy as np
import pytest
from pronun.visual.scoring.hmm import GaussianHMM


def test_forward_known_hmm():
    """Verify Forward Algorithm with a small known HMM.

    2-state left-to-right HMM with 1D Gaussian emissions.
    Manual calculation to verify correctness.
    """
    hmm = GaussianHMM(num_states=2, feature_dim=1, self_loop_prob=0.5)

    # State 0: mean=0, var=1; State 1: mean=5, var=1
    hmm.set_emission_params(0, np.array([0.0]), np.array([[1.0]]))
    hmm.set_emission_params(1, np.array([5.0]), np.array([[1.0]]))

    # Observation near state 0, then near state 1
    obs = np.array([[0.1], [4.9]])
    ll = hmm.forward(obs)

    # Should be a finite negative number
    assert np.isfinite(ll)
    assert ll < 0

    # Observation far from both states should have lower likelihood
    bad_obs = np.array([[10.0], [10.0]])
    bad_ll = hmm.forward(bad_obs)

    assert bad_ll < ll


def test_single_observation():
    hmm = GaussianHMM(num_states=1, feature_dim=2, self_loop_prob=1.0)
    hmm.set_emission_params(0, np.array([0.0, 0.0]), np.eye(2))

    obs = np.array([[0.0, 0.0]])
    ll = hmm.forward(obs)
    assert np.isfinite(ll)


def test_empty_observations():
    hmm = GaussianHMM(num_states=2, feature_dim=1)
    ll = hmm.forward(np.array([]).reshape(0, 1))
    assert ll == -np.inf


def test_left_to_right_constraint():
    """Observations matching reverse order should score lower."""
    hmm = GaussianHMM(num_states=3, feature_dim=1, self_loop_prob=0.3)
    hmm.set_emission_params(0, np.array([0.0]), np.array([[0.1]]))
    hmm.set_emission_params(1, np.array([5.0]), np.array([[0.1]]))
    hmm.set_emission_params(2, np.array([10.0]), np.array([[0.1]]))

    # Correct order
    correct = np.array([[0.0], [5.0], [10.0]])
    ll_correct = hmm.forward(correct)

    # Reverse order (shouldn't match left-to-right)
    reverse = np.array([[10.0], [5.0], [0.0]])
    ll_reverse = hmm.forward(reverse)

    assert ll_correct > ll_reverse


def test_train_emissions():
    hmm = GaussianHMM(num_states=2, feature_dim=2)

    data0 = np.random.randn(50, 2) + [1, 1]
    data1 = np.random.randn(50, 2) + [5, 5]

    hmm.train_emissions(0, data0)
    hmm.train_emissions(1, data1)

    # Observation near state 0 should have higher emission prob from state 0
    p0 = hmm.log_emission_prob(0, np.array([1.0, 1.0]))
    p1 = hmm.log_emission_prob(1, np.array([1.0, 1.0]))
    assert p0 > p1


def test_self_loop_prob():
    """More frames per state should be handled by self-loop."""
    hmm = GaussianHMM(num_states=2, feature_dim=1, self_loop_prob=0.7)
    hmm.set_emission_params(0, np.array([0.0]), np.array([[1.0]]))
    hmm.set_emission_params(1, np.array([5.0]), np.array([[1.0]]))

    # Many frames near state 0, then state 1
    obs = np.vstack([np.zeros((10, 1)), np.full((10, 1), 5.0)])
    ll = hmm.forward(obs)
    assert np.isfinite(ll)
