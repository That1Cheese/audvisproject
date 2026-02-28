"""Tests for GOP scoring."""

import numpy as np
import pytest
from pronun.audio.gop_scorer import compute_gop, overall_gop_score


def _make_log_probs(T, V, target_ids, quality="good"):
    """Create synthetic log-prob matrix.

    For 'good' quality, target phonemes have high probability.
    For 'bad' quality, target phonemes have low probability.
    """
    log_probs = np.full((T, V), -10.0)  # low baseline

    if quality == "good":
        # High probability for target phonemes at their frames
        frames_per = T // max(len(target_ids), 1)
        for i, tid in enumerate(target_ids):
            start = i * frames_per
            end = min(start + frames_per, T)
            log_probs[start:end, tid] = -0.2
    else:
        # Low probability everywhere
        log_probs[:, :] = -5.0

    return log_probs


def test_good_pronunciation_scores_higher():
    V = 50  # vocab size
    T = 30
    target_phonemes = ["a", "b", "c"]
    vocab = {"a": 1, "b": 2, "c": 3, "<pad>": 0}

    # Predicted IDs: each phoneme spans 10 frames
    predicted_ids = [0]*2 + [1]*8 + [0]*2 + [2]*8 + [0]*2 + [3]*8
    predicted_phonemes = ["a", "b", "c"]

    good_lp = _make_log_probs(T, V, [1, 2, 3], "good")
    bad_lp = _make_log_probs(T, V, [1, 2, 3], "bad")

    good_scores = compute_gop(good_lp, predicted_ids, predicted_phonemes,
                               target_phonemes, vocab, blank_id=0)
    bad_scores = compute_gop(bad_lp, predicted_ids, predicted_phonemes,
                              target_phonemes, vocab, blank_id=0)

    good_overall = overall_gop_score(good_scores)
    bad_overall = overall_gop_score(bad_scores)

    assert good_overall > bad_overall


def test_gop_score_range():
    V = 10
    T = 20
    target_phonemes = ["x"]
    vocab = {"x": 1, "<pad>": 0}
    predicted_ids = [0]*5 + [1]*10 + [0]*5
    predicted_phonemes = ["x"]

    log_probs = np.full((T, V), -2.0)
    log_probs[5:15, 1] = -0.5

    scores = compute_gop(log_probs, predicted_ids, predicted_phonemes,
                          target_phonemes, vocab, blank_id=0)

    for s in scores:
        assert 0.0 <= s["gop_score"] <= 100.0


def test_overall_gop_score_empty():
    assert overall_gop_score([]) == 0.0


def test_missing_phoneme_gets_zero():
    V = 10
    T = 10
    target_phonemes = ["x", "y"]
    vocab = {"x": 1, "<pad>": 0}  # "y" not in vocab
    predicted_ids = [0]*3 + [1]*7
    predicted_phonemes = ["x"]

    log_probs = np.full((T, V), -2.0)

    scores = compute_gop(log_probs, predicted_ids, predicted_phonemes,
                          target_phonemes, vocab, blank_id=0)

    # y should get a score of 0
    y_score = [s for s in scores if s["phoneme"] == "y"]
    assert len(y_score) == 1
    assert y_score[0]["gop_score"] == 0.0
