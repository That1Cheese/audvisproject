"""Tests for GRID training pipeline: alignment parsing, data collection, and HMM integration."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pronun.training.train_emissions import (
    VisemeDataCollector,
    parse_grid_align,
    train_from_collector,
    _word_to_viseme_ids,
)
from pronun.visual.scoring.emission_store import EmissionStore
from pronun.visual.scoring.visual_scorer import VisualScorer


# ---------------------------------------------------------------------------
# parse_grid_align
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_align_file(tmp_path):
    content = (
        "0 23750 sil\n"
        "23750 29500 bin\n"
        "29500 34000 blue\n"
        "34000 35500 at\n"
        "35500 41000 f\n"
        "41000 47250 two\n"
        "47250 53000 now\n"
        "53000 74500 sil\n"
    )
    align_path = tmp_path / "test.align"
    align_path.write_text(content)
    return align_path


def test_parse_grid_align(sample_align_file):
    segments = parse_grid_align(sample_align_file)
    # Should exclude sil segments
    assert all(s["word"] != "sil" for s in segments)
    assert len(segments) == 6  # bin, blue, at, f, two, now

    # Check first word: bin at timestamps 23750-29500
    assert segments[0]["word"] == "bin"
    assert segments[0]["start_frame"] == 24  # round(23750/1000)
    assert segments[0]["end_frame"] == 30    # round(29500/1000) (actually 29.5 rounds to 30)


def test_parse_grid_align_frame_boundaries(sample_align_file):
    segments = parse_grid_align(sample_align_file)
    # Verify no gaps or overlaps between consecutive segments
    for i in range(len(segments) - 1):
        assert segments[i]["end_frame"] <= segments[i + 1]["start_frame"] + 1


def test_parse_grid_align_empty(tmp_path):
    empty = tmp_path / "empty.align"
    empty.write_text("")
    assert parse_grid_align(empty) == []


def test_parse_grid_align_only_silence(tmp_path):
    sil_only = tmp_path / "sil.align"
    sil_only.write_text("0 50000 sil\n50000 74500 sil\n")
    assert parse_grid_align(sil_only) == []


# ---------------------------------------------------------------------------
# _word_to_viseme_ids
# ---------------------------------------------------------------------------

def test_word_to_viseme_ids():
    visemes = _word_to_viseme_ids("bin")
    assert isinstance(visemes, list)
    assert all(isinstance(v, int) for v in visemes)
    assert all(0 <= v <= 12 for v in visemes)
    assert len(visemes) > 0  # "bin" → B, IH, N → viseme IDs 1, 7, 3


# ---------------------------------------------------------------------------
# VisemeDataCollector
# ---------------------------------------------------------------------------

FEATURE_DIM = 10  # Smaller than real 248 for fast tests


def _make_features(n_frames: int, dim: int = FEATURE_DIM) -> list[np.ndarray]:
    rng = np.random.default_rng(42)
    return [rng.standard_normal(dim) for _ in range(n_frames)]


def test_collector_add_grid_sample():
    collector = VisemeDataCollector()
    features = _make_features(75)
    segments = [
        {"word": "bin", "start_frame": 24, "end_frame": 30},
        {"word": "blue", "start_frame": 30, "end_frame": 34},
    ]

    collector.add_grid_sample(features, segments)
    summary = collector.summary()
    # Should have collected observations for some viseme IDs
    assert len(summary) > 0
    assert all(count > 0 for count in summary.values())


def test_collector_frame_assignment():
    """Verify frames are correctly distributed across visemes within a word."""
    collector = VisemeDataCollector()

    # Create distinct features so we can identify them
    dim = 4
    features = [np.full(dim, i, dtype=float) for i in range(50)]
    segments = [
        {"word": "at", "start_frame": 10, "end_frame": 20},
    ]

    collector.add_grid_sample(features, segments)
    summary = collector.summary()
    total_obs = sum(summary.values())
    # "at" spans frames 10-19 = 10 frames, all should be assigned
    assert total_obs == 10


def test_collector_add_sample():
    collector = VisemeDataCollector()
    features = _make_features(50)
    collector.add_sample(features, "hello world")
    summary = collector.summary()
    assert len(summary) > 0


def test_collector_get_observations_empty():
    collector = VisemeDataCollector()
    assert collector.get_observations(99) is None


def test_collector_get_observations_shape():
    collector = VisemeDataCollector()
    features = _make_features(75)
    segments = [
        {"word": "bin", "start_frame": 24, "end_frame": 30},
    ]
    collector.add_grid_sample(features, segments)

    summary = collector.summary()
    for vid, count in summary.items():
        obs = collector.get_observations(vid)
        assert obs is not None
        assert obs.shape == (count, FEATURE_DIM)


def test_collector_handles_out_of_range_frames():
    """Frames beyond feature length should be silently ignored."""
    collector = VisemeDataCollector()
    features = _make_features(10)
    segments = [
        {"word": "bin", "start_frame": 5, "end_frame": 20},  # end > len(features)
    ]
    collector.add_grid_sample(features, segments)
    summary = collector.summary()
    total = sum(summary.values())
    assert total == 5  # Only frames 5-9


# ---------------------------------------------------------------------------
# train_from_collector → EmissionStore
# ---------------------------------------------------------------------------

def test_train_from_collector():
    collector = VisemeDataCollector()
    rng = np.random.default_rng(42)

    # Simulate enough data for a few visemes
    for vid in [1, 3, 7]:
        for _ in range(20):
            collector._observations[vid].append(rng.standard_normal(FEATURE_DIM))

    store = train_from_collector(collector)
    assert isinstance(store, EmissionStore)
    assert len(store) == 3
    assert 1 in store
    assert 3 in store
    assert 7 in store

    for vid in [1, 3, 7]:
        mean, cov = store.get_params(vid)
        assert mean.shape == (FEATURE_DIM,)
        assert cov.shape == (FEATURE_DIM,)


def test_train_from_collector_skips_insufficient_data():
    collector = VisemeDataCollector()
    # Only 1 observation — not enough to compute covariance
    collector._observations[5].append(np.zeros(FEATURE_DIM))
    store = train_from_collector(collector)
    assert 5 not in store


# ---------------------------------------------------------------------------
# Integration: trained store → build_hmm → forward → reasonable L_norm
# ---------------------------------------------------------------------------

def test_trained_hmm_gives_reasonable_log_likelihood():
    """With trained emissions, L_norm should be much better than -300."""
    dim = FEATURE_DIM
    rng = np.random.default_rng(42)

    # Create synthetic emission data for viseme IDs 1 and 7
    store = EmissionStore()
    for vid in [1, 7]:
        obs = rng.standard_normal((50, dim))
        mean = obs.mean(axis=0)
        cov = np.cov(obs, rowvar=False)
        store.set_params(vid, mean, cov)

    scorer = VisualScorer()
    viseme_seq = [1, 7]  # Simple 2-state sequence
    hmm = scorer.build_hmm(viseme_seq, store.to_dict(), dim)

    # Generate test observations from same distribution
    test_obs = rng.standard_normal((20, dim))
    log_ll = hmm.forward(test_obs)
    l_norm = log_ll / len(test_obs)

    # With trained emissions matching the data distribution,
    # L_norm should be much better than -300 (untrained default)
    assert l_norm > -100, f"L_norm = {l_norm} is too low, emissions may not be loaded"


def test_build_hmm_backward_compatible_with_raw_arrays():
    """build_hmm still works with raw ndarray observations (not tuples)."""
    dim = FEATURE_DIM
    rng = np.random.default_rng(42)

    scorer = VisualScorer()
    raw_obs = {1: rng.standard_normal((30, dim))}
    hmm = scorer.build_hmm([1], raw_obs, dim)

    test_obs = rng.standard_normal((10, dim))
    log_ll = hmm.forward(test_obs)
    assert np.isfinite(log_ll)
