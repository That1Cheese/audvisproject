"""Tests for EmissionStore save/load and dict interface."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pronun.visual.scoring.emission_store import EmissionStore


@pytest.fixture
def store():
    s = EmissionStore()
    rng = np.random.default_rng(42)
    for vid in range(5):
        dim = 10
        mean = rng.standard_normal(dim)
        cov = np.ones(dim) * (vid + 1)
        s.set_params(vid, mean, cov)
    return s


def test_set_and_get_params(store):
    params = store.get_params(0)
    assert params is not None
    mean, cov = params
    assert mean.shape == (10,)
    assert cov.shape == (10,)


def test_missing_viseme_returns_none(store):
    assert store.get_params(99) is None


def test_contains(store):
    assert 0 in store
    assert 99 not in store


def test_len(store):
    assert len(store) == 5


def test_viseme_ids(store):
    assert store.viseme_ids() == [0, 1, 2, 3, 4]


def test_to_dict(store):
    d = store.to_dict()
    assert isinstance(d, dict)
    assert set(d.keys()) == {0, 1, 2, 3, 4}
    for vid, (mean, cov) in d.items():
        assert mean.shape == (10,)
        assert cov.shape == (10,)


def test_save_load_roundtrip(store):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_emissions.npz"
        store.save(path)
        assert path.exists()

        loaded = EmissionStore.from_file(path)
        assert len(loaded) == len(store)
        assert loaded.viseme_ids() == store.viseme_ids()

        for vid in store.viseme_ids():
            orig_mean, orig_cov = store.get_params(vid)
            load_mean, load_cov = loaded.get_params(vid)
            np.testing.assert_array_almost_equal(orig_mean, load_mean)
            np.testing.assert_array_almost_equal(orig_cov, load_cov)


def test_from_file_missing_raises():
    with pytest.raises(FileNotFoundError):
        EmissionStore.from_file("/nonexistent/path/emissions.npz")


def test_empty_store():
    store = EmissionStore()
    assert len(store) == 0
    assert store.viseme_ids() == []
    assert store.to_dict() == {}
