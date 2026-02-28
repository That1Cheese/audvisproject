"""Tests for K-means viseme clustering."""

import numpy as np
import pytest
import tempfile
import os
from pronun.visual.viseme.kmeans_viseme import KMeansViseme


def _make_clustered_data(k=5, dim=10, n_per_cluster=50, seed=42):
    """Generate synthetic data with clear cluster structure."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(k, dim) * 10
    data = []
    for center in centers:
        cluster = center + rng.randn(n_per_cluster, dim) * 0.5
        data.append(cluster)
    return np.vstack(data), centers


def test_train_and_predict():
    data, _ = _make_clustered_data(k=5, dim=10)
    model = KMeansViseme(k=5)
    model.train(data)

    predictions = model.predict(data)
    assert predictions.shape == (len(data),)
    assert set(predictions) == {0, 1, 2, 3, 4}


def test_predict_single():
    data, _ = _make_clustered_data(k=3, dim=5)
    model = KMeansViseme(k=3)
    model.train(data)

    vid = model.predict_single(data[0])
    assert isinstance(vid, int)
    assert 0 <= vid < 3


def test_untrained_raises():
    model = KMeansViseme(k=5)
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((10, 5)))
    with pytest.raises(RuntimeError):
        _ = model.centroids
    with pytest.raises(RuntimeError):
        _ = model.inertia


def test_centroids_shape():
    data, _ = _make_clustered_data(k=4, dim=8)
    model = KMeansViseme(k=4)
    model.train(data)

    assert model.centroids.shape == (4, 8)


def test_save_and_load():
    data, _ = _make_clustered_data(k=3, dim=6)
    model = KMeansViseme(k=3)
    model.train(data)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name

    try:
        model.save(path)
        loaded = KMeansViseme.load(path)
        assert loaded.fitted
        assert loaded.k == 3

        # Predictions should match
        orig = model.predict(data[:10])
        load = loaded.predict(data[:10])
        np.testing.assert_array_equal(orig, load)
    finally:
        os.unlink(path)


def test_clusters_correspond_to_distinct_shapes():
    """Verify clusters separate visually distinct groups."""
    data, centers = _make_clustered_data(k=4, dim=10, n_per_cluster=100)
    model = KMeansViseme(k=4)
    model.train(data)

    # Each original cluster should map predominantly to one K-means cluster
    for i in range(4):
        cluster_data = data[i * 100:(i + 1) * 100]
        preds = model.predict(cluster_data)
        # Majority should be same label
        unique, counts = np.unique(preds, return_counts=True)
        majority = counts.max() / counts.sum()
        assert majority > 0.9, f"Cluster {i} not well separated: {majority:.2f}"
