"""Mode A: Data-driven viseme learning via K-means clustering."""

import numpy as np
from sklearn.cluster import KMeans
import joblib

from pronun.config import KMEANS_K, KMEANS_RANDOM_STATE


class KMeansViseme:
    """K-means clustering-based viseme learner.

    Clusters mouth shape feature vectors into K viseme categories.
    Each cluster centroid represents a canonical mouth shape (viseme).
    """

    def __init__(self, k: int = KMEANS_K, random_state: int = KMEANS_RANDOM_STATE):
        self.k = k
        self.model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        self.fitted = False

    def train(self, features: np.ndarray) -> "KMeansViseme":
        """Train K-means on feature vectors.

        Args:
            features: Array of shape (num_samples, feature_dim).
                      Each row is a feature vector from one video frame.

        Returns:
            self for chaining.
        """
        self.model.fit(features)
        self.fitted = True
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Assign viseme IDs to feature vectors.

        Args:
            features: Array of shape (num_frames, feature_dim).

        Returns:
            Array of viseme IDs, shape (num_frames,).
        """
        if not self.fitted:
            raise RuntimeError("KMeansViseme must be trained before prediction")
        return self.model.predict(features)

    def predict_single(self, feature: np.ndarray) -> int:
        """Assign viseme ID to a single feature vector."""
        return int(self.predict(feature.reshape(1, -1))[0])

    @property
    def centroids(self) -> np.ndarray:
        """Cluster centroids, shape (K, feature_dim)."""
        if not self.fitted:
            raise RuntimeError("KMeansViseme must be trained first")
        return self.model.cluster_centers_

    @property
    def inertia(self) -> float:
        """Sum of squared distances to closest centroid (training loss)."""
        if not self.fitted:
            raise RuntimeError("KMeansViseme must be trained first")
        return self.model.inertia_

    def save(self, path: str):
        """Save trained model to disk."""
        if not self.fitted:
            raise RuntimeError("Cannot save untrained model")
        joblib.dump({"k": self.k, "model": self.model}, path)

    @classmethod
    def load(cls, path: str) -> "KMeansViseme":
        """Load trained model from disk."""
        data = joblib.load(path)
        instance = cls(k=data["k"])
        instance.model = data["model"]
        instance.fitted = True
        return instance
