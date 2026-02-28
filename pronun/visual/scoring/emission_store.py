"""Persistent storage for trained viseme HMM emission parameters."""

from pathlib import Path

import numpy as np

from pronun.config import EMISSION_STORE_PATH


class EmissionStore:
    """Stores per-viseme Gaussian emission parameters (mean, variance).

    Each viseme ID (0-12) maps to a (mean, cov) tuple where:
        - mean: ndarray of shape (feature_dim,)
        - cov: ndarray of shape (feature_dim,) — diagonal variance vector
    """

    def __init__(self):
        self._params: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def set_params(self, viseme_id: int, mean: np.ndarray, cov: np.ndarray):
        """Set Gaussian parameters for a viseme ID."""
        self._params[viseme_id] = (np.asarray(mean), np.asarray(cov))

    def get_params(self, viseme_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Get (mean, cov) for a viseme ID, or None if not trained."""
        return self._params.get(viseme_id)

    def to_dict(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Return dict mapping viseme_id → (mean, cov) for build_hmm()."""
        return dict(self._params)

    def viseme_ids(self) -> list[int]:
        """Return sorted list of trained viseme IDs."""
        return sorted(self._params.keys())

    def __len__(self) -> int:
        return len(self._params)

    def __contains__(self, viseme_id: int) -> bool:
        return viseme_id in self._params

    def save(self, path: str | Path | None = None):
        """Save emission parameters to disk via np.savez.

        File format: keys "mean_{id}" and "cov_{id}" for each viseme,
        plus "viseme_ids" listing all stored IDs.
        """
        path = Path(path) if path else EMISSION_STORE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays = {"viseme_ids": np.array(sorted(self._params.keys()))}
        for vid, (mean, cov) in self._params.items():
            arrays[f"mean_{vid}"] = mean
            arrays[f"cov_{vid}"] = cov

        # np.savez auto-appends .npz, so strip it to avoid double extension
        save_path = str(path)
        if save_path.endswith(".npz"):
            save_path = save_path[:-4]
        np.savez(save_path, **arrays)

    def load(self, path: str | Path | None = None):
        """Load emission parameters from disk."""
        path = Path(path) if path else EMISSION_STORE_PATH
        data = np.load(str(path))

        self._params.clear()
        viseme_ids = data["viseme_ids"]
        for vid in viseme_ids:
            vid = int(vid)
            cov = data[f"cov_{vid}"]
            if cov.ndim == 2:
                cov = np.diag(cov)
            self._params[vid] = (data[f"mean_{vid}"], cov)

    @classmethod
    def from_file(cls, path: str | Path | None = None) -> "EmissionStore":
        """Load an EmissionStore from disk. Raises FileNotFoundError if missing."""
        store = cls()
        store.load(path)
        return store
