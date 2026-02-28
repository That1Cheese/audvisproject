"""Mode A vs Mode B comparison output."""

import numpy as np


def compare_modes(
    score_a: dict,
    score_b: dict,
    viseme_seq_a: np.ndarray | list,
    viseme_seq_b: list[int],
) -> dict:
    """Compare Mode A (K-means) vs Mode B (Lee's map) results.

    Args:
        score_a: Visual scorer result dict for Mode A.
        score_b: Visual scorer result dict for Mode B.
        viseme_seq_a: Predicted viseme sequence from K-means.
        viseme_seq_b: Target viseme sequence from Lee's map.

    Returns:
        Comparison dict with scores, differences, and analysis.
    """
    ll_diff = score_a["log_likelihood_norm"] - score_b["log_likelihood_norm"]
    score_diff = score_a["score"] - score_b["score"]

    return {
        "mode_a": {
            "name": "K-means (data-driven)",
            "score": score_a["score"],
            "log_likelihood": score_a["log_likelihood"],
            "log_likelihood_norm": score_a["log_likelihood_norm"],
            "viseme_count": len(set(viseme_seq_a)) if len(viseme_seq_a) > 0 else 0,
        },
        "mode_b": {
            "name": "Lee's Viseme Map (linguistic)",
            "score": score_b["score"],
            "log_likelihood": score_b["log_likelihood"],
            "log_likelihood_norm": score_b["log_likelihood_norm"],
            "viseme_count": len(set(viseme_seq_b)),
        },
        "comparison": {
            "likelihood_difference": ll_diff,
            "score_difference": score_diff,
            "preferred_mode": "A" if score_a["score"] > score_b["score"] else "B",
        },
    }


def build_confusion_matrix(
    seq_a: np.ndarray | list,
    seq_b: list[int],
    num_visemes_a: int,
    num_visemes_b: int,
) -> np.ndarray:
    """Build confusion matrix between Mode A and Mode B viseme assignments.

    Requires both sequences to have the same length (frame-aligned).

    Args:
        seq_a: K-means viseme IDs per frame.
        seq_b: Lee viseme IDs per frame (requires frame-level alignment).
        num_visemes_a: Number of K-means clusters.
        num_visemes_b: Number of Lee visemes (13).

    Returns:
        Confusion matrix of shape (num_visemes_a, num_visemes_b).
    """
    min_len = min(len(seq_a), len(seq_b))
    matrix = np.zeros((num_visemes_a, num_visemes_b), dtype=int)

    for i in range(min_len):
        a = int(seq_a[i])
        b = int(seq_b[i])
        if 0 <= a < num_visemes_a and 0 <= b < num_visemes_b:
            matrix[a][b] += 1

    return matrix


def format_comparison(comparison: dict) -> str:
    """Format comparison results as a readable string."""
    lines = [
        "=== Mode Comparison ===",
        "",
        f"Mode A ({comparison['mode_a']['name']}):",
        f"  Score: {comparison['mode_a']['score']:.1f}/100",
        f"  Log-likelihood (norm): {comparison['mode_a']['log_likelihood_norm']:.4f}",
        f"  Unique visemes: {comparison['mode_a']['viseme_count']}",
        "",
        f"Mode B ({comparison['mode_b']['name']}):",
        f"  Score: {comparison['mode_b']['score']:.1f}/100",
        f"  Log-likelihood (norm): {comparison['mode_b']['log_likelihood_norm']:.4f}",
        f"  Unique visemes: {comparison['mode_b']['viseme_count']}",
        "",
        f"Likelihood difference: {comparison['comparison']['likelihood_difference']:.4f}",
        f"Score difference: {comparison['comparison']['score_difference']:.1f}",
        f"Preferred mode: {comparison['comparison']['preferred_mode']}",
    ]
    return "\n".join(lines)
