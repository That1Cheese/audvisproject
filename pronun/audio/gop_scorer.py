"""GOP (Goodness of Pronunciation) scoring via CTC self-alignment."""

import numpy as np

from pronun.config import GOP_MAX, GOP_MIN


def _edit_distance_align(predicted: list[str], target: list[str]) -> list[tuple[str | None, str | None]]:
    """Edit-distance alignment between predicted and target phoneme sequences.

    Returns list of (predicted_phoneme, target_phoneme) pairs.
    None indicates insertion/deletion.
    """
    n, m = len(predicted), len(target)
    dp = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if predicted[i - 1] == target[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    # Backtrace
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and (
            predicted[i - 1] == target[j - 1]
            or dp[i][j] == dp[i - 1][j - 1] + 1
        ):
            alignment.append((predicted[i - 1], target[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append((predicted[i - 1], None))
            i -= 1
        else:
            alignment.append((None, target[j - 1]))
            j -= 1

    alignment.reverse()
    return alignment


def _get_frame_boundaries(predicted_ids: list[int], blank_id: int) -> list[tuple[int, int, int]]:
    """Extract frame boundaries for each non-blank predicted token.

    Returns list of (token_id, start_frame, end_frame).
    """
    segments = []
    current_id = None
    start = 0

    for frame, tid in enumerate(predicted_ids):
        if tid != current_id:
            if current_id is not None and current_id != blank_id:
                segments.append((current_id, start, frame))
            current_id = tid
            start = frame

    if current_id is not None and current_id != blank_id:
        segments.append((current_id, start, len(predicted_ids)))

    return segments


def compute_gop(
    log_probs: np.ndarray,
    predicted_ids: list[int],
    predicted_phonemes: list[str],
    target_phonemes: list[str],
    vocab: dict[str, int],
    blank_id: int = 0,
) -> list[dict]:
    """Compute per-phoneme GOP scores.

    Args:
        log_probs: [T, V] log-softmax probabilities.
        predicted_ids: Raw frame-level predicted token IDs (before collapse).
        predicted_phonemes: Collapsed predicted phoneme sequence.
        target_phonemes: Target phoneme sequence from G2P.
        vocab: Token-to-ID mapping.
        blank_id: CTC blank token ID.

    Returns:
        List of dicts with 'phoneme', 'gop_raw', 'gop_score', 'frames'.
    """
    # Get frame boundaries for predicted phonemes
    segments = _get_frame_boundaries(predicted_ids, blank_id)

    # Align predicted to target
    alignment = _edit_distance_align(predicted_phonemes, target_phonemes)

    # Map each target phoneme to its frame ranges
    seg_idx = 0
    results = []

    for pred_p, target_p in alignment:
        if target_p is None:
            # Insertion in predicted — skip this segment
            if pred_p is not None and seg_idx < len(segments):
                seg_idx += 1
            continue

        target_id = vocab.get(target_p)

        if pred_p is not None and seg_idx < len(segments):
            _, start, end = segments[seg_idx]
            seg_idx += 1
        else:
            # Deletion — target phoneme not found in predicted
            results.append({
                "phoneme": target_p,
                "gop_raw": GOP_MIN,
                "gop_score": 0.0,
                "frames": 0,
            })
            continue

        if target_id is None or end <= start:
            results.append({
                "phoneme": target_p,
                "gop_raw": GOP_MIN,
                "gop_score": 0.0,
                "frames": end - start,
            })
            continue

        # GOP = (1/D) * sum of log P(target_phoneme | frame_t)
        duration = end - start
        frame_log_probs = log_probs[start:end, target_id]
        gop_raw = float(np.mean(frame_log_probs))

        # Normalize to 0-100
        gop_score = (gop_raw - GOP_MIN) / (GOP_MAX - GOP_MIN) * 100.0
        gop_score = max(0.0, min(100.0, gop_score))

        results.append({
            "phoneme": target_p,
            "gop_raw": gop_raw,
            "gop_score": gop_score,
            "frames": duration,
        })

    return results


def overall_gop_score(phoneme_scores: list[dict]) -> float:
    """Compute overall GOP score as weighted average by frame count."""
    total_frames = sum(s["frames"] for s in phoneme_scores)
    if total_frames == 0:
        return 0.0
    weighted = sum(s["gop_score"] * s["frames"] for s in phoneme_scores)
    return weighted / total_frames
