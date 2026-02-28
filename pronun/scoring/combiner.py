"""Weighted audio + visual score combination."""

from pronun.config import (
    AMBIGUOUS_AUDIO_WEIGHT,
    AMBIGUOUS_VISUAL_WEIGHT,
    DEFAULT_AUDIO_WEIGHT,
    DEFAULT_VISUAL_WEIGHT,
    DISTINCTIVE_AUDIO_WEIGHT,
    DISTINCTIVE_VISUAL_WEIGHT,
    VISUAL_DISTINCTIVE_PHONEMES,
)


def combine_scores(
    audio_score: float,
    visual_score: float | None,
    audio_weight: float = DEFAULT_AUDIO_WEIGHT,
    visual_weight: float = DEFAULT_VISUAL_WEIGHT,
) -> float:
    """Combine audio and visual scores with given weights.

    Falls back to audio-only if visual_score is None (no webcam).

    Returns:
        Combined score in range 0-100.
    """
    if visual_score is None:
        return audio_score

    combined = audio_weight * audio_score + visual_weight * visual_score
    return max(0.0, min(100.0, combined))


def adaptive_combine(
    phoneme_audio_scores: list[dict],
    visual_score: float | None,
) -> list[dict]:
    """Per-phoneme adaptive weighting based on visual distinctiveness.

    Visually distinctive phonemes (bilabial, labiodental, rounded)
    get higher visual weight. Ambiguous phonemes rely more on audio.

    Args:
        phoneme_audio_scores: List of dicts with 'phoneme' and 'gop_score'.
        visual_score: Overall visual score (applied uniformly per-phoneme
                      since per-phoneme visual scoring requires per-phoneme
                      alignment which isn't always available).

    Returns:
        List of dicts with 'phoneme', 'audio_score', 'visual_score',
        'combined_score', 'audio_weight', 'visual_weight'.
    """
    results = []
    for ps in phoneme_audio_scores:
        phoneme = ps["phoneme"].upper().rstrip("012 ")

        if visual_score is None:
            combined = ps["gop_score"]
            aw, vw = 1.0, 0.0
        elif phoneme in VISUAL_DISTINCTIVE_PHONEMES:
            aw, vw = DISTINCTIVE_AUDIO_WEIGHT, DISTINCTIVE_VISUAL_WEIGHT
            combined = aw * ps["gop_score"] + vw * visual_score
        else:
            aw, vw = AMBIGUOUS_AUDIO_WEIGHT, AMBIGUOUS_VISUAL_WEIGHT
            combined = aw * ps["gop_score"] + vw * visual_score

        results.append({
            "phoneme": ps["phoneme"],
            "audio_score": ps["gop_score"],
            "visual_score": visual_score,
            "combined_score": max(0.0, min(100.0, combined)),
            "audio_weight": aw,
            "visual_weight": vw,
        })

    return results
