"""Human-readable pronunciation tips per phoneme."""

# Per-phoneme improvement advice
PHONEME_TIPS = {
    # Consonants commonly difficult for Mandarin speakers
    "θ": "Place tongue tip between teeth and blow air gently (like 'th' in 'think').",
    "ð": "Place tongue tip between teeth and vibrate vocal cords (like 'th' in 'the').",
    "ɹ": "Curl tongue tip back without touching the roof of mouth. Lips slightly rounded.",
    "l": "Press tongue tip firmly against the ridge behind upper teeth.",
    "v": "Gently bite lower lip with upper teeth and vibrate. Don't use 'w'.",
    "z": "Like 's' but with vocal cord vibration. Feel the buzz.",
    "ʒ": "Like 'sh' but with vocal cord vibration (as in 'measure').",
    "dʒ": "Start with tongue at roof of mouth, release into 'zh' sound.",
    "b": "Lips together, then release with a voiced burst. Distinct from 'p'.",
    "d": "Tongue tip at ridge behind teeth, release with voice. Distinct from 't'.",
    "ɡ": "Back of tongue touches soft palate, release with voice. Distinct from 'k'.",

    # Vowels
    "æ": "Open mouth wide, tongue low and front. Like 'a' in 'cat', not 'e'.",
    "ɪ": "Short, relaxed 'i'. Tongue slightly lower than for 'ee'. As in 'sit'.",
    "ʊ": "Short, relaxed 'u'. Lips slightly rounded. As in 'put'.",
    "ʌ": "Short, open 'uh'. Mouth relaxed and slightly open. As in 'cup'.",
    "ɛ": "Open 'e'. Mouth more open than for 'ay'. As in 'bed'.",
    "ɝ": "R-colored vowel. Curl tongue tip back while saying 'er'. As in 'bird'.",
    "ə": "The schwa — most relaxed vowel. Quick, unstressed 'uh'.",

    # Diphthongs
    "aɪ": "Start with open 'ah', glide smoothly to 'ee'. As in 'buy'.",
    "aʊ": "Start with open 'ah', glide smoothly to 'oo'. As in 'how'.",
    "oʊ": "Start with 'oh', glide to a slight 'oo'. Keep lips rounded.",
    "eɪ": "Start with 'eh', glide to 'ee'. As in 'say'.",
    "ɔɪ": "Start with 'aw', glide to 'ee'. As in 'boy'.",
}

# Score thresholds for feedback levels
EXCELLENT_THRESHOLD = 85
GOOD_THRESHOLD = 70
FAIR_THRESHOLD = 50


def get_phoneme_tip(phoneme: str) -> str:
    """Get improvement tip for a specific phoneme."""
    return PHONEME_TIPS.get(phoneme, f"Practice the /{phoneme}/ sound carefully.")


def generate_feedback(phoneme_scores: list[dict]) -> list[dict]:
    """Generate per-phoneme feedback with tips.

    Args:
        phoneme_scores: List of dicts with 'phoneme' and 'combined_score'
                        (or 'gop_score').

    Returns:
        List of dicts with 'phoneme', 'score', 'level', 'tip'.
    """
    feedback = []
    for ps in phoneme_scores:
        score = ps.get("combined_score", ps.get("gop_score", 0))
        phoneme = ps["phoneme"]

        if score >= EXCELLENT_THRESHOLD:
            level = "excellent"
            tip = None
        elif score >= GOOD_THRESHOLD:
            level = "good"
            tip = get_phoneme_tip(phoneme)
        elif score >= FAIR_THRESHOLD:
            level = "fair"
            tip = get_phoneme_tip(phoneme)
        else:
            level = "needs_work"
            tip = get_phoneme_tip(phoneme)

        feedback.append({
            "phoneme": phoneme,
            "score": score,
            "level": level,
            "tip": tip,
        })

    return feedback


def generate_word_feedback(word_scores: list[dict]) -> list[dict]:
    """Generate per-word feedback with levels.

    Args:
        word_scores: List of dicts with 'word' and 'score'.

    Returns:
        List of dicts with 'word', 'score', 'level'.
    """
    feedback = []
    for ws in word_scores:
        score = ws["score"]
        if score >= EXCELLENT_THRESHOLD:
            level = "excellent"
        elif score >= GOOD_THRESHOLD:
            level = "good"
        elif score >= FAIR_THRESHOLD:
            level = "fair"
        else:
            level = "needs_work"
        feedback.append({
            "word": ws["word"],
            "score": score,
            "level": level,
        })
    return feedback


def overall_feedback(score: float, word_scores: list[dict] = None) -> str:
    """Generate overall feedback message based on total score.

    If word_scores are provided, names problematic words in the message.
    """
    if score >= EXCELLENT_THRESHOLD:
        msg = "Excellent pronunciation! Keep it up."
    elif score >= GOOD_THRESHOLD:
        msg = "Good pronunciation. Minor improvements possible."
    elif score >= FAIR_THRESHOLD:
        msg = "Fair pronunciation. Focus on the highlighted phonemes."
    else:
        msg = "Needs practice. Review the tips for each phoneme below."

    if word_scores:
        weak_words = [
            ws["word"] for ws in word_scores
            if ws["score"] < GOOD_THRESHOLD
        ]
        if weak_words:
            quoted = ", ".join(f"'{w}'" for w in weak_words)
            msg += f" Focus on: {quoted}"

    return msg
