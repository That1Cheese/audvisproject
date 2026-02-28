"""Grapheme-to-phoneme conversion with ARPAbet-to-IPA mapping."""

import nltk
from g2p_en import G2p

from pronun.data.lee_map import arpabet_to_viseme

_g2p = None
_nltk_ready = False


def _ensure_nltk_data():
    global _nltk_ready
    if _nltk_ready:
        return
    for resource in ("averaged_perceptron_tagger_eng", "cmudict"):
        try:
            nltk.data.find(f"taggers/{resource}" if "tagger" in resource else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
    _nltk_ready = True


def _get_g2p() -> G2p:
    global _g2p
    if _g2p is None:
        _ensure_nltk_data()
        _g2p = G2p()
    return _g2p


# ARPAbet (without stress) → IPA mapping for wav2vec2 compatibility
ARPABET_TO_IPA = {
    # Consonants
    "P": "p", "B": "b", "T": "t", "D": "d", "K": "k", "G": "ɡ",
    "CH": "tʃ", "JH": "dʒ",
    "F": "f", "V": "v", "TH": "θ", "DH": "ð",
    "S": "s", "Z": "z", "SH": "ʃ", "ZH": "ʒ",
    "HH": "h", "M": "m", "N": "n", "NG": "ŋ",
    "L": "l", "R": "ɹ", "Y": "j", "W": "w",
    # Vowels
    "IY": "iː", "IH": "ɪ", "EY": "eɪ", "EH": "ɛ", "AE": "æ",
    "AA": "ɑː", "AO": "ɔː", "AH": "ʌ", "UH": "ʊ", "UW": "uː",
    "ER": "ɝ", "AX": "ə",
    # Diphthongs
    "AY": "aɪ", "AW": "aʊ", "OY": "ɔɪ", "OW": "oʊ",
}


def text_to_arpabet(text: str) -> list[str]:
    """Convert text to ARPAbet phoneme sequence (no stress digits)."""
    g2p = _get_g2p()
    raw = g2p(text)
    phonemes = []
    for p in raw:
        p = p.strip()
        if not p or p == " ":
            continue
        clean = p.rstrip("012")
        if clean:
            phonemes.append(clean)
    return phonemes


def text_to_ipa(text: str) -> list[str]:
    """Convert text to IPA phoneme sequence for wav2vec2."""
    arpabet = text_to_arpabet(text)
    ipa = []
    for p in arpabet:
        if p in ARPABET_TO_IPA:
            ipa.append(ARPABET_TO_IPA[p])
    return ipa


def text_to_visemes(text: str) -> list[int]:
    """Convert text to Lee viseme ID sequence."""
    g2p = _get_g2p()
    raw = g2p(text)
    visemes = []
    for p in raw:
        p = p.strip()
        if not p or p == " ":
            continue
        vid = arpabet_to_viseme(p)
        visemes.append(vid)
    return visemes


def text_to_arpabet_by_word(text: str) -> tuple[list[str], list[dict]]:
    """Convert text to ARPAbet with per-word segment boundaries.

    Returns:
        (flat_phonemes, word_segments) where word_segments is a list of
        {"word": str, "start": int, "end": int} marking each word's
        phoneme range in the flat list.
    """
    g2p = _get_g2p()
    words = text.strip().split()
    flat_phonemes: list[str] = []
    word_segments: list[dict] = []

    for word in words:
        raw = g2p(word)
        start = len(flat_phonemes)
        for p in raw:
            p = p.strip()
            if not p or p == " ":
                continue
            clean = p.rstrip("012")
            if clean:
                flat_phonemes.append(clean)
        end = len(flat_phonemes)
        if end > start:
            word_segments.append({"word": word, "start": start, "end": end})

    return flat_phonemes, word_segments


def text_to_ipa_by_word(text: str) -> tuple[list[str], list[dict]]:
    """Convert text to IPA with per-word segment boundaries.

    Returns:
        (flat_ipa, word_segments) where word_segments is a list of
        {"word": str, "start": int, "end": int} marking each word's
        phoneme range in the flat list.
    """
    arpabet, word_segments = text_to_arpabet_by_word(text)
    flat_ipa: list[str] = []
    new_segments: list[dict] = []

    for seg in word_segments:
        start = len(flat_ipa)
        for i in range(seg["start"], seg["end"]):
            p = arpabet[i]
            if p in ARPABET_TO_IPA:
                flat_ipa.append(ARPABET_TO_IPA[p])
        end = len(flat_ipa)
        if end > start:
            new_segments.append({"word": seg["word"], "start": start, "end": end})

    return flat_ipa, new_segments


def arpabet_to_ipa(phoneme: str) -> str:
    """Convert a single ARPAbet phoneme to IPA."""
    clean = phoneme.rstrip("012")
    return ARPABET_TO_IPA.get(clean, clean)
