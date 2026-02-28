"""Tests for word-segmented G2P conversion."""

import pytest
from pronun.audio.g2p import text_to_arpabet_by_word, text_to_ipa_by_word


def test_arpabet_by_word_hello_world():
    phonemes, segments = text_to_arpabet_by_word("hello world")
    assert len(phonemes) > 0
    assert len(segments) == 2
    assert segments[0]["word"] == "hello"
    assert segments[1]["word"] == "world"
    # Segments should cover the full phoneme list without gaps
    assert segments[0]["start"] == 0
    assert segments[0]["end"] == segments[1]["start"]
    assert segments[1]["end"] == len(phonemes)


def test_arpabet_by_word_single_word():
    phonemes, segments = text_to_arpabet_by_word("cat")
    assert len(segments) == 1
    assert segments[0]["word"] == "cat"
    assert segments[0]["start"] == 0
    assert segments[0]["end"] == len(phonemes)


def test_arpabet_by_word_three_words():
    phonemes, segments = text_to_arpabet_by_word("I like coffee")
    assert len(segments) == 3
    words = [s["word"] for s in segments]
    assert words == ["I", "like", "coffee"]


def test_arpabet_by_word_no_stress():
    phonemes, _ = text_to_arpabet_by_word("hello")
    for p in phonemes:
        assert not any(c.isdigit() for c in p), f"Stress digit in {p}"


def test_ipa_by_word_hello_world():
    ipa, segments = text_to_ipa_by_word("hello world")
    assert len(ipa) > 0
    assert len(segments) == 2
    assert segments[0]["word"] == "hello"
    assert segments[1]["word"] == "world"
    # All should be IPA, not ARPAbet
    for p in ipa:
        assert not p.isupper(), f"Non-IPA token: {p}"


def test_ipa_by_word_indices_match():
    ipa, segments = text_to_ipa_by_word("thank you")
    for seg in segments:
        assert seg["start"] >= 0
        assert seg["end"] <= len(ipa)
        assert seg["start"] < seg["end"]


def test_ipa_by_word_segment_boundaries_contiguous():
    """Segments should be contiguous — no gaps between words."""
    ipa, segments = text_to_ipa_by_word("I like this coffee")
    for i in range(len(segments) - 1):
        assert segments[i]["end"] == segments[i + 1]["start"]
