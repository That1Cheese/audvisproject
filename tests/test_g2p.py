"""Tests for G2P conversion."""

import pytest
from pronun.audio.g2p import text_to_arpabet, text_to_ipa, text_to_visemes, arpabet_to_ipa


def test_text_to_arpabet_hello():
    result = text_to_arpabet("hello")
    assert len(result) > 0
    # "hello" should contain HH, AH/EH, L, OW
    assert "HH" in result


def test_text_to_arpabet_strips_stress():
    result = text_to_arpabet("cat")
    for p in result:
        assert not any(c.isdigit() for c in p), f"Stress digit in {p}"


def test_text_to_ipa_hello():
    result = text_to_ipa("hello")
    assert len(result) > 0
    assert "h" in result  # HH → h


def test_text_to_visemes_hello():
    result = text_to_visemes("hello")
    assert len(result) > 0
    # HH maps to viseme 3 (K group)
    assert 3 in result


def test_arpabet_to_ipa_known():
    assert arpabet_to_ipa("P") == "p"
    assert arpabet_to_ipa("TH") == "θ"
    assert arpabet_to_ipa("IY") == "iː"
    assert arpabet_to_ipa("AE") == "æ"


def test_arpabet_to_ipa_with_stress():
    assert arpabet_to_ipa("IY1") == "iː"
    assert arpabet_to_ipa("AH0") == "ʌ"


def test_text_to_ipa_produces_valid_ipa():
    result = text_to_ipa("pronunciation")
    assert len(result) > 5  # reasonable number of phonemes
    # All should be IPA strings, not ARPAbet
    for p in result:
        assert not p.isupper(), f"Non-IPA token: {p}"
