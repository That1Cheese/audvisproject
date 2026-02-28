"""Tests for Lee's viseme mapping."""

import pytest
from pronun.visual.viseme.lee_viseme import LeeViseme
from pronun.data.lee_map import arpabet_to_viseme, NUM_VISEMES


def test_bilabial_group():
    """P, B, M should all map to viseme 1."""
    assert arpabet_to_viseme("P") == 1
    assert arpabet_to_viseme("B") == 1
    assert arpabet_to_viseme("M") == 1


def test_labiodental_group():
    """F, V should map to viseme 5."""
    assert arpabet_to_viseme("F") == 5
    assert arpabet_to_viseme("V") == 5


def test_vowel_groups():
    assert arpabet_to_viseme("IY") == 7
    assert arpabet_to_viseme("IH") == 7
    assert arpabet_to_viseme("EY") == 8
    assert arpabet_to_viseme("AA") == 9
    assert arpabet_to_viseme("AO") == 10
    assert arpabet_to_viseme("UH") == 11
    assert arpabet_to_viseme("ER") == 12


def test_stress_stripped():
    """Stress digits should be stripped before lookup."""
    assert arpabet_to_viseme("IY1") == 7
    assert arpabet_to_viseme("AH0") == 9
    assert arpabet_to_viseme("EY2") == 8


def test_unknown_returns_silence():
    assert arpabet_to_viseme("???") == 0


def test_lee_viseme_class():
    lv = LeeViseme()
    assert lv.num_visemes == NUM_VISEMES

    # Test label
    assert lv.viseme_label(1) == "P"
    assert lv.viseme_label(5) == "F"

    # Test phoneme_to_viseme
    assert lv.phoneme_to_viseme("M") == 1


def test_text_to_viseme_sequence():
    lv = LeeViseme()
    seq = lv.text_to_viseme_sequence("hello")
    assert len(seq) > 0
    assert all(0 <= v < NUM_VISEMES for v in seq)


def test_describe_sequence():
    lv = LeeViseme()
    labels = lv.describe_sequence([0, 1, 5, 7])
    assert labels == ["SIL", "P", "F", "IY"]
