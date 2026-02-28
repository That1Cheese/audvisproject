"""Tests for sentence list data."""

import pytest
from pronun.data.sentence_lists import (
    BEGINNER_SENTENCES,
    INTERMEDIATE_SENTENCES,
    ADVANCED_SENTENCES,
    SENTENCE_FOCUS,
    ALL_SENTENCES,
)


def test_beginner_sentences_not_empty():
    assert len(BEGINNER_SENTENCES) > 0


def test_intermediate_sentences_not_empty():
    assert len(INTERMEDIATE_SENTENCES) > 0


def test_advanced_sentences_not_empty():
    assert len(ADVANCED_SENTENCES) > 0


def test_all_sentences_is_union():
    assert len(ALL_SENTENCES) == (
        len(BEGINNER_SENTENCES) + len(INTERMEDIATE_SENTENCES) + len(ADVANCED_SENTENCES)
    )


def test_sentence_focus_groups():
    assert "r_vs_l" in SENTENCE_FOCUS
    assert "th_voiced" in SENTENCE_FOCUS
    assert "th_voiceless" in SENTENCE_FOCUS
    for group, sents in SENTENCE_FOCUS.items():
        assert len(sents) > 0, f"Empty group: {group}"


def test_beginner_sentences_are_short():
    for s in BEGINNER_SENTENCES:
        word_count = len(s.split())
        assert word_count <= 6, f"Too long for beginner: '{s}' ({word_count} words)"


def test_sentences_are_strings():
    for s in ALL_SENTENCES:
        assert isinstance(s, str)
        assert len(s) > 0
