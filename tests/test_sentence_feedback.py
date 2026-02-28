"""Tests for word-level and sentence-level feedback."""

import pytest
from pronun.scoring.feedback import (
    generate_word_feedback,
    overall_feedback,
    generate_feedback,
)


def test_generate_word_feedback_levels():
    word_scores = [
        {"word": "hello", "score": 90},
        {"word": "world", "score": 75},
        {"word": "three", "score": 55},
        {"word": "thought", "score": 30},
    ]
    fb = generate_word_feedback(word_scores)
    assert len(fb) == 4
    assert fb[0]["level"] == "excellent"
    assert fb[1]["level"] == "good"
    assert fb[2]["level"] == "fair"
    assert fb[3]["level"] == "needs_work"


def test_generate_word_feedback_preserves_word():
    word_scores = [{"word": "cat", "score": 80}]
    fb = generate_word_feedback(word_scores)
    assert fb[0]["word"] == "cat"
    assert fb[0]["score"] == 80


def test_overall_feedback_without_words():
    msg = overall_feedback(90)
    assert "Excellent" in msg


def test_overall_feedback_with_weak_words():
    word_scores = [
        {"word": "hello", "score": 90},
        {"word": "three", "score": 40},
        {"word": "thought", "score": 55},
    ]
    msg = overall_feedback(60, word_scores)
    assert "'three'" in msg
    # "thought" has score 55 which is < 70 so should also be listed
    assert "'thought'" in msg
    assert "'hello'" not in msg  # hello is above threshold


def test_overall_feedback_no_weak_words():
    word_scores = [
        {"word": "hello", "score": 90},
        {"word": "world", "score": 85},
    ]
    msg = overall_feedback(87, word_scores)
    assert "Focus on:" not in msg
