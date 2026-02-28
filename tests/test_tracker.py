"""Tests for in-memory session tracker."""

import pytest
from pronun.workflow.tracker import SessionTracker


def _make_result(sentence="hello world", audio=70, combined=75):
    return {
        "sentence": sentence,
        "audio_score": audio,
        "visual_score_b": None,
        "combined_score": combined,
        "sentence_score": combined,
        "word_scores": [],
    }


def test_empty_tracker():
    tracker = SessionTracker()
    assert tracker.get_history() == []
    summary = tracker.summary()
    assert summary["total_attempts"] == 0


def test_record_and_history():
    tracker = SessionTracker()
    tracker.record(_make_result())
    assert len(tracker.get_history()) == 1


def test_history_filter_by_sentence():
    tracker = SessionTracker()
    tracker.record(_make_result("hello world"))
    tracker.record(_make_result("good morning"))
    tracker.record(_make_result("hello world"))
    assert len(tracker.get_history("hello world")) == 2
    assert len(tracker.get_history("good morning")) == 1
    assert len(tracker.get_history("nonexistent")) == 0


def test_trend():
    tracker = SessionTracker()
    tracker.record(_make_result("hello", combined=60))
    tracker.record(_make_result("hello", combined=70))
    tracker.record(_make_result("hello", combined=80))
    trend = tracker.get_trend("hello")
    assert len(trend) == 3
    assert trend[0]["attempt_number"] == 1
    assert trend[2]["combined_score"] == 80


def test_summary():
    tracker = SessionTracker()
    tracker.record(_make_result(combined=60))
    tracker.record(_make_result(combined=80))
    summary = tracker.summary()
    assert summary["total_attempts"] == 2
    assert summary["avg_score"] == 70
    assert summary["best_score"] == 80
    assert summary["latest_score"] == 80
    assert summary["improvement"] == 20


def test_summary_single_attempt():
    tracker = SessionTracker()
    tracker.record(_make_result(combined=75))
    summary = tracker.summary()
    assert summary["improvement"] == 0
    assert summary["total_attempts"] == 1
