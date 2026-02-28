"""In-memory session tracker for pronunciation practice attempts."""

from datetime import datetime


class SessionTracker:
    """Tracks practice attempts during a single app session.

    All data is in-memory and resets when the app exits.
    """

    def __init__(self):
        self.attempts: list[dict] = []

    def record(self, result: dict):
        """Store a practice attempt result."""
        self.attempts.append({
            "sentence": result.get("sentence", ""),
            "timestamp": datetime.now().isoformat(),
            "audio_score": result.get("audio_score", 0),
            "visual_score": result.get("visual_score_b"),
            "combined_score": result.get("combined_score", 0),
            "sentence_score": result.get("sentence_score", 0),
            "word_scores": result.get("word_scores", []),
        })

    def get_history(self, sentence: str = None) -> list[dict]:
        """Get attempts, optionally filtered by sentence."""
        if sentence is None:
            return list(self.attempts)
        return [a for a in self.attempts if a["sentence"] == sentence]

    def get_trend(self, sentence: str = None) -> list[dict]:
        """Return chronological list of score trends."""
        history = self.get_history(sentence)
        trend = []
        for i, attempt in enumerate(history):
            trend.append({
                "attempt_number": i + 1,
                "audio_score": attempt["audio_score"],
                "visual_score": attempt["visual_score"],
                "combined_score": attempt["combined_score"],
            })
        return trend

    def summary(self) -> dict:
        """Overall session stats."""
        if not self.attempts:
            return {
                "total_attempts": 0,
                "avg_score": 0,
                "best_score": 0,
                "latest_score": 0,
                "improvement": 0,
            }

        scores = [a["combined_score"] for a in self.attempts]
        return {
            "total_attempts": len(self.attempts),
            "avg_score": sum(scores) / len(scores),
            "best_score": max(scores),
            "latest_score": scores[-1],
            "improvement": scores[-1] - scores[0] if len(scores) > 1 else 0,
        }
