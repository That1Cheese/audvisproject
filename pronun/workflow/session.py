"""Practice session loop — orchestrates recording, scoring, and feedback."""

import tempfile
import os
import sys

import numpy as np

from pronun.audio.g2p import text_to_ipa, text_to_arpabet, text_to_ipa_by_word
from pronun.audio.gop_scorer import compute_gop, overall_gop_score
from pronun.audio.phoneme_recognizer import recognize
from pronun.scoring.combiner import adaptive_combine, combine_scores
from pronun.scoring.feedback import (
    generate_feedback,
    generate_word_feedback,
    overall_feedback,
)
from pronun.visual.features.feature_builder import build_feature_sequence
from pronun.visual.features.landmark_extractor import LandmarkExtractor
from pronun.visual.features.normalizer import normalize_sequence
from pronun.visual.scoring.reference import ReferenceBaseline
from pronun.visual.scoring.visual_scorer import VisualScorer
from pronun.visual.viseme.kmeans_viseme import KMeansViseme
from pronun.visual.viseme.lee_viseme import LeeViseme
from pronun.workflow.camera import Camera
from pronun.workflow.recorder_sync import SyncRecorder
from pronun.workflow.tracker import SessionTracker


class Session:
    """A pronunciation practice session."""

    def __init__(
        self,
        use_camera: bool = True,
        mode: str = "both",  # "A", "B", or "both"
        kmeans_model: KMeansViseme | None = None,
    ):
        self.use_camera = use_camera
        self.mode = mode.upper()
        self.kmeans_model = kmeans_model
        self.lee_viseme = LeeViseme()
        self.visual_scorer = VisualScorer()
        self.landmark_extractor = None
        self.camera = None
        self.tracker = SessionTracker()

    def setup(self):
        """Initialize camera and landmark extractor if using video."""
        if self.use_camera:
            try:
                self.camera = Camera()
                self.camera.open()
                self.landmark_extractor = LandmarkExtractor()
            except (RuntimeError, AttributeError, ImportError, OSError) as e:
                self.use_camera = False
                self.camera = None
                print(
                    f"\n[Warning] Visual scoring disabled: {e}\n"
                    "[Warning] Possible causes:\n"
                    "  - Camera permission not granted "
                    "(System Settings → Privacy & Security → Camera)\n"
                    "  - MediaPipe model download failed (SSL issue) — "
                    "check ~/.cache/pronun/face_landmarker.task exists\n"
                    "[Warning] Falling back to audio-only scoring (Visual: N/A)\n",
                    file=sys.stderr,
                )

    def teardown(self):
        """Release resources."""
        if self.camera:
            self.camera.close()
        if self.landmark_extractor:
            self.landmark_extractor.close()

    def _record_and_get_audio_visual(self):
        """Record audio+video and return (audio, video_frames, audio_path)."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = f.name

        recorder = SyncRecorder(camera=self.camera)
        audio, video_frames = recorder.record(audio_path)
        return audio, video_frames, audio_path

    def _compute_visual_score(self, video_frames, text):
        """Compute visual scores from video frames. Returns (score_a, score_b, score)."""
        visual_score_a = None
        visual_score_b = None
        visual_score = None

        if not (self.use_camera and video_frames and self.landmark_extractor):
            return visual_score_a, visual_score_b, visual_score

        landmarks = self.landmark_extractor.extract_sequence(video_frames)
        normalized = normalize_sequence(landmarks)
        features = build_feature_sequence(normalized)

        if not features:
            return visual_score_a, visual_score_b, visual_score

        obs = np.array(features)

        if self.mode in ("B", "BOTH"):
            viseme_seq = self.lee_viseme.text_to_viseme_sequence(text)
            hmm = self.visual_scorer.build_hmm(
                viseme_seq, {}, obs.shape[1],
            )
            score_b = self.visual_scorer.score(hmm, obs, text)
            visual_score_b = score_b["score"]

        if self.mode in ("A", "BOTH") and self.kmeans_model:
            pred_visemes = self.kmeans_model.predict(obs)
            hmm = self.visual_scorer.build_hmm(
                list(pred_visemes), {}, obs.shape[1],
            )
            score_a = self.visual_scorer.score(hmm, obs, text)
            visual_score_a = score_a["score"]

        if visual_score_b is not None:
            visual_score = visual_score_b
        if visual_score_a is not None:
            visual_score = visual_score_a

        return visual_score_a, visual_score_b, visual_score

    def practice_word(self, word: str) -> dict:
        """Run a single word practice iteration.

        Args:
            word: The word to practice.

        Returns:
            Dict with all scoring results and feedback.
        """
        target_ipa = text_to_ipa(word)
        target_arpabet = text_to_arpabet(word)

        audio, video_frames, audio_path = self._record_and_get_audio_visual()

        try:
            # Audio scoring
            result = recognize(audio)
            phoneme_scores = compute_gop(
                log_probs=result.log_probs,
                predicted_ids=result.predicted_ids,
                predicted_phonemes=result.predicted_phonemes,
                target_phonemes=target_ipa,
                vocab=result.vocab,
            )
            audio_score = overall_gop_score(phoneme_scores)

            # Visual scoring
            visual_score_a, visual_score_b, visual_score = (
                self._compute_visual_score(video_frames, word)
            )

            # Combined scoring
            combined = combine_scores(audio_score, visual_score)
            per_phoneme = adaptive_combine(phoneme_scores, visual_score)
            fb = generate_feedback(per_phoneme)
            overall_msg = overall_feedback(combined)

            return {
                "word": word,
                "target_phonemes": target_ipa,
                "audio_score": audio_score,
                "visual_score_a": visual_score_a,
                "visual_score_b": visual_score_b,
                "combined_score": combined,
                "phoneme_details": per_phoneme,
                "feedback": fb,
                "overall_feedback": overall_msg,
            }
        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    def practice_sentence(self, sentence: str) -> dict:
        """Run a full sentence practice iteration.

        Records audio+video ONCE for the entire sentence, runs G2P with
        word segmentation, scores phonemes, and maps scores back to words.

        Args:
            sentence: The sentence to practice.

        Returns:
            Dict with sentence_score, word_scores, phoneme_details, feedback.
        """
        target_ipa, word_segments = text_to_ipa_by_word(sentence)

        audio, video_frames, audio_path = self._record_and_get_audio_visual()

        try:
            # Audio scoring on full sentence
            result = recognize(audio)
            phoneme_scores = compute_gop(
                log_probs=result.log_probs,
                predicted_ids=result.predicted_ids,
                predicted_phonemes=result.predicted_phonemes,
                target_phonemes=target_ipa,
                vocab=result.vocab,
            )
            audio_score = overall_gop_score(phoneme_scores)

            # Visual scoring
            visual_score_a, visual_score_b, visual_score = (
                self._compute_visual_score(video_frames, sentence)
            )

            # Combined per-phoneme scoring
            combined = combine_scores(audio_score, visual_score)
            per_phoneme = adaptive_combine(phoneme_scores, visual_score)

            # Map phoneme scores to words using segment boundaries
            word_scores = []
            for seg in word_segments:
                seg_phonemes = per_phoneme[seg["start"]:seg["end"]]
                if seg_phonemes:
                    word_avg = sum(
                        p.get("combined_score", p.get("gop_score", 0))
                        for p in seg_phonemes
                    ) / len(seg_phonemes)
                else:
                    word_avg = 0
                word_scores.append({
                    "word": seg["word"],
                    "score": word_avg,
                    "phoneme_start": seg["start"],
                    "phoneme_end": seg["end"],
                })

            # Sentence score is the average of word scores
            sentence_score = (
                sum(ws["score"] for ws in word_scores) / len(word_scores)
                if word_scores else combined
            )

            # Feedback
            word_fb = generate_word_feedback(word_scores)
            phoneme_fb = generate_feedback(per_phoneme)
            overall_msg = overall_feedback(sentence_score, word_scores)

            practice_result = {
                "sentence": sentence,
                "sentence_score": sentence_score,
                "audio_score": audio_score,
                "visual_score_a": visual_score_a,
                "visual_score_b": visual_score_b,
                "combined_score": combined,
                "word_scores": word_fb,
                "word_segments": word_segments,
                "phoneme_details": per_phoneme,
                "feedback": phoneme_fb,
                "overall_feedback": overall_msg,
            }

            self.tracker.record(practice_result)
            return practice_result

        finally:
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    def get_progress(self, sentence: str = None) -> dict:
        """Get session progress data from the tracker."""
        return {
            "history": self.tracker.get_history(sentence),
            "trend": self.tracker.get_trend(sentence),
            "summary": self.tracker.summary(),
        }

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, *args):
        self.teardown()
