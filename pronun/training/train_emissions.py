"""Train HMM emission parameters and calibrate reference baseline from GRID corpus."""

import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from pronun.audio.g2p import text_to_arpabet
from pronun.data.lee_map import arpabet_to_viseme
from pronun.visual.features.feature_builder import build_feature_sequence
from pronun.visual.features.landmark_extractor import LandmarkExtractor
from pronun.visual.features.normalizer import normalize_sequence
from pronun.visual.scoring.emission_store import EmissionStore
from pronun.visual.scoring.reference import ReferenceBaseline
from pronun.visual.scoring.visual_scorer import VisualScorer
from pronun.visual.viseme.lee_viseme import LeeViseme
from pronun.config import EMISSION_STORE_PATH, REFERENCE_BASELINE_PATH, HMM_COV_REGULARIZATION, AUG_COUNT
from pronun.visual.features.augmentation import augment_sequence


# ---------------------------------------------------------------------------
# GRID alignment parsing
# ---------------------------------------------------------------------------

def parse_grid_align(align_path: str | Path) -> list[dict]:
    """Parse a GRID .align file into word segments with frame boundaries.

    GRID timestamps use units where 1000 ≈ 1 frame at 25fps (i.e., the
    unit is 1/25000 second, so frame = round(timestamp / 1000)).

    Args:
        align_path: Path to a .align file.

    Returns:
        List of dicts with keys: "word", "start_frame", "end_frame".
        Silence segments ("sil", "sp") are excluded.
    """
    segments = []
    with open(align_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            start_ts, end_ts, word = int(parts[0]), int(parts[1]), parts[2]
            if word.lower() in ("sil", "sp", ""):
                continue
            start_frame = round(start_ts / 1000)
            end_frame = round(end_ts / 1000)
            if end_frame > start_frame:
                segments.append({
                    "word": word,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                })
    return segments


def _word_to_viseme_ids(word: str) -> list[int]:
    """Convert a single word to a list of viseme IDs via ARPAbet → Lee map."""
    arpabet = text_to_arpabet(word)
    return [arpabet_to_viseme(p) for p in arpabet]


# ---------------------------------------------------------------------------
# Viseme data collector
# ---------------------------------------------------------------------------

class VisemeDataCollector:
    """Accumulates feature vectors labeled by viseme ID."""

    def __init__(self):
        self._observations: dict[int, list[np.ndarray]] = defaultdict(list)

    def add_grid_sample(
        self,
        features: list[np.ndarray],
        align_segments: list[dict],
    ):
        """Add features from a GRID clip using word-level alignment.

        For each word segment, converts the word to viseme IDs and
        uniformly distributes the word's frames across its visemes.

        Args:
            features: List of feature vectors (one per frame).
            align_segments: Output of parse_grid_align().
        """
        num_frames = len(features)
        for seg in align_segments:
            start = seg["start_frame"]
            end = min(seg["end_frame"], num_frames)
            if start >= end:
                continue

            viseme_ids = _word_to_viseme_ids(seg["word"])
            if not viseme_ids:
                continue

            word_frames = list(range(start, end))
            n_visemes = len(viseme_ids)
            frames_per_viseme = len(word_frames) / n_visemes

            for v_idx, vid in enumerate(viseme_ids):
                f_start = int(round(v_idx * frames_per_viseme))
                f_end = int(round((v_idx + 1) * frames_per_viseme))
                for fi in range(f_start, min(f_end, len(word_frames))):
                    frame_idx = word_frames[fi]
                    if frame_idx < num_frames:
                        self._observations[vid].append(features[frame_idx])

    def add_sample(self, features: list[np.ndarray], text: str):
        """Add features with uniform segmentation across entire utterance.

        Fallback for non-GRID data without word-level alignment.
        """
        lee = LeeViseme()
        viseme_ids = lee.text_to_viseme_sequence(text)
        if not viseme_ids or not features:
            return

        frames_per_viseme = len(features) / len(viseme_ids)
        for v_idx, vid in enumerate(viseme_ids):
            f_start = int(round(v_idx * frames_per_viseme))
            f_end = int(round((v_idx + 1) * frames_per_viseme))
            for fi in range(f_start, min(f_end, len(features))):
                self._observations[vid].append(features[fi])

    def get_observations(self, viseme_id: int) -> np.ndarray | None:
        """Get stacked observations for a viseme ID, or None if empty."""
        obs = self._observations.get(viseme_id)
        if not obs:
            return None
        return np.array(obs)

    def summary(self) -> dict[int, int]:
        """Return observation counts per viseme ID."""
        return {vid: len(obs) for vid, obs in sorted(self._observations.items())}


# ---------------------------------------------------------------------------
# Feature extraction from video
# ---------------------------------------------------------------------------

def extract_features_from_video(
    video_path: str | Path,
    extractor: LandmarkExtractor,
    n_aug: int = 0,
) -> list[list[np.ndarray]]:
    """Extract 248-dim lip feature vectors from a video file.

    Args:
        video_path: Path to video file (e.g. .mpg).
        extractor: Initialized LandmarkExtractor instance.
        n_aug: Number of augmented copies to generate (0 = original only).

    Returns:
        List of feature lists. First element is the original, followed by
        n_aug augmented variants. Each inner list contains one feature
        vector per frame with a detected face.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    cap.release()

    if not frames:
        return []

    landmarks = extractor.extract_sequence(frames)
    normalized = normalize_sequence(landmarks)

    # Original features
    original_features = build_feature_sequence(normalized)
    if not original_features:
        return []

    result = [original_features]

    # Augmented copies
    if n_aug > 0:
        rng = np.random.default_rng()
        aug_sequences = augment_sequence(normalized, n_aug, rng)
        for aug_norm in aug_sequences:
            aug_features = build_feature_sequence(aug_norm)
            if aug_features:
                result.append(aug_features)

    return result


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def train_from_collector(collector: VisemeDataCollector) -> EmissionStore:
    """Compute emission parameters from collected observations.

    Args:
        collector: Populated VisemeDataCollector.

    Returns:
        EmissionStore with trained (mean, cov) per viseme.
    """
    store = EmissionStore()
    for vid in range(13):  # 0-12 viseme IDs
        obs = collector.get_observations(vid)
        if obs is None or len(obs) < 2:
            continue
        mean = obs.mean(axis=0)
        cov = np.var(obs, axis=0)
        cov += HMM_COV_REGULARIZATION
        store.set_params(vid, mean, cov)
    return store


def train_from_grid(
    corpus_dir: str | Path,
    speakers: list[str],
    max_clips_per_speaker: int = 0,
    output_path: str | Path | None = None,
    progress_callback=None,
    n_aug: int = AUG_COUNT,
) -> tuple[EmissionStore, dict[int, int]]:
    """Train HMM emission parameters from GRID corpus videos.

    Args:
        corpus_dir: Path to grid_corpus directory containing speaker folders.
        speakers: List of speaker IDs (e.g. ["s1", "s2"]).
        max_clips_per_speaker: Max clips to process per speaker (0 = all).
        output_path: Where to save the EmissionStore (default: config path).
        progress_callback: Optional callable(speaker, clip_idx, clip_name).
        n_aug: Number of augmented copies per clip (default: AUG_COUNT).

    Returns:
        (EmissionStore, summary_dict) where summary_dict maps viseme_id → count.
    """
    corpus_dir = Path(corpus_dir)
    output_path = Path(output_path) if output_path else EMISSION_STORE_PATH

    collector = VisemeDataCollector()
    extractor = LandmarkExtractor()

    try:
        for speaker in speakers:
            align_dir = corpus_dir / speaker / "align"
            video_dir = corpus_dir / speaker / "video"

            if not align_dir.exists() or not video_dir.exists():
                continue

            align_files = sorted(align_dir.glob("*.align"))
            if max_clips_per_speaker > 0:
                align_files = align_files[:max_clips_per_speaker]

            for clip_idx, align_path in enumerate(align_files):
                clip_name = align_path.stem
                video_path = video_dir / f"{clip_name}.mpg"

                if not video_path.exists():
                    continue

                if progress_callback:
                    progress_callback(speaker, clip_idx, clip_name)

                segments = parse_grid_align(align_path)
                feature_variants = extract_features_from_video(
                    video_path, extractor, n_aug=n_aug,
                )

                for features in feature_variants:
                    if features and segments:
                        collector.add_grid_sample(features, segments)
    finally:
        extractor.close()

    store = train_from_collector(collector)
    store.save(output_path)
    return store, collector.summary()


# ---------------------------------------------------------------------------
# Baseline calibration
# ---------------------------------------------------------------------------

def calibrate_baseline(
    corpus_dir: str | Path,
    speakers: list[str],
    emission_store: EmissionStore,
    max_clips_per_speaker: int = 0,
    output_path: str | Path | None = None,
    progress_callback=None,
    n_aug: int = AUG_COUNT,
) -> ReferenceBaseline:
    """Calibrate the reference baseline from native speaker recordings.

    For each GRID clip, builds an HMM with trained emissions, runs the
    Forward Algorithm, and collects per-word log-likelihood norms. These
    are aggregated into the ReferenceBaseline.

    Augmented variants of each clip are also processed, which broadens
    the baseline bandwidth to generalize across input domains.

    Args:
        corpus_dir: Path to grid_corpus directory.
        speakers: Speaker IDs to use for calibration.
        emission_store: Trained EmissionStore.
        max_clips_per_speaker: Max clips per speaker (0 = all).
        output_path: Where to save the baseline (default: config path).
        progress_callback: Optional callable(speaker, clip_idx, clip_name).
        n_aug: Number of augmented copies per clip (default: AUG_COUNT).

    Returns:
        Calibrated ReferenceBaseline.
    """
    corpus_dir = Path(corpus_dir)
    output_path = Path(output_path) if output_path else REFERENCE_BASELINE_PATH

    baseline = ReferenceBaseline()
    scorer = VisualScorer(reference=baseline)
    lee = LeeViseme()
    emission_dict = emission_store.to_dict()

    # Collect per-word log-likelihoods: word → [(log_ll, duration), ...]
    word_samples: dict[str, list[tuple[float, int]]] = defaultdict(list)

    extractor = LandmarkExtractor()
    try:
        for speaker in speakers:
            align_dir = corpus_dir / speaker / "align"
            video_dir = corpus_dir / speaker / "video"

            if not align_dir.exists() or not video_dir.exists():
                continue

            align_files = sorted(align_dir.glob("*.align"))
            if max_clips_per_speaker > 0:
                align_files = align_files[:max_clips_per_speaker]

            for clip_idx, align_path in enumerate(align_files):
                clip_name = align_path.stem
                video_path = video_dir / f"{clip_name}.mpg"

                if not video_path.exists():
                    continue

                if progress_callback:
                    progress_callback(speaker, clip_idx, clip_name)

                segments = parse_grid_align(align_path)
                feature_variants = extract_features_from_video(
                    video_path, extractor, n_aug=n_aug,
                )

                for features in feature_variants:
                    if not features or not segments:
                        continue

                    obs_array = np.array(features)
                    num_frames = len(features)

                    for seg in segments:
                        start = seg["start_frame"]
                        end = min(seg["end_frame"], num_frames)
                        if start >= end:
                            continue

                        word_obs = obs_array[start:end]
                        viseme_seq = lee.text_to_viseme_sequence(seg["word"])
                        if not viseme_seq or len(word_obs) == 0:
                            continue

                        hmm = scorer.build_hmm(viseme_seq, emission_dict, word_obs.shape[1])
                        log_ll = hmm.forward(word_obs)

                        if math.isfinite(log_ll):
                            word_samples[seg["word"]].append((log_ll, len(word_obs)))
    finally:
        extractor.close()

    # Aggregate into baseline
    all_norms = []
    for word, samples in word_samples.items():
        log_lls = [s[0] for s in samples]
        durations = [s[1] for s in samples]
        baseline.update_from_samples(word, log_lls, durations)
        all_norms.extend(ll / d for ll, d in zip(log_lls, durations) if d > 0)

    # Set the default reference and bandwidth from calibration data
    if all_norms:
        baseline.default_reference = float(np.median(all_norms))
        baseline.bandwidth = max(float(np.std(all_norms)), 0.1)  # floor at 0.1

    baseline.save(str(output_path))
    return baseline
