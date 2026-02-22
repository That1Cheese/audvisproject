# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the CLI:**
```bash
python -m pronun.cli practice --level beginner
python -m pronun.cli practice --level intermediate --mode B --no-camera
python -m pronun.cli practice-word hello world
python -m pronun.cli list beginner
python -m pronun.cli compare hello
```

**Run tests:**
```bash
pytest tests/                        # all tests (79 unit tests, no hardware required)
pytest tests/test_scorer.py          # single test file
pytest tests/test_gop_scorer.py -v   # verbose single file
```

**Check camera is functional** (requires webcam + macOS camera permission granted to Terminal):
```bash
pytest tests/test_camera.py -v
```
Tests auto-skip if no camera is available. If they skip but you expect a camera, grant permission in **System Settings → Privacy & Security → Camera**, then quit and reopen Terminal.

There is no `setup.py` or `pyproject.toml`; the package must be run from the repo root so that `pronun` is importable.

## Architecture

The system is a pronunciation correction tool combining audio (GOP) and visual (3D mouth landmark) scoring. The `pronun/` package is divided into five sub-packages plus top-level `cli.py` and `config.py`.

### Data flow through a practice session

`Session` (`workflow/session.py`) orchestrates everything:
1. **Record**: `SyncRecorder` captures audio + video frames simultaneously
2. **G2P**: `text_to_ipa_by_word()` (`audio/g2p.py`) converts text → ARPAbet (via `g2p-en`) → IPA with per-word segment boundaries
3. **Audio recognition**: `recognize()` (`audio/phoneme_recognizer.py`) runs wav2vec2 CTC inference (model: `facebook/wav2vec2-lv-60-espeak-cv-ft`, auto-downloaded from HuggingFace)
4. **GOP scoring**: `compute_gop()` (`audio/gop_scorer.py`) aligns predicted vs. target phonemes via edit-distance and averages frame-level log-probs; normalizes to 0–100
5. **Visual scoring** (if camera enabled):
   - `LandmarkExtractor` → `normalize_sequence` → `build_feature_sequence` produces per-frame vectors (4 geometric features + flattened 3D landmarks + deltas)
   - **Mode B** (default): `LeeViseme.text_to_viseme_sequence()` maps text → viseme IDs; `VisualScorer.build_hmm()` + `score()` runs Forward Algorithm on a `GaussianHMM`
   - **Mode A**: `KMeansViseme` clusters frames into viseme IDs instead
6. **Combine**: `adaptive_combine()` (`scoring/combiner.py`) weights audio vs. visual per phoneme — bilabial phonemes (P, B, M, F, V, W…) get 50/50; visually ambiguous get 90/10
7. **Feedback**: `generate_feedback()` / `generate_word_feedback()` produce per-phoneme and per-word dicts; `SessionTracker` records in-memory history

### Sub-package responsibilities

| Package | Key files | Purpose |
|---------|-----------|---------|
| `audio/` | `phoneme_recognizer.py`, `gop_scorer.py`, `g2p.py` | wav2vec2 inference, CTC alignment, G2P |
| `visual/features/` | `landmark_extractor.py`, `normalizer.py`, `feature_builder.py` | MediaPipe face landmarks → normalized feature vectors |
| `visual/scoring/` | `hmm.py`, `visual_scorer.py`, `reference.py` | Gaussian HMM Forward Algorithm, log-likelihood → 0-100 score |
| `visual/viseme/` | `lee_viseme.py`, `kmeans_viseme.py` | ARPAbet→viseme (Lee map) and K-means viseme clustering |
| `scoring/` | `combiner.py`, `feedback.py` | Adaptive audio/visual weighting, human-readable feedback |
| `workflow/` | `session.py`, `recorder_sync.py`, `camera.py`, `tracker.py` | Orchestration, A/V capture, progress tracking |
| `data/` | `word_lists.py`, `sentence_lists.py`, `lee_map.py` | Practice content, ARPAbet→viseme ID table |

### Configuration

All numeric constants (model name, recording params, lip landmark indices, GOP normalization bounds, HMM regularization, scoring weights) live in `config.py`. The MediaPipe FaceLandmarker model is auto-downloaded to `~/.cache/pronun/face_landmarker.task` on first use.

### Scoring modes

- **Mode B** (default): Uses the Lee viseme map to derive expected viseme sequence from text, then HMM-scores actual lip movements against it
- **Mode A**: Uses K-means clustering to predict viseme IDs from raw lip features, requires a pre-trained `KMeansViseme` model passed to `Session`
- **`--no-camera`**: Audio-only; visual score is `None` and combined score equals audio score

## Known Gaps and To-Do List

### Critical — visual score is always 0.0 until resolved

The visual scoring pipeline is fully implemented but produces 0 because two things have never been built:

1. **HMM emissions are untrained** — `Session._compute_visual_score()` calls `build_hmm(viseme_seq, {}, feature_dim)` with an empty observations dict, so every HMM state defaults to `N(mean=0, cov=I)`. For 248-dim features this gives `L_norm ≈ -300` per frame.
2. **`ReferenceBaseline` default is wrong** — `default_reference = -5.0` was designed for trained emissions. With untrained HMMs, `exp(L_norm − L_ref) = exp(−295) ≈ 0`, so score always rounds to 0.

**To fix visual scoring, these must be built in order:**

| # | Task | Blocks |
|---|------|--------|
| 1 | **Build viseme data collection pipeline** — record speakers, extract 248-dim lip feature vectors, label by viseme ID (0–12) | #2, #3, #4 |
| 2 | **Train HMM emission parameters** — call `hmm.train_emissions(viseme_id, observations)` per viseme, save to disk, load in `Session` | visual score |
| 3 | **Calibrate `ReferenceBaseline`** — record native speakers, run trained HMM Forward Algorithm, call `ReferenceBaseline.update_from_samples()`, save/load via `ReferenceBaseline.save()` | visual score |
| 4 | **Train `KMeansViseme` (Mode A)** — call `KMeansViseme(k=12).train(all_features)`, save model, load in `Session.__init__` | Mode A |

### Other bugs and improvements

| # | Task | Notes |
|---|------|-------|
| 5 | **Fix SSL certificates permanently on macOS** | `landmark_extractor._ensure_model()` fails silently on python.org Python 3.13; disables visual scoring with no output. Workaround: `_create_unverified_context` fallback or run `/Applications/Python 3.13/Install Certificates.command` |
| 6 | **Per-phoneme visual scoring** | `adaptive_combine()` applies one sentence-level visual score to all phonemes. Should use GOP frame timestamps to slice video frames per phoneme and score each separately |
| 7 | **Remove dead code `audio/recorder.py`** | `record_audio()` is never called; `Session` uses `SyncRecorder` instead |
| 8 | **Fix empty Word column in Phoneme Details table** | `_phoneme_to_word()` in `cli.py` looks up `phoneme_start/phoneme_end` from `word_scores`, but `generate_word_feedback()` strips those keys — use `word_segments` from the result dict instead |
