"""wav2vec2-based phoneme recognition.

Uses Wav2Vec2FeatureExtractor + manual vocab loading to avoid
the phonemizer/espeak-ng system dependency required by
Wav2Vec2PhonemeCTCTokenizer.
"""

import json

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

from pronun.config import SAMPLE_RATE, WAV2VEC2_MODEL

_model = None
_feature_extractor = None
_vocab: dict[str, int] = {}
_id_to_token: dict[int, str] = {}
_blank_id: int = 0


def _load_model():
    global _model, _feature_extractor, _vocab, _id_to_token, _blank_id
    if _model is None:
        _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC2_MODEL)
        _model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_MODEL)
        _model.eval()

        # Load vocab.json directly to avoid phonemizer dependency
        vocab_path = hf_hub_download(repo_id=WAV2VEC2_MODEL, filename="vocab.json")
        with open(vocab_path, encoding="utf-8") as f:
            _vocab = json.load(f)
        _id_to_token = {v: k for k, v in _vocab.items()}

        # Blank/pad token is typically "<pad>" at index 0
        _blank_id = _vocab.get("<pad>", 0)

    return _model, _feature_extractor


class PhonemeResult:
    """Result of phoneme recognition."""

    def __init__(self, logits: np.ndarray, log_probs: np.ndarray,
                 predicted_ids: list[int], predicted_phonemes: list[str],
                 vocab: dict[str, int]):
        self.logits = logits              # [T, V] raw logits
        self.log_probs = log_probs        # [T, V] log-softmax
        self.predicted_ids = predicted_ids
        self.predicted_phonemes = predicted_phonemes
        self.vocab = vocab                # token → id mapping


def recognize(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> PhonemeResult:
    """Run wav2vec2 phoneme recognition on audio.

    Args:
        audio: 1D float array of audio samples.
        sample_rate: Sample rate of the audio.

    Returns:
        PhonemeResult with logits, log probs, and decoded phonemes.
    """
    model, feature_extractor = _load_model()

    if sample_rate != 16000:
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, 16000)
        audio = audio_tensor.squeeze(0).numpy()

    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(inputs.input_values)
        logits = outputs.logits  # [1, T, V]

    logits_np = logits.squeeze(0).numpy()
    log_probs = torch.log_softmax(logits.squeeze(0), dim=-1).numpy()

    # CTC greedy decode
    predicted_ids = np.argmax(logits_np, axis=-1).tolist()

    # Collapse repeated tokens and remove blanks
    collapsed = []
    prev_id = None
    for idx in predicted_ids:
        if idx != prev_id and idx != _blank_id:
            collapsed.append(idx)
        prev_id = idx

    predicted_phonemes = [_id_to_token.get(i, "<unk>") for i in collapsed]

    return PhonemeResult(
        logits=logits_np,
        log_probs=log_probs,
        predicted_ids=predicted_ids,
        predicted_phonemes=predicted_phonemes,
        vocab=_vocab,
    )
