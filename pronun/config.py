"""Configuration constants and defaults."""

# Audio recording
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.02
SILENCE_DURATION = 1.5  # seconds of silence before auto-stop
MAX_RECORD_SECONDS = 10

# wav2vec2 model
WAV2VEC2_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"

# GOP scoring — linear map from raw log-prob to 0-100
GOP_MIN = -4.0   # maps to 0
GOP_MAX = -0.1   # maps to 100

# Visual — MediaPipe mouth landmark indices (lip region)
# Outer lip: 61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185
# Inner lip: 78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95
OUTER_LIP_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                     291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
INNER_LIP_INDICES = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                     308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
ALL_LIP_INDICES = OUTER_LIP_INDICES + INNER_LIP_INDICES

# Lip corner indices for width normalization (3D)
LEFT_LIP_CORNER = 61
RIGHT_LIP_CORNER = 291

# Upper/lower lip midpoints for height
UPPER_LIP_MID = 13
LOWER_LIP_MID = 14

# K-means viseme clustering
KMEANS_K = 12
KMEANS_RANDOM_STATE = 42

# HMM
HMM_COV_REGULARIZATION = 1.0

# Combined scoring weights
DEFAULT_AUDIO_WEIGHT = 0.7
DEFAULT_VISUAL_WEIGHT = 0.3

# Visually distinctive phonemes get higher visual weight
VISUAL_DISTINCTIVE_PHONEMES = {
    "P", "B", "M",      # bilabial
    "F", "V",            # labiodental
    "W", "UW", "OW", "AO",  # rounded
}
DISTINCTIVE_AUDIO_WEIGHT = 0.5
DISTINCTIVE_VISUAL_WEIGHT = 0.5
AMBIGUOUS_AUDIO_WEIGHT = 0.9
AMBIGUOUS_VISUAL_WEIGHT = 0.1

# FaceLandmarker model
from pathlib import Path
FACE_LANDMARKER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
FACE_LANDMARKER_MODEL_PATH = Path.home() / ".cache" / "pronun" / "face_landmarker.task"
EMISSION_STORE_PATH = Path.home() / ".cache" / "pronun" / "viseme_emissions.npz"
REFERENCE_BASELINE_PATH = Path.home() / ".cache" / "pronun" / "reference_baseline.npz"

# Data augmentation (applied to normalized landmarks during training)
AUG_COUNT = 3           # augmented copies per frame sequence
AUG_JITTER_STD = 0.02   # landmark coordinate noise std
AUG_ROTATION_DEG = 5.0  # max rotation in degrees per axis
AUG_SCALE_STD = 0.03    # scale perturbation std

# Camera
CAMERA_INDEX = 0
CAMERA_FPS = 30
