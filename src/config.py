"""
Central configuration for ParkVision AI.

Every tunable constant used by the data pipeline, the model, the insight
engine and the Streamlit app lives here so that a reviewer can see the whole
system contract in one file.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                 # full parking-lot images + slot geometry
PROCESSED_DIR = DATA_DIR / "processed"     # /train /val /test -> /occupied /empty
DEMO_DIR = DATA_DIR / "demo"               # held-out full-lot scenes for the web app

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SCREENSHOT_DIR = RESULTS_DIR / "screenshots"
DOCS_DIR = PROJECT_ROOT / "docs"

MODEL_PATH = MODELS_DIR / "parkvision_mobilenetv2.keras"
CLASS_INDEX_PATH = MODELS_DIR / "class_indices.json"
METRICS_PATH = RESULTS_DIR / "metrics.json"
HISTORY_PATH = RESULTS_DIR / "training_history.json"

# Slot geometry for the demo scenes, written by scripts/01_build_dataset.py
DEMO_INDEX_PATH = DEMO_DIR / "demo_index.json"

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
# PKLot (Almeida et al., 2015) mirrored on the Hugging Face Hub by Voxel51.
HF_REPO = "Voxel51/PKLot"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/"
HF_SAMPLES_URL = HF_BASE_URL + "samples.json"

CLASS_NAMES = ["empty", "occupied"]   # index 0 -> empty, index 1 -> occupied
POSITIVE_CLASS = "occupied"           # the class the sigmoid head predicts as 1

IMG_SIZE = (224, 224)                 # required by the brief and by MobileNetV2
IMG_CHANNELS = 3

# Target slot crops per class. The brief asks for "at least 100 images per
# class"; we use a far larger, weather-balanced sample so that the model sees
# sunny, overcast and rainy conditions.
CROPS_PER_CLASS = 3000

# Number of full parking-lot scenes reserved for the Streamlit demo. These
# scenes come from days that never appear in train/val/test.
DEMO_SCENES = 12

# Minimum pixel area for a slot crop to be usable (rejects degenerate boxes).
MIN_SLOT_AREA = 300

# Minimum pixel standard deviation for a crop to carry any signal. A warp that
# lands on a featureless region - a bay fully hidden behind a tree, a blown-out
# highlight - produces a flat patch that is identical whatever its label says,
# so admitting it into training is pure label noise.
MIN_CROP_STD = 6.0

# Split ratios required by the brief.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS_HEAD = 12          # phase 1: frozen MobileNetV2 backbone
EPOCHS_FINETUNE = 10      # phase 2: top blocks unfrozen
TOTAL_EPOCHS = EPOCHS_HEAD + EPOCHS_FINETUNE   # 22 -> inside the brief's 15-30 band

LR_HEAD = 1e-3
LR_FINETUNE = 1e-5
FINE_TUNE_FROM_LAYER = 100   # unfreeze MobileNetV2 from this layer index upward
DROPOUT = 0.30
LABEL_SMOOTHING = 0.02
EARLY_STOPPING_PATIENCE = 6

# Augmentation strength (applied to the training split only).
AUG_ROTATION = 0.06        # +/- ~21 degrees, expressed as a fraction of 2*pi
AUG_ZOOM = 0.12
AUG_TRANSLATION = 0.08
AUG_CONTRAST = 0.25
AUG_BRIGHTNESS = 0.25
# Sensor noise. The first model had never seen a grainy image: under sigma=18
# noise it called 29 empty bays occupied, reading grain on bare tarmac as car
# texture. Half of all training images now get noise with a per-image strength
# drawn uniformly from [0, 25] on the 0-255 scale. The range covers daylight
# sensors (sigma ~2-5) through low-light high-ISO ones (~15-25); it is not
# tuned to the sigma=18 stress test, which sits inside it, and sigma=30 is kept
# outside it deliberately so the stress test can check generalisation.
AUG_NOISE_MAX_STDDEV = 25.0
AUG_NOISE_PROBABILITY = 0.5

# Decision threshold on the sigmoid output. Re-tuned on the validation split by
# scripts/03_evaluate.py and persisted into metrics.json.
DEFAULT_THRESHOLD = 0.5

# --------------------------------------------------------------------------
# Parking insight logic (Step 4 of the brief)
# --------------------------------------------------------------------------
CONGESTION_LOW_MAX = 40.0      # occupancy < 40%          -> Low
CONGESTION_MODERATE_MAX = 75.0 # 40% <= occupancy <= 75%  -> Moderate
                               # occupancy > 75%          -> High

CONGESTION_LEVELS = {
    "Low": {
        "colour": "#16A34A",
        "emoji": "🟢",
        "headline": "Plenty of space available",
        "recommendation": "Slots available — proceed to park.",
        "detail": "The lot is lightly used. Drive in and pick any highlighted green slot.",
    },
    "Moderate": {
        "colour": "#D97706",
        "emoji": "🟠",
        "headline": "Filling up — park soon",
        "recommendation": "Slots available — proceed, but park promptly.",
        "detail": "Availability is shrinking. Head to the nearest green slot before it is taken.",
    },
    "High": {
        "colour": "#DC2626",
        "emoji": "🔴",
        "headline": "Heavy congestion",
        "recommendation": "Parking almost full — try another area.",
        "detail": "Very few slots remain. Consider an alternative lot to avoid circling traffic.",
    },
}

# Overlay colours, in RGB (green = empty, red = occupied) as the brief requires.
COLOUR_EMPTY = (22, 163, 74)
COLOUR_OCCUPIED = (220, 38, 38)
COLOUR_UNCERTAIN = (234, 179, 8)

# Predictions whose sigmoid score sits inside this band around the threshold are
# flagged as low-confidence in the dashboard rather than silently trusted.
UNCERTAIN_BAND = 0.15

# --------------------------------------------------------------------------
# Derived helpers
# --------------------------------------------------------------------------
SPLITS = ("train", "val", "test")


def split_dir(split: str, class_name: str) -> Path:
    """Return the folder holding crops of ``class_name`` for ``split``."""
    return PROCESSED_DIR / split / class_name


def ensure_dirs() -> None:
    """Create every directory the pipeline writes into."""
    for path in (RAW_DIR, PROCESSED_DIR, DEMO_DIR, MODELS_DIR, RESULTS_DIR, SCREENSHOT_DIR):
        path.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for class_name in CLASS_NAMES:
            split_dir(split, class_name).mkdir(parents=True, exist_ok=True)
