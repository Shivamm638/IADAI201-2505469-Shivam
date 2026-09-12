"""
Slot-level inference: photograph in, per-bay occupancy out.

The pipeline is deliberately identical to the one used to build the training
set — the same :func:`src.slots.crop_slot` warps each bay to 224x224 — so no
train/serve skew can creep in.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config as C
from .insights import SlotPrediction, score_to_prediction
from .slots import crop_slot, polygon_area, polygon_to_pixels

_MODEL = None
_MODEL_PATH: Optional[Path] = None


def load_model(path: Path = C.MODEL_PATH, force_reload: bool = False):
    """
    Load and memoise the trained classifier.

    Kept module-level rather than in the Streamlit layer so scripts, notebooks
    and the app all share one loading path.
    """
    global _MODEL, _MODEL_PATH
    path = Path(path)
    if _MODEL is not None and _MODEL_PATH == path and not force_reload:
        return _MODEL
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run scripts/02_train.py first."
        )

    import tensorflow as tf  # imported lazily: keeps CLI tools that only need
                             # geometry free of a multi-second TF import

    # Registers RandomSensorNoise. The saved graph refers to it by name, and
    # Keras can only rebuild a custom layer whose class has been registered.
    from . import model as _model  # noqa: F401

    _MODEL = tf.keras.models.load_model(path)
    _MODEL_PATH = path
    return _MODEL


def load_threshold(metrics_path: Path = C.METRICS_PATH) -> float:
    """
    Read the decision threshold chosen on the validation split.

    Falls back to 0.5 when the model has not been evaluated yet, so the app
    still runs on a freshly trained model.
    """
    try:
        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
        return float(metrics.get("operating_threshold", C.DEFAULT_THRESHOLD))
    except Exception:
        return C.DEFAULT_THRESHOLD


def crops_from_spaces(
    image_rgb: np.ndarray,
    spaces: Sequence[dict],
    min_area: int = C.MIN_SLOT_AREA,
) -> Tuple[np.ndarray, List[dict]]:
    """
    Warp every bay in ``spaces`` out of ``image_rgb``.

    Returns the stacked crops and the subset of ``spaces`` they correspond to;
    bays whose polygon is degenerate or falls outside the frame are dropped so
    the model is never asked to classify a sliver of nothing.
    """
    height, width = image_rgb.shape[:2]
    patches: List[np.ndarray] = []
    kept: List[dict] = []

    for space in spaces:
        polygon = polygon_to_pixels(space["points"], width, height)
        if polygon_area(polygon) < min_area:
            continue
        if (polygon[:, 0].max() < 0 or polygon[:, 1].max() < 0
                or polygon[:, 0].min() > width or polygon[:, 1].min() > height):
            continue
        patches.append(crop_slot(image_rgb, polygon, C.IMG_SIZE))
        kept.append(space)

    if not patches:
        return np.zeros((0, *C.IMG_SIZE, C.IMG_CHANNELS), dtype=np.float32), []

    return np.stack(patches).astype(np.float32), kept


def predict_slots(
    image_rgb: np.ndarray,
    spaces: Sequence[dict],
    model=None,
    threshold: Optional[float] = None,
    batch_size: int = 64,
) -> Tuple[List[SlotPrediction], Dict[str, float]]:
    """
    Classify every bay in one photograph.

    Returns the per-bay predictions plus timing information the dashboard shows
    so a user can see the system is fast enough for live use.
    """
    model = model if model is not None else load_model()
    threshold = load_threshold() if threshold is None else threshold

    started = time.perf_counter()
    patches, kept = crops_from_spaces(image_rgb, spaces)
    crop_seconds = time.perf_counter() - started

    if len(patches) == 0:
        # Return the same keys as the normal path: callers render these
        # directly, and a missing key would surface as a crash rather than as
        # the "no bays found" message the user should see.
        return [], {
            "crop_seconds": round(crop_seconds, 3),
            "inference_seconds": 0.0,
            "total_seconds": round(crop_seconds, 3),
            "n_slots": 0,
            "ms_per_slot": 0.0,
        }

    started = time.perf_counter()
    scores = model.predict(patches, batch_size=batch_size, verbose=0).ravel()
    inference_seconds = time.perf_counter() - started

    predictions = [
        score_to_prediction(
            space_id=int(space.get("space_id", index + 1)),
            score=float(score),
            threshold=threshold,
            polygon=space["points"],
        )
        for index, (space, score) in enumerate(zip(kept, scores))
    ]

    timing = {
        "crop_seconds": round(crop_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "total_seconds": round(crop_seconds + inference_seconds, 3),
        "n_slots": len(predictions),
        "ms_per_slot": round(1000.0 * (crop_seconds + inference_seconds) / len(predictions), 2),
    }
    return predictions, timing


def accuracy_against_truth(
    predictions: Sequence[SlotPrediction], spaces: Sequence[dict]
) -> Optional[Dict[str, float]]:
    """
    Compare predictions with ground truth when the scene carries labels.

    Used on the demo scenes so the dashboard can show, honestly, how many bays
    the model got right on a photograph it never trained on.
    """
    truth = {
        int(s["space_id"]): s["label"]
        for s in spaces
        if s.get("label") in ("occupied", "empty")
    }
    if not truth:
        return None

    matched = [(p, truth[p.space_id]) for p in predictions if p.space_id in truth]
    if not matched:
        return None

    correct = sum(1 for p, label in matched if p.status == label)
    true_occupied = sum(1 for _, label in matched if label == "occupied")
    predicted_occupied = sum(1 for p, _ in matched if p.is_occupied)

    return {
        "n_compared": len(matched),
        "n_correct": correct,
        "accuracy": round(correct / len(matched), 4),
        "true_occupied": true_occupied,
        "true_available": len(matched) - true_occupied,
        "predicted_occupied": predicted_occupied,
        "predicted_available": len(matched) - predicted_occupied,
        "count_error": abs(predicted_occupied - true_occupied),
    }
