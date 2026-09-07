"""
Parking-lot layout templates.

Why this exists
---------------
A slot-level system has to know *where* the bays are before it can say whether
each one is free. In a real deployment the camera is bolted to a building and
never moves, so the bay layout is surveyed once at installation and reused for
every frame thereafter — exactly how PKLot itself is annotated. This module
stores one surveyed layout per camera and recognises which camera an uploaded
photograph came from.

Three ways a photograph gets its geometry, in order of preference:

1. **Recognised camera.** The upload is matched against each stored layout's
   reference view; a confident match reuses that camera's surveyed bays.
2. **Manual choice.** The operator picks the camera from a dropdown.
3. **Grid fallback.** For an unseen lot, a rows x columns grid is laid over the
   image so the system still returns slot-level output, clearly labelled as an
   approximation rather than a survey.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import config as C

LAYOUT_PATH = C.MODELS_DIR / "layouts.json"
THUMB_SIZE = (96, 54)          # width, height of the matching fingerprint

LOT_LABELS = {
    "pucpr": "PUCPR — rooftop view, ~100 bays",
    "ufpr04": "UFPR04 — 4th-floor view, ~28 bays",
    "ufpr05": "UFPR05 — 5th-floor view, ~40 bays",
}


@dataclass
class Layout:
    """One surveyed camera: its bay polygons and a visual fingerprint."""

    lot: str
    label: str
    n_spaces: int
    spaces: List[dict]                     # [{"space_id": int, "points": [[x, y], ...]}]
    fingerprint: Optional[np.ndarray] = None

    @property
    def display_name(self) -> str:
        return self.label


# --------------------------------------------------------------------------
# Fingerprinting
# --------------------------------------------------------------------------
def fingerprint(image_rgb: np.ndarray) -> np.ndarray:
    """
    Reduce an image to a small zero-mean, unit-norm greyscale vector.

    Zero-meaning and normalising make the comparison invariant to overall
    brightness and contrast, so the same camera still matches on a rainy
    afternoon after being surveyed on a sunny morning.
    """
    grey = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(grey, THUMB_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    small -= small.mean()
    norm = np.linalg.norm(small)
    if norm > 1e-6:
        small /= norm
    return small.ravel()


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation in [0, 1]; 1.0 is a perfect structural match."""
    return float(np.clip(np.dot(a, b), -1.0, 1.0) * 0.5 + 0.5)


# --------------------------------------------------------------------------
# Building the library
# --------------------------------------------------------------------------
def build_library(
    demo_index_path: Path = C.DEMO_INDEX_PATH,
    demo_dir: Path = C.DEMO_DIR,
    out_path: Path = LAYOUT_PATH,
) -> Dict[str, Layout]:
    """
    Derive one layout per lot from the reserved demo scenes.

    For each lot we keep the scene with the most annotated bays (the fullest
    survey) and fingerprint the *median* of that lot's scenes, so the reference
    view is not skewed by one unusually empty or unusually bright frame.
    """
    entries = json.loads(Path(demo_index_path).read_text(encoding="utf-8"))

    by_lot: Dict[str, List[dict]] = {}
    for entry in entries:
        by_lot.setdefault(entry["lot"], []).append(entry)

    library: Dict[str, Layout] = {}
    payload = []

    for lot, group in sorted(by_lot.items()):
        richest = max(group, key=lambda e: e["n_spaces"])

        stack = []
        for entry in group:
            image = cv2.imread(str(Path(demo_dir) / entry["image"]), cv2.IMREAD_COLOR)
            if image is None:
                continue
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            stack.append(cv2.resize(grey, THUMB_SIZE, interpolation=cv2.INTER_AREA))
        if not stack:
            continue

        median = np.median(np.stack(stack), axis=0).astype(np.float32)
        median -= median.mean()
        norm = np.linalg.norm(median)
        if norm > 1e-6:
            median /= norm

        spaces = [
            {"space_id": s["space_id"], "points": s["points"]} for s in richest["spaces"]
        ]
        library[lot] = Layout(
            lot=lot,
            label=LOT_LABELS.get(lot, f"{lot} — {len(spaces)} bays"),
            n_spaces=len(spaces),
            spaces=spaces,
            fingerprint=median.ravel(),
        )
        payload.append(
            {
                "lot": lot,
                "label": library[lot].label,
                "n_spaces": len(spaces),
                "reference_image": richest["image"],
                "fingerprint": [round(float(v), 6) for v in median.ravel()],
                "spaces": spaces,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return library


def load_library(path: Path = LAYOUT_PATH) -> Dict[str, Layout]:
    """Load the layout library written by :func:`build_library`."""
    path = Path(path)
    if not path.exists():
        return {}
    library: Dict[str, Layout] = {}
    for item in json.loads(path.read_text(encoding="utf-8")):
        library[item["lot"]] = Layout(
            lot=item["lot"],
            label=item["label"],
            n_spaces=item["n_spaces"],
            spaces=item["spaces"],
            fingerprint=np.asarray(item["fingerprint"], dtype=np.float32),
        )
    return library


# --------------------------------------------------------------------------
# Matching and fallback
# --------------------------------------------------------------------------
def match_layout(
    image_rgb: np.ndarray,
    library: Dict[str, Layout],
    min_similarity: float = 0.80,
) -> Tuple[Optional[Layout], float, List[Tuple[str, float]]]:
    """
    Identify which surveyed camera an image came from.

    Returns ``(layout | None, best_score, all_scores)``. ``None`` means no
    stored camera was close enough and the caller should fall back to a grid.
    """
    if not library:
        return None, 0.0, []

    probe = fingerprint(image_rgb)
    scores = sorted(
        ((lot, similarity(probe, layout.fingerprint)) for lot, layout in library.items()),
        key=lambda pair: pair[1],
        reverse=True,
    )
    best_lot, best_score = scores[0]
    if best_score < min_similarity:
        return None, best_score, scores
    return library[best_lot], best_score, scores


def grid_layout(
    rows: int = 4,
    cols: int = 10,
    margin: float = 0.06,
    gap: float = 0.12,
) -> List[dict]:
    """
    Lay a regular rows x cols grid of bays over the whole frame.

    Used only when no surveyed camera matches. ``gap`` leaves an aisle between
    bays so adjacent crops do not overlap, which would otherwise let one car
    appear in two bays and be counted twice.
    """
    spaces: List[dict] = []
    usable = 1.0 - 2.0 * margin
    cell_w = usable / cols
    cell_h = usable / rows
    pad_x = cell_w * gap / 2.0
    pad_y = cell_h * gap / 2.0

    space_id = 1
    for row in range(rows):
        for col in range(cols):
            x1 = margin + col * cell_w + pad_x
            x2 = margin + (col + 1) * cell_w - pad_x
            y1 = margin + row * cell_h + pad_y
            y2 = margin + (row + 1) * cell_h - pad_y
            spaces.append(
                {
                    "space_id": space_id,
                    "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                }
            )
            space_id += 1
    return spaces


def layout_choices(library: Dict[str, Layout]) -> List[Tuple[str, str]]:
    """``(key, label)`` pairs for the dashboard's camera dropdown."""
    return [(lot, layout.label) for lot, layout in sorted(library.items())]
