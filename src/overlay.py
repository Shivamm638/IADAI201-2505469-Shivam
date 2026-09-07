"""
Smart visual overlays.

The brief asks for "color-coded bounding boxes (green for empty and red for
occupied)". This module draws those on the original photograph, plus a compact
legend and an optional confidence badge per bay, and returns an RGB array ready
for ``st.image``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from . import config as C
from .insights import ParkingInsight, SlotPrediction
from .slots import polygon_to_pixels

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _colour_for(prediction: SlotPrediction, mark_uncertain: bool) -> Tuple[int, int, int]:
    if mark_uncertain and prediction.uncertain:
        return C.COLOUR_UNCERTAIN
    return C.COLOUR_OCCUPIED if prediction.is_occupied else C.COLOUR_EMPTY


def draw_overlay(
    image_rgb: np.ndarray,
    predictions: Sequence[SlotPrediction],
    show_ids: bool = False,
    show_confidence: bool = False,
    mark_uncertain: bool = True,
    fill_alpha: float = 0.22,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw one polygon per bay onto a copy of ``image_rgb``.

    Green marks an empty bay, red an occupied one, and amber a call the model
    was not confident about. Interiors are filled at low alpha so the colour
    reads at a glance on a 100-bay lot without hiding the cars underneath.
    """
    canvas = np.ascontiguousarray(image_rgb.copy())
    height, width = canvas.shape[:2]
    scale = max(0.35, min(1.0, width / 1400.0))

    fill_layer = canvas.copy()
    outlines = []

    for prediction in predictions:
        if not prediction.polygon:
            continue
        polygon = polygon_to_pixels(prediction.polygon, width, height).astype(np.int32)
        colour = _colour_for(prediction, mark_uncertain)
        cv2.fillPoly(fill_layer, [polygon], colour)
        outlines.append((polygon, colour, prediction))

    cv2.addWeighted(fill_layer, fill_alpha, canvas, 1.0 - fill_alpha, 0, canvas)

    for polygon, colour, prediction in outlines:
        cv2.polylines(canvas, [polygon], True, colour, thickness, lineType=cv2.LINE_AA)

        if show_ids or show_confidence:
            parts = []
            if show_ids:
                parts.append(str(prediction.space_id))
            if show_confidence:
                parts.append(f"{prediction.confidence * 100:.0f}%")
            label = " ".join(parts)
            anchor = polygon.min(axis=0)
            _text_with_halo(
                canvas, label,
                (int(anchor[0]), max(10, int(anchor[1]) - 4)),
                0.34 * scale, colour, 1,
            )

    return canvas


def _text_with_halo(canvas, text, origin, scale, colour, thickness) -> None:
    """Draw text with a dark outline so it stays readable over any background."""
    cv2.putText(canvas, text, origin, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, origin, FONT, scale, colour, thickness, cv2.LINE_AA)


def draw_legend(
    image_rgb: np.ndarray,
    insight: Optional[ParkingInsight] = None,
    show_uncertain: bool = True,
) -> np.ndarray:
    """Stamp a translucent legend and headline metrics onto the overlay."""
    canvas = np.ascontiguousarray(image_rgb.copy())
    height, width = canvas.shape[:2]
    scale = max(0.42, min(1.05, width / 1300.0))

    entries = [("Empty", C.COLOUR_EMPTY), ("Occupied", C.COLOUR_OCCUPIED)]
    if show_uncertain:
        entries.append(("Low confidence", C.COLOUR_UNCERTAIN))

    pad = int(10 * scale)
    swatch = int(16 * scale)
    line_h = int(24 * scale)
    box_w = int(210 * scale)
    box_h = pad * 2 + line_h * len(entries) + (int(30 * scale) if insight else 0)

    x0, y0 = pad, pad
    panel = canvas[y0:y0 + box_h, x0:x0 + box_w].copy()
    dark = np.zeros_like(panel)
    cv2.addWeighted(dark, 0.62, panel, 0.38, 0, panel)
    canvas[y0:y0 + box_h, x0:x0 + box_w] = panel
    cv2.rectangle(canvas, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), 1, cv2.LINE_AA)

    y = y0 + pad + swatch
    for label, colour in entries:
        cv2.rectangle(
            canvas, (x0 + pad, y - swatch), (x0 + pad + swatch, y), colour, -1, cv2.LINE_AA
        )
        cv2.putText(
            canvas, label, (x0 + pad + swatch + int(8 * scale), y - int(3 * scale)),
            FONT, 0.46 * scale, (255, 255, 255), 1, cv2.LINE_AA,
        )
        y += line_h

    if insight:
        text = (f"{insight.available_slots}/{insight.total_slots} free  "
                f"| {insight.occupancy_pct:.0f}% full")
        cv2.putText(
            canvas, text, (x0 + pad, y + int(10 * scale)),
            FONT, 0.5 * scale, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return canvas


def annotate(
    image_rgb: np.ndarray,
    predictions: Sequence[SlotPrediction],
    insight: Optional[ParkingInsight] = None,
    show_ids: bool = False,
    show_confidence: bool = False,
    mark_uncertain: bool = True,
    legend: bool = True,
) -> np.ndarray:
    """Overlay plus legend, in one call — what the dashboard actually uses."""
    canvas = draw_overlay(
        image_rgb, predictions,
        show_ids=show_ids,
        show_confidence=show_confidence,
        mark_uncertain=mark_uncertain,
    )
    if legend:
        canvas = draw_legend(canvas, insight, show_uncertain=mark_uncertain)
    return canvas


def side_by_side(original_rgb: np.ndarray, annotated_rgb: np.ndarray) -> np.ndarray:
    """Stack the input and the annotated output for a before/after screenshot."""
    height = max(original_rgb.shape[0], annotated_rgb.shape[0])

    def pad(image):
        if image.shape[0] == height:
            return image
        filler = np.full((height - image.shape[0], image.shape[1], 3), 255, np.uint8)
        return np.vstack([image, filler])

    divider = np.full((height, 6, 3), 255, np.uint8)
    return np.hstack([pad(original_rgb), divider, pad(annotated_rgb)])
