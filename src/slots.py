"""
Slot geometry helpers.

PKLot annotates every parking bay as a rotated quadrilateral, not an
axis-aligned box. The same two functions here are used when building the
training set and when running inference in the web app, which guarantees that
a crop the model sees at prediction time is produced exactly like the crops it
was trained on.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


def polygon_to_pixels(points: Sequence[Sequence[float]], width: int, height: int) -> np.ndarray:
    """Convert normalised ``[[x, y], ...]`` polygon points to pixel coordinates."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] *= width
    pts[:, 1] *= height
    return pts


def polygon_bbox(polygon_px: np.ndarray) -> Tuple[int, int, int, int]:
    """Axis-aligned ``(x1, y1, x2, y2)`` bounds of a pixel-space polygon."""
    x1, y1 = polygon_px.min(axis=0)
    x2, y2 = polygon_px.max(axis=0)
    return int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2))


def polygon_area(polygon_px: np.ndarray) -> float:
    """Shoelace area of the polygon in square pixels."""
    x = polygon_px[:, 0]
    y = polygon_px[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """
    Put four corners into a consistent clockwise order.

    The corners are sorted by their angle about the centroid. The textbook
    shortcut — smallest ``x + y`` is top-left, smallest ``y - x`` is top-right
    — is *not* used here because it breaks on the very bays this dataset is
    full of: for a quadrilateral rotated near 45 degrees the same corner can
    win two of those four tests, the source quad collapses to three distinct
    points, and ``getPerspectiveTransform`` then returns a singular matrix that
    warps the bay to a flat, featureless patch.

    Sorting by angle is correct for any convex quadrilateral at any rotation.
    The ordering is then rolled to start at the corner nearest the polygon's
    top-left bound, so the same bay always deskews to the same orientation at
    training and at inference time.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    centre = pts.mean(axis=0)

    # Image coordinates have y pointing down, so ascending atan2 walks the
    # corners clockwise on screen — matching the destination rectangle below.
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    ordered = pts[np.argsort(angles)]

    top_left = pts.min(axis=0)
    start = int(np.argmin(((ordered - top_left) ** 2).sum(axis=1)))
    return np.roll(ordered, -start, axis=0).astype(np.float32)


def crop_slot(
    image: np.ndarray,
    polygon_px: np.ndarray,
    out_size: Tuple[int, int] = (224, 224),
    pad_ratio: float = 0.08,
) -> np.ndarray:
    """
    Deskew one parking bay out of a full lot image.

    The quadrilateral is expanded slightly (``pad_ratio``) so that a car's roof
    or bumper overhanging the painted line still lands inside the patch, then
    warped to a square of ``out_size``. Bays annotated with more than four
    points fall back to their minimum-area rotated rectangle.
    """
    pts = np.asarray(polygon_px, dtype=np.float32).reshape(-1, 2)

    if pts.shape[0] != 4:
        rect = cv2.minAreaRect(pts)
        pts = cv2.boxPoints(rect).astype(np.float32)

    quad = _order_quad(pts)

    if pad_ratio:
        centre = quad.mean(axis=0, keepdims=True)
        quad = centre + (quad - centre) * (1.0 + pad_ratio)

    out_w, out_h = out_size
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(
        image, matrix, (out_w, out_h), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE
    )


def crop_many(
    image: np.ndarray,
    polygons_px: Iterable[np.ndarray],
    out_size: Tuple[int, int] = (224, 224),
    pad_ratio: float = 0.08,
) -> List[np.ndarray]:
    """Vectorise :func:`crop_slot` over every bay in a scene."""
    return [crop_slot(image, poly, out_size, pad_ratio) for poly in polygons_px]
