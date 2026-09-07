"""
Step 4 of the brief: parking insight logic.

This module is the layer that turns a list of per-bay predictions into
something a driver or a municipal operator can act on: how many bays are free,
how congested the lot is, and what the driver should do next. It is deliberately
free of TensorFlow and OpenCV imports so that the decision rules can be unit
tested on plain numbers, independent of any model.

Congestion bands (fixed by the brief)
    occupancy  <  40 %          -> Low
    40 % <= occupancy <= 75 %   -> Moderate
    occupancy  >  75 %          -> High
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence

from . import config as C


@dataclass
class SlotPrediction:
    """One parking bay after the classifier has looked at it."""

    space_id: int
    status: str                    # "occupied" | "empty"
    confidence: float              # probability of the reported status, 0-1
    score: float                   # raw sigmoid output = P(occupied)
    uncertain: bool = False
    polygon: Optional[list] = None

    @property
    def is_occupied(self) -> bool:
        return self.status == "occupied"


@dataclass
class ParkingInsight:
    """The complete answer the dashboard renders."""

    total_slots: int
    occupied_slots: int
    available_slots: int
    occupancy_pct: float
    congestion_level: str
    headline: str
    recommendation: str
    detail: str
    colour: str
    emoji: str
    uncertain_slots: int = 0
    mean_confidence: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def classify_congestion(occupancy_pct: float) -> str:
    """Map an occupancy percentage onto the brief's three congestion bands."""
    if occupancy_pct < C.CONGESTION_LOW_MAX:
        return "Low"
    if occupancy_pct <= C.CONGESTION_MODERATE_MAX:
        return "Moderate"
    return "High"


def score_to_prediction(
    space_id: int,
    score: float,
    threshold: float = C.DEFAULT_THRESHOLD,
    uncertain_band: float = C.UNCERTAIN_BAND,
    polygon: Optional[list] = None,
) -> SlotPrediction:
    """
    Turn one sigmoid score into a labelled bay.

    ``score`` is P(occupied). A score sitting within ``uncertain_band`` of the
    decision threshold is still given a label — the driver needs an answer —
    but is flagged so the dashboard can show it as low confidence rather than
    presenting a coin-flip as a fact.
    """
    occupied = score >= threshold
    status = "occupied" if occupied else "empty"
    confidence = float(score if occupied else 1.0 - score)
    return SlotPrediction(
        space_id=space_id,
        status=status,
        confidence=confidence,
        score=float(score),
        uncertain=abs(score - threshold) < uncertain_band,
        polygon=polygon,
    )


def generate_insight(predictions: Sequence[SlotPrediction]) -> ParkingInsight:
    """
    Aggregate per-bay predictions into counts, an occupancy percentage, a
    congestion level and a recommendation.
    """
    total = len(predictions)
    if total == 0:
        return ParkingInsight(
            total_slots=0,
            occupied_slots=0,
            available_slots=0,
            occupancy_pct=0.0,
            congestion_level="Unknown",
            headline="No parking bays detected",
            recommendation="Upload a clearer parking-lot image, or pick a demo scene.",
            detail="The system could not find any parking bays to analyse in this image.",
            colour="#6B7280",
            emoji="⚪",
        )

    occupied = sum(1 for p in predictions if p.is_occupied)
    available = total - occupied
    occupancy_pct = round(100.0 * occupied / total, 1)

    level = classify_congestion(occupancy_pct)
    profile = C.CONGESTION_LEVELS[level]

    uncertain = sum(1 for p in predictions if p.uncertain)
    mean_confidence = round(sum(p.confidence for p in predictions) / total, 4)

    notes: List[str] = []
    if available == 0:
        notes.append("Every annotated bay is occupied — no space is available right now.")
    elif available <= 3:
        notes.append(f"Only {available} bay(s) left; they may be taken within minutes.")
    if uncertain:
        share = 100.0 * uncertain / total
        notes.append(
            f"{uncertain} of {total} bays ({share:.0f}%) were borderline calls — "
            "usually caused by shadow, glare or a partly occluded car."
        )
    if mean_confidence < 0.80:
        notes.append(
            "Average confidence is below 80%; consider a clearer or better-lit image."
        )

    return ParkingInsight(
        total_slots=total,
        occupied_slots=occupied,
        available_slots=available,
        occupancy_pct=occupancy_pct,
        congestion_level=level,
        headline=profile["headline"],
        recommendation=profile["recommendation"],
        detail=profile["detail"],
        colour=profile["colour"],
        emoji=profile["emoji"],
        uncertain_slots=uncertain,
        mean_confidence=mean_confidence,
        notes=notes,
    )


def row_summary(predictions: Sequence[SlotPrediction], n_rows: int = 3) -> List[Dict]:
    """
    Split the lot into horizontal bands and report availability per band.

    A driver does not just want "12 free somewhere" — they want to know which
    end of the lot to drive to. Bays are bucketed by the vertical centre of
    their polygon, so band 1 is the row nearest the camera.
    """
    with_geometry = [p for p in predictions if p.polygon]
    if len(with_geometry) < n_rows:
        return []

    centres = [
        (sum(pt[1] for pt in p.polygon) / len(p.polygon), p) for p in with_geometry
    ]
    lo = min(c for c, _ in centres)
    hi = max(c for c, _ in centres)
    span = (hi - lo) or 1.0

    bands: List[List[SlotPrediction]] = [[] for _ in range(n_rows)]
    for centre, prediction in centres:
        index = min(n_rows - 1, int((centre - lo) / span * n_rows))
        bands[index].append(prediction)

    summary = []
    for index, band in enumerate(bands, 1):
        if not band:
            continue
        free = sum(1 for p in band if not p.is_occupied)
        summary.append(
            {
                "zone": f"Zone {index}",
                "position": ["Near the entrance", "Middle of the lot", "Far end"][
                    min(index - 1, 2)
                ],
                "total": len(band),
                "available": free,
                "occupied": len(band) - free,
                "occupancy_pct": round(100.0 * (len(band) - free) / len(band), 1),
            }
        )

    return summary


def best_zone(zones: Sequence[Dict]) -> Optional[Dict]:
    """The zone a driver should head for: most free bays, ties broken by proximity."""
    usable = [z for z in zones if z["available"] > 0]
    if not usable:
        return None
    return max(usable, key=lambda z: (z["available"], -int(z["zone"].split()[-1])))
