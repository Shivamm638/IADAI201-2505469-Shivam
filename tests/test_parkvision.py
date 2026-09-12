"""
Tests for the parts of ParkVision AI whose behaviour must be exact.

Deliberately split in two:

* the **logic** tests (geometry, congestion bands, aggregation, layouts) need no
  trained model and run in about a second — they are the ones that catch a
  regression in the decision rules;
* the **model** tests are skipped when no checkpoint exists, so the suite still
  passes on a fresh clone before training.

Written to run either way::

    python tests/test_parkvision.py     # no dependencies beyond the project
    pytest tests/ -v                    # if pytest is installed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C
from src import insights as I
from src import layouts as L
from src import overlay as O
from src import slots as S


# ==========================================================================
# Geometry
# ==========================================================================
def test_polygon_to_pixels_scales_normalised_points():
    points = [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]
    pixels = S.polygon_to_pixels(points, 1280, 720)
    assert pixels.shape == (4, 2)
    assert np.allclose(pixels[2], [640.0, 360.0])


def test_polygon_area_matches_shoelace():
    square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)
    assert abs(S.polygon_area(square) - 100.0) < 1e-6


def test_polygon_area_is_orientation_independent():
    clockwise = np.array([[0, 0], [0, 10], [10, 10], [10, 0]], dtype=np.float32)
    assert abs(S.polygon_area(clockwise) - 100.0) < 1e-6


def test_crop_slot_returns_the_requested_size():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    polygon = np.array([[100, 100], [200, 110], [195, 210], [95, 200]], dtype=np.float32)
    patch = S.crop_slot(image, polygon, (224, 224))
    assert patch.shape == (224, 224, 3)


def test_crop_slot_deskews_a_rotated_bay():
    """A white quadrilateral on black must fill the warped patch."""
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    polygon = np.array([[100, 120], [260, 90], [280, 210], [120, 240]], dtype=np.float32)
    import cv2
    cv2.fillPoly(image, [polygon.astype(np.int32)], (255, 255, 255))

    patch = S.crop_slot(image, polygon, (64, 64), pad_ratio=0.0)
    centre = patch[16:48, 16:48]
    assert centre.mean() > 240, "the bay's interior should dominate the warped patch"


def test_crop_slot_handles_more_than_four_points():
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    pentagon = np.array([[50, 50], [150, 40], [180, 120], [120, 180], [40, 140]],
                        dtype=np.float32)
    assert S.crop_slot(image, pentagon, (32, 32)).shape == (32, 32, 3)


def test_order_quad_never_collapses_at_any_rotation():
    """
    Regression test.

    The textbook corner-ordering shortcut (smallest x+y is top-left, smallest
    y-x is top-right) reuses the same corner for two slots when a quadrilateral
    sits near 45 degrees. The source quad then has only three distinct points,
    getPerspectiveTransform returns a singular matrix, and the bay warps to a
    flat featureless patch. That silently poisoned 3.5% of the first dataset
    build with blank crops carrying real labels.
    """
    base = np.array([[-50, -25], [50, -25], [50, 25], [-50, 25]], dtype=np.float32)
    for degrees in range(0, 360, 3):
        theta = np.radians(degrees)
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]], dtype=np.float32)
        quad = base @ rotation.T + np.array([160, 120], dtype=np.float32)

        ordered = S._order_quad(quad)
        unique = {tuple(np.round(point, 3)) for point in ordered}
        assert len(unique) == 4, f"collapsed at {degrees}deg: {ordered.tolist()}"


def test_crop_slot_keeps_content_at_every_rotation():
    """A textured bay must stay textured however the bay is rotated."""
    import cv2

    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
    base = np.array([[-60, -30], [60, -30], [60, 30], [-60, 30]], dtype=np.float32)

    for degrees in range(0, 360, 15):
        theta = np.radians(degrees)
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]], dtype=np.float32)
        quad = base @ rotation.T + np.array([200, 150], dtype=np.float32)

        patch = S.crop_slot(image, quad, (64, 64))
        assert patch.std() > 20, f"flat patch at {degrees}deg (std={patch.std():.2f})"


def _signed_area(quad: np.ndarray) -> float:
    """Shoelace sum. In image coordinates (y down) a clockwise ring is positive."""
    total = 0.0
    for i in range(len(quad)):
        x1, y1 = quad[i]
        x2, y2 = quad[(i + 1) % len(quad)]
        total += float(x1 * y2 - x2 * y1)
    return total


def test_order_quad_is_wound_like_the_destination_rectangle():
    """
    The ordering must wind the same way as the destination rectangle
    ``[[0,0], [w,0], [w,h], [0,h]]``; otherwise the warp mirrors the bay.
    """
    destination = np.array([[0, 0], [63, 0], [63, 63], [0, 63]], dtype=np.float32)
    expected_sign = np.sign(_signed_area(destination))

    quads = [
        np.array([[10, 10], [90, 20], [80, 70], [20, 60]], dtype=np.float32),
        np.array([[20, 60], [80, 70], [90, 20], [10, 10]], dtype=np.float32),   # reversed
        np.array([[50, 10], [90, 50], [50, 90], [10, 50]], dtype=np.float32),   # diamond
    ]
    for quad in quads:
        ordered = S._order_quad(quad)
        assert np.sign(_signed_area(ordered)) == expected_sign, ordered.tolist()


def test_order_quad_is_invariant_to_input_point_order():
    """The same four corners must deskew identically however they are listed."""
    quad = np.array([[10, 10], [90, 20], [80, 70], [20, 60]], dtype=np.float32)
    reference = S._order_quad(quad)
    for shift in range(4):
        rolled = np.roll(quad, shift, axis=0)
        assert np.allclose(S._order_quad(rolled), reference), f"shift {shift}"


# ==========================================================================
# Congestion bands — the brief fixes these exactly
# ==========================================================================
def test_congestion_band_boundaries():
    cases = [
        (0.0, "Low"), (20.0, "Low"), (39.9, "Low"),
        (40.0, "Moderate"), (60.0, "Moderate"), (75.0, "Moderate"),
        (75.1, "High"), (90.0, "High"), (100.0, "High"),
    ]
    for occupancy, expected in cases:
        assert I.classify_congestion(occupancy) == expected, f"{occupancy}% -> {expected}"


def test_every_congestion_level_has_a_recommendation():
    for level in ("Low", "Moderate", "High"):
        profile = C.CONGESTION_LEVELS[level]
        for key in ("colour", "emoji", "headline", "recommendation", "detail"):
            assert profile.get(key), f"{level} is missing {key}"


# ==========================================================================
# Score to prediction
# ==========================================================================
def test_score_above_threshold_is_occupied():
    prediction = I.score_to_prediction(1, 0.92, threshold=0.5)
    assert prediction.status == "occupied"
    assert prediction.is_occupied
    assert abs(prediction.confidence - 0.92) < 1e-6


def test_confidence_is_always_for_the_reported_status():
    empty = I.score_to_prediction(1, 0.10, threshold=0.5)
    assert empty.status == "empty"
    assert abs(empty.confidence - 0.90) < 1e-6, "an empty call reports 1 - P(occupied)"


def test_scores_near_the_threshold_are_flagged_uncertain():
    assert I.score_to_prediction(1, 0.52, threshold=0.5, uncertain_band=0.15).uncertain
    assert not I.score_to_prediction(1, 0.95, threshold=0.5, uncertain_band=0.15).uncertain


def test_threshold_is_inclusive_at_the_boundary():
    assert I.score_to_prediction(1, 0.50, threshold=0.50).status == "occupied"


# ==========================================================================
# Aggregation
# ==========================================================================
def _predictions(scores, threshold=0.5):
    return [I.score_to_prediction(i, s, threshold) for i, s in enumerate(scores, 1)]


def test_counts_and_percentage():
    insight = I.generate_insight(_predictions([0.9, 0.9, 0.9, 0.1]))
    assert insight.total_slots == 4
    assert insight.occupied_slots == 3
    assert insight.available_slots == 1
    assert abs(insight.occupancy_pct - 75.0) < 1e-6
    assert insight.congestion_level == "Moderate"


def test_available_always_complements_occupied():
    for scores in ([0.9] * 10, [0.1] * 10, [0.9, 0.1] * 7, [0.5] * 3):
        insight = I.generate_insight(_predictions(scores))
        assert insight.occupied_slots + insight.available_slots == insight.total_slots


def test_full_lot_produces_a_high_congestion_recommendation():
    insight = I.generate_insight(_predictions([0.99] * 20))
    assert insight.available_slots == 0
    assert insight.congestion_level == "High"
    assert "another area" in insight.recommendation.lower()
    assert any("no space is available" in note.lower() for note in insight.notes)


def test_empty_lot_recommends_proceeding():
    insight = I.generate_insight(_predictions([0.01] * 20))
    assert insight.occupancy_pct == 0.0
    assert insight.congestion_level == "Low"
    assert "proceed" in insight.recommendation.lower()


def test_almost_full_lot_warns_about_the_last_few_bays():
    insight = I.generate_insight(_predictions([0.99] * 97 + [0.01] * 3))
    assert insight.available_slots == 3
    assert any("may be taken" in note for note in insight.notes)


def test_no_bays_does_not_divide_by_zero():
    insight = I.generate_insight([])
    assert insight.total_slots == 0
    assert insight.occupancy_pct == 0.0
    assert insight.congestion_level == "Unknown"
    assert insight.recommendation


def test_insight_is_json_serialisable():
    json.dumps(I.generate_insight(_predictions([0.9, 0.1])).to_dict())


# ==========================================================================
# Zone guidance
# ==========================================================================
def _grid_predictions(rows=3, cols=6, occupied_rows=(0,)):
    predictions = []
    space_id = 1
    for row in range(rows):
        for col in range(cols):
            y1 = row / rows
            y2 = (row + 1) / rows
            x1 = col / cols
            x2 = (col + 1) / cols
            score = 0.95 if row in occupied_rows else 0.05
            predictions.append(
                I.score_to_prediction(
                    space_id, score,
                    polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                )
            )
            space_id += 1
    return predictions


def test_row_summary_splits_the_lot_into_zones():
    zones = I.row_summary(_grid_predictions(), n_rows=3)
    assert len(zones) == 3
    assert sum(z["total"] for z in zones) == 18


def test_best_zone_picks_the_emptiest():
    zones = I.row_summary(_grid_predictions(occupied_rows=(0,)), n_rows=3)
    target = I.best_zone(zones)
    assert target is not None
    assert target["available"] == 6, "zones 2 and 3 are fully free"


def test_best_zone_returns_none_when_the_lot_is_full():
    zones = I.row_summary(_grid_predictions(occupied_rows=(0, 1, 2)), n_rows=3)
    assert I.best_zone(zones) is None


def test_row_summary_needs_geometry():
    assert I.row_summary(_predictions([0.9, 0.1])) == []


# ==========================================================================
# Overlays
# ==========================================================================
def test_annotate_preserves_the_image_shape():
    image = np.full((240, 320, 3), 128, dtype=np.uint8)
    predictions = _grid_predictions(2, 3)
    output = O.annotate(image, predictions, I.generate_insight(predictions))
    assert output.shape == image.shape
    assert output.dtype == image.dtype


def test_annotate_does_not_mutate_the_input():
    image = np.full((240, 320, 3), 128, dtype=np.uint8)
    original = image.copy()
    O.annotate(image, _grid_predictions(2, 3))
    assert np.array_equal(image, original)


def test_overlay_paints_green_for_empty_and_red_for_occupied():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    empty = O.draw_overlay(image, _grid_predictions(1, 1, occupied_rows=()), fill_alpha=1.0)
    occupied = O.draw_overlay(image, _grid_predictions(1, 1, occupied_rows=(0,)), fill_alpha=1.0)
    assert empty[..., 1].sum() > empty[..., 0].sum(), "empty bays should be green"
    assert occupied[..., 0].sum() > occupied[..., 1].sum(), "occupied bays should be red"


def test_side_by_side_widens_the_canvas():
    a = np.zeros((100, 120, 3), dtype=np.uint8)
    b = np.zeros((100, 120, 3), dtype=np.uint8)
    assert O.side_by_side(a, b).shape[1] == 246   # 120 + 6 divider + 120


# ==========================================================================
# Layouts
# ==========================================================================
def test_grid_layout_produces_the_requested_number_of_bays():
    spaces = L.grid_layout(rows=4, cols=10)
    assert len(spaces) == 40
    assert all(len(s["points"]) == 4 for s in spaces)
    assert len({s["space_id"] for s in spaces}) == 40


def test_grid_layout_stays_inside_the_frame():
    for space in L.grid_layout(rows=3, cols=5):
        for x, y in space["points"]:
            assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


def test_grid_layout_bays_do_not_overlap():
    """Overlapping bays would let one car be counted twice."""
    spaces = L.grid_layout(rows=2, cols=3)
    boxes = [(min(p[0] for p in s["points"]), min(p[1] for p in s["points"]),
              max(p[0] for p in s["points"]), max(p[1] for p in s["points"]))
             for s in spaces]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            overlap_x = min(a[2], b[2]) - max(a[0], b[0])
            overlap_y = min(a[3], b[3]) - max(a[1], b[1])
            assert overlap_x <= 1e-9 or overlap_y <= 1e-9


def test_fingerprint_is_normalised_and_brightness_invariant():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (200, 320, 3), dtype=np.uint8)
    brighter = np.clip(image.astype(np.int16) + 40, 0, 255).astype(np.uint8)

    a = L.fingerprint(image)
    b = L.fingerprint(brighter)
    assert abs(np.linalg.norm(a) - 1.0) < 1e-4
    assert L.similarity(a, b) > 0.99, "a uniform brightness shift must not change the match"


def test_similarity_of_unrelated_images_is_near_the_midpoint():
    rng = np.random.default_rng(1)
    a = L.fingerprint(rng.integers(0, 255, (200, 320, 3), dtype=np.uint8))
    b = L.fingerprint(rng.integers(0, 255, (200, 320, 3), dtype=np.uint8))
    assert 0.3 < L.similarity(a, b) < 0.7


def test_match_layout_returns_nothing_for_an_empty_library():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    layout, score, scores = L.match_layout(image, {})
    assert layout is None and score == 0.0 and scores == []


# ==========================================================================
# Dataset integrity (skipped if the dataset has not been built)
# ==========================================================================
def test_splits_are_class_balanced_and_correctly_sized():
    if not (C.PROCESSED_DIR / "manifest.json").exists():
        return _skip("dataset not built")

    from src import dataset as DS
    counts = DS.split_counts()
    total = sum(v for split in counts.values() for v in split.values())

    for split, row in counts.items():
        assert row["empty"] == row["occupied"], f"{split} is not class balanced"
        assert row["empty"] >= 100, "the brief requires at least 100 images per class"

    for split, expected in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
        share = sum(counts[split].values()) / total
        assert abs(share - expected) < 0.02, f"{split} share is {share:.3f}, expected {expected}"


def test_no_calendar_day_is_shared_between_splits():
    manifest_path = C.PROCESSED_DIR / "manifest.json"
    if not manifest_path.exists():
        return _skip("dataset not built")

    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = {}
    for record in records:
        groups.setdefault((record["lot"], record["date"]), set()).add(record["split"])

    leaking = {key for key, splits in groups.items() if len(splits) > 1}
    assert not leaking, f"{len(leaking)} (lot, date) groups leak across splits"


def test_demo_scenes_come_only_from_test_days():
    manifest_path = C.PROCESSED_DIR / "manifest.json"
    if not (manifest_path.exists() and C.DEMO_INDEX_PATH.exists()):
        return _skip("dataset not built")

    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_of = {(r["lot"], r["date"]): r["split"] for r in records}

    for entry in json.loads(C.DEMO_INDEX_PATH.read_text(encoding="utf-8")):
        split = split_of.get((entry["lot"], entry["date"]))
        assert split in (None, "test"), (
            f"demo scene {entry['image']} comes from a {split} day"
        )


# ==========================================================================
# Trained model (skipped until a checkpoint exists)
# ==========================================================================
def test_model_predicts_slots_end_to_end():
    if not C.MODEL_PATH.exists() or not C.DEMO_INDEX_PATH.exists():
        return _skip("no trained model")

    import cv2
    from src import inference as INF

    entry = json.loads(C.DEMO_INDEX_PATH.read_text(encoding="utf-8"))[0]
    image = cv2.cvtColor(cv2.imread(str(C.DEMO_DIR / entry["image"])), cv2.COLOR_BGR2RGB)

    predictions, timing = INF.predict_slots(image, entry["spaces"])
    assert len(predictions) == timing["n_slots"] > 0
    assert all(0.0 <= p.score <= 1.0 for p in predictions)

    insight = I.generate_insight(predictions)
    assert insight.total_slots == len(predictions)
    assert insight.occupied_slots + insight.available_slots == insight.total_slots


def test_model_beats_chance_on_a_held_out_scene():
    if not C.MODEL_PATH.exists() or not C.DEMO_INDEX_PATH.exists():
        return _skip("no trained model")

    import cv2
    from src import inference as INF

    entry = json.loads(C.DEMO_INDEX_PATH.read_text(encoding="utf-8"))[0]
    image = cv2.cvtColor(cv2.imread(str(C.DEMO_DIR / entry["image"])), cv2.COLOR_BGR2RGB)

    predictions, _ = INF.predict_slots(image, entry["spaces"])
    truth = INF.accuracy_against_truth(predictions, entry["spaces"])
    assert truth is not None
    assert truth["accuracy"] > 0.80, f"only {truth['accuracy']:.2%} on an unseen scene"


# ==========================================================================
# Sensor-noise augmentation
# ==========================================================================
def _noise_batch(n: int = 64) -> np.ndarray:
    return np.random.default_rng(0).uniform(0, 255, (n, 32, 32, 3)).astype("float32")


def test_sensor_noise_is_the_identity_at_inference():
    # Any leak into inference would add grain to every real prediction.
    from src.model import RandomSensorNoise

    layer = RandomSensorNoise(seed=1)
    batch = _noise_batch()
    for flag in (False, None):
        assert np.array_equal(np.asarray(layer(batch, training=flag)), batch), \
            f"noise applied with training={flag}"


def test_sensor_noise_touches_part_of_the_batch_and_stays_in_range():
    from src.model import RandomSensorNoise

    layer = RandomSensorNoise(max_stddev=25.0, probability=0.5, seed=7)
    batch = _noise_batch(256)
    out = np.asarray(layer(batch, training=True))

    noised = sum(not np.array_equal(out[i], batch[i]) for i in range(len(batch)))
    # probability 0.5 over 256 images: anything outside 90-166 is a ~6-sigma event.
    assert 90 <= noised <= 166, f"{noised}/256 images noised — expected about half"
    assert out.min() >= 0.0 and out.max() <= 255.0, "noise pushed pixels out of [0, 255]"


def test_scripts_load_the_model_through_the_shared_loader():
    # The noise layer is a registered custom class, and registration happens
    # when src.model is imported. A script that calls keras' load_model directly
    # skips that import: 03_evaluate.py, 04_robustness_test.py and the notebook
    # all did, and none of them could open the retrained model. 02_train.py is
    # exempt because it builds the model, so src.model is always imported there.
    offenders = []
    candidates = [*sorted((C.PROJECT_ROOT / "scripts").glob("*.py")),
                  C.PROJECT_ROOT / "app.py"]
    for path in candidates:
        if path.name == "02_train.py":
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "models.load_model(" in line:
                offenders.append(f"{path.name}:{number}")
    assert not offenders, (
        "load through src.inference.load_model instead: " + ", ".join(offenders)
    )


def test_model_with_sensor_noise_round_trips_through_a_keras_file():
    # The layer is custom; if registration breaks, the app cannot load the model.
    import tempfile
    import keras
    from src.model import build_model

    model = build_model(input_shape=(32, 32, 3))
    path = Path(tempfile.mkdtemp()) / "noise_roundtrip.keras"
    model.save(path)
    reloaded = keras.models.load_model(path)

    names = [layer.name for layer in reloaded.get_layer("augmentation").layers]
    assert "aug_noise" in names, f"noise layer lost on reload: {names}"
    probe = _noise_batch(4)
    assert np.allclose(model.predict(probe, verbose=0), reloaded.predict(probe, verbose=0))


# ==========================================================================
# Runner
# ==========================================================================
_SKIPPED: list = []


def _skip(reason: str) -> None:
    _SKIPPED.append(reason)


def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]

    passed, failed, skipped = 0, [], 0
    for name, fn in tests:
        before = len(_SKIPPED)
        try:
            fn()
        except AssertionError as exc:
            failed.append((name, str(exc) or "assertion failed"))
            print(f"  FAIL  {name}\n          {exc}")
            continue
        except Exception as exc:                       # noqa: BLE001
            failed.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  ERROR {name}\n          {type(exc).__name__}: {exc}")
            continue

        if len(_SKIPPED) > before:
            skipped += 1
            print(f"  SKIP  {name}  ({_SKIPPED[-1]})")
        else:
            passed += 1
            print(f"  pass  {name}")

    print(f"\n{passed} passed, {skipped} skipped, {len(failed)} failed "
          f"(of {len(tests)} tests)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
