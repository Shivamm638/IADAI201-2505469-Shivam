"""
Evaluate on EVERY bay from the held-out days, not a 900-crop sample.

`scripts/03_evaluate.py` reports a balanced 900-crop test set: 450 empty and
450 occupied, capped per scene so no single photograph dominates. That is the
right headline metric — it is balanced, and it matches the sampling the brief
describes.

It is also small. A perfect score on 900 samples only bounds the true accuracy
at 99.59% with 95% confidence. The same fourteen held-out `(lot, date)` groups
contain **103,508 bays across 1,773 photographs**, every one of them from a
calendar day that appears in no other split. Scoring all of them does two
things the sampled test cannot:

  * tightens the interval by two orders of magnitude, and
  * gives the rare failure modes enough chances to actually show up.

It is also the *natural* distribution — 55% occupied, PUCPR-heavy — rather than
a balanced one, so it answers "what would this system do in the field?" where
the sampled test answers "is it equally good at both classes?". Both are
reported; neither replaces the other.

Memory: scenes are streamed one at a time and crops are discarded after
scoring. Holding all 103k crops resident would cost ~15 GB.

    python scripts/09_full_holdout_eval.py                 # everything
    python scripts/09_full_holdout_eval.py --max-scenes 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C
from src import data_prep as D
from src import inference as INF
from src import insights as I

OUT_PATH = C.RESULTS_DIR / "full_holdout_eval.json"
ERRORS_PATH = C.RESULTS_DIR / "full_holdout_errors.png"


def wilson_interval(successes: int, total: int, z: float = 1.959963985) -> tuple:
    """
    Wilson score interval.

    The exact Clopper-Pearson interval used in `03_evaluate.py` is fine for 900
    samples but needs scipy; Wilson is closed-form, has better coverage than the
    normal approximation, and is indistinguishable from exact at n > 10,000.
    """
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def test_day_groups() -> set:
    """The (lot, date) pairs the dataset builder assigned to the test split."""
    manifest = json.loads((C.PROCESSED_DIR / "manifest.json").read_text(encoding="utf-8"))
    return {(row["lot"], row["date"]) for row in manifest if row["split"] == "test"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-scenes", type=int, default=0,
                        help="stop after this many scenes (0 = all)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12,
                        help="parallel downloads for uncached photographs")
    args = parser.parse_args()

    samples_path = C.RAW_DIR / "samples.json"
    if not samples_path.exists():
        print(f"error: {samples_path} not found — run scripts/01_build_dataset.py first.",
              file=sys.stderr)
        return 1

    groups = test_day_groups()
    print(f"held-out day-groups: {len(groups)}")

    # ---- collect the scenes belonging to those days ---------------------
    print("scanning the annotation document…")
    scenes = []
    for sample in D.iter_samples(samples_path):
        filepath = sample.get("filepath", "")
        if not filepath:
            continue
        lot = str(sample.get("source", "unknown")).lower()
        date = D._date_of(sample)
        if (lot, date) not in groups:
            continue
        spaces = D._spaces_of(sample)
        if not spaces:
            continue
        scenes.append({
            "filepath": filepath, "lot": lot, "date": date,
            "weather": D._weather_of(sample), "time": D._time_of(sample),
            "spaces": spaces,
        })
        if args.max_scenes and len(scenes) >= args.max_scenes:
            break

    total_bays = sum(len(s["spaces"]) for s in scenes)
    print(f"scenes: {len(scenes):,}   bays: {total_bays:,}")

    # ---- make sure the photographs are on disk --------------------------
    print("fetching photographs (cached ones are skipped)…")
    started = time.perf_counter()
    scenes = D.download_scenes(scenes, workers=args.workers, verbose=True)
    print(f"  {len(scenes):,} available after download "
          f"({time.perf_counter() - started:.0f}s)")

    threshold = INF.load_threshold()
    print(f"loading model (threshold {threshold:.2f})…")
    model = INF.load_model()

    # ---- score, one scene at a time -------------------------------------
    import cv2

    tp = tn = fp = fn = 0
    by_lot = defaultdict(lambda: [0, 0])          # [correct, total]
    by_weather = defaultdict(lambda: [0, 0])
    by_truth = defaultdict(lambda: [0, 0])
    scene_rows = []
    worst_errors = []                              # (confidence, crop, meta)
    all_errors = []                                # every disagreement
    error_crops = []                               # aligned with all_errors
    skipped_blank = 0

    started = time.perf_counter()
    for index, scene in enumerate(scenes, 1):
        image = cv2.imread(scene["local_path"])
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        patches, kept = INF.crops_from_spaces(image, scene["spaces"])
        if len(patches) == 0:
            continue

        # A featureless crop is a bay hidden behind a tree or a pole; the
        # dataset builder rejects these at build time for exactly this reason,
        # so they are excluded here too rather than scored against a label that
        # no image content supports.
        stds = patches.reshape(len(patches), -1).std(axis=1)
        usable = stds >= C.MIN_CROP_STD
        skipped_blank += int((~usable).sum())
        if not usable.any():
            continue
        patches, kept = patches[usable], [k for k, u in zip(kept, usable) if u]

        scores = model.predict(patches, batch_size=args.batch_size, verbose=0).ravel()
        predicted_occupied = scores >= threshold
        actual_occupied = np.array([k["label"] == "occupied" for k in kept])

        tp += int((predicted_occupied & actual_occupied).sum())
        tn += int((~predicted_occupied & ~actual_occupied).sum())
        fp += int((predicted_occupied & ~actual_occupied).sum())
        fn += int((~predicted_occupied & actual_occupied).sum())

        correct = predicted_occupied == actual_occupied
        for bucket, key in ((by_lot, scene["lot"]), (by_weather, scene["weather"])):
            bucket[key][0] += int(correct.sum())
            bucket[key][1] += len(correct)
        for label, mask in (("occupied", actual_occupied), ("empty", ~actual_occupied)):
            by_truth[label][0] += int(correct[mask].sum())
            by_truth[label][1] += int(mask.sum())

        # scene-level: the number a driver actually feels
        predicted_free = int((~predicted_occupied).sum())
        actual_free = int((~actual_occupied).sum())
        n = len(correct)
        scene_rows.append({
            "filepath": scene["filepath"], "lot": scene["lot"],
            "weather": scene["weather"], "date": scene["date"], "time": scene["time"],
            "n_bays": n,
            "slot_accuracy": float(correct.mean()),
            "predicted_free": predicted_free, "actual_free": actual_free,
            "free_error": abs(predicted_free - actual_free),
            "congestion_correct": (
                I.classify_congestion(100.0 * int(predicted_occupied.sum()) / n)
                == I.classify_congestion(100.0 * int(actual_occupied.sum()) / n)
            ),
        })

        # Every disagreement is kept, not just the most confident ones. The
        # high-confidence set is biased by construction — a model that is very
        # sure and "wrong" is exactly where a mislabelled bay shows up — so an
        # unbiased random sample is needed to estimate how many of these are
        # label errors rather than model errors.
        for position in np.flatnonzero(~correct):
            confidence = float(max(scores[position], 1 - scores[position]))
            meta = {"scene": Path(scene["filepath"]).name,
                    "lot": scene["lot"], "weather": scene["weather"],
                    "space_id": int(kept[position].get("space_id", -1)),
                    "truth": "occupied" if actual_occupied[position] else "empty",
                    "score": round(float(scores[position]), 4),
                    "confidence": round(confidence, 4)}
            all_errors.append(meta)
            error_crops.append(patches[position].astype(np.uint8))
            if len(worst_errors) < 24 or confidence > worst_errors[-1][0]:
                worst_errors.append((confidence, patches[position].astype(np.uint8), meta))
                worst_errors.sort(key=lambda r: -r[0])
                del worst_errors[24:]

        if index % 50 == 0 or index == len(scenes):
            done = tp + tn + fp + fn
            elapsed = time.perf_counter() - started
            print(f"  {index:>5,}/{len(scenes):,} scenes  {done:>7,} bays  "
                  f"acc {(tp + tn) / max(1, done):.5f}  "
                  f"{elapsed:.0f}s  ({done / max(elapsed, 1e-9):.0f} bays/s)",
                  flush=True)

    total = tp + tn + fp + fn
    if total == 0:
        print("error: no bays scored.", file=sys.stderr)
        return 1

    accuracy = (tp + tn) / total
    low, high = wilson_interval(tp + tn, total)
    errors = fp + fn

    print()
    print("=== full held-out evaluation ===")
    print(f"  scenes scored      {len(scene_rows):,}")
    print(f"  bays scored        {total:,}   (blank crops skipped: {skipped_blank:,})")
    print(f"  accuracy           {accuracy:.6f}   95% CI [{low:.6f}, {high:.6f}]")
    print(f"  errors             {errors:,}  ({fp:,} empty called occupied, "
          f"{fn:,} occupied called empty)")
    if tp + fp:
        print(f"  precision/recall   {tp / (tp + fp):.6f} / {tp / max(1, tp + fn):.6f}")
    print(f"  confusion          [[{tn} {fp}] [{fn} {tp}]]  (rows = actual empty/occupied)")

    print("\n  accuracy by lot:")
    for key, (ok, n) in sorted(by_lot.items()):
        print(f"    {key:<9} {ok / n:.6f}  (n={n:,})")
    print("  accuracy by weather:")
    for key, (ok, n) in sorted(by_weather.items()):
        print(f"    {key:<9} {ok / n:.6f}  (n={n:,})")
    print("  accuracy by true class:")
    for key, (ok, n) in sorted(by_truth.items()):
        print(f"    {key:<9} {ok / n:.6f}  (n={n:,})")

    free_errors = [r["free_error"] for r in scene_rows]
    perfect = sum(1 for r in scene_rows if r["slot_accuracy"] == 1.0)
    congestion_ok = sum(1 for r in scene_rows if r["congestion_correct"])
    print(f"\n  scenes with every bay correct   {perfect:,} / {len(scene_rows):,} "
          f"({100 * perfect / len(scene_rows):.1f}%)")
    print(f"  congestion level correct        {congestion_ok:,} / {len(scene_rows):,} "
          f"({100 * congestion_ok / len(scene_rows):.1f}%)")
    print(f"  mean free-bay counting error    {np.mean(free_errors):.3f} bays")
    print(f"  worst free-bay error            {max(free_errors)} bays")

    report = {
        "n_scenes": len(scene_rows),
        "n_bays": total,
        "blank_crops_skipped": skipped_blank,
        "threshold": threshold,
        "accuracy": accuracy,
        "accuracy_ci95_wilson": [low, high],
        "errors": errors,
        "false_positives_empty_called_occupied": fp,
        "false_negatives_occupied_called_empty": fn,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "precision_occupied": tp / (tp + fp) if tp + fp else None,
        "recall_occupied": tp / (tp + fn) if tp + fn else None,
        "by_lot": {k: {"accuracy": v[0] / v[1], "n": v[1]} for k, v in by_lot.items()},
        "by_weather": {k: {"accuracy": v[0] / v[1], "n": v[1]} for k, v in by_weather.items()},
        "by_true_class": {k: {"accuracy": v[0] / v[1], "n": v[1]} for k, v in by_truth.items()},
        "scenes_all_bays_correct": perfect,
        "congestion_correct": congestion_ok,
        "mean_free_bay_error": float(np.mean(free_errors)),
        "max_free_bay_error": int(max(free_errors)),
        "worst_errors": [meta for _c, _crop, meta in worst_errors],
        "all_errors": all_errors,
    }
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")

    _plot_errors(worst_errors)
    _plot_random_errors(all_errors, error_crops)
    return 0


def _plot_random_errors(all_errors: list, error_crops: list, sample: int = 24) -> None:
    """
    An UNBIASED sample of disagreements, for judging how many are label errors.

    The "most confident mistakes" sheet is useful but self-selecting: a model
    that is certain and nominally wrong is exactly the signature of a wrongly
    labelled bay, so that sheet over-states how often the label is at fault.
    This one draws uniformly at random from every disagreement, with a fixed
    seed, so what it shows can be counted.
    """
    if not all_errors:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(C.RANDOM_SEED)
    picks = rng.choice(len(all_errors), size=min(sample, len(all_errors)), replace=False)

    columns = 6
    rows = (len(picks) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(2.3 * columns, 2.75 * rows))
    for axis in np.atleast_1d(axes).ravel():
        axis.axis("off")
    for axis, index in zip(np.atleast_1d(axes).ravel(), picks):
        meta = all_errors[index]
        axis.imshow(error_crops[index])
        axis.set_title(
            f"{meta['lot'].upper()} {meta['weather']}" + chr(10)
            + f"label: {meta['truth']}" + chr(10)
            + f"P(occ)={meta['score']:.3f}",
            fontsize=7,
        )
    figure.suptitle(
        f"Random sample of {len(picks)} of {len(all_errors)} disagreements "
        "(seeded — for label-error auditing)", fontsize=11)
    figure.tight_layout()
    path = C.RESULTS_DIR / "full_holdout_random_errors.png"
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")


def _plot_errors(worst_errors: list) -> None:
    """A contact sheet of the model's most confident mistakes."""
    if not worst_errors:
        print("no errors to plot — nothing was misclassified.")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = worst_errors[:12]
    columns = 4
    rows = (len(shown) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 3.5 * rows))
    for axis in np.atleast_1d(axes).ravel():
        axis.axis("off")
    for axis, (confidence, crop, meta) in zip(np.atleast_1d(axes).ravel(), shown):
        axis.imshow(crop)
        axis.set_title(
            f"{meta['lot'].upper()} bay {meta['space_id']} · {meta['weather']}\n"
            f"truth {meta['truth']} · P(occ)={meta['score']:.3f}",
            fontsize=8,
        )
    figure.suptitle(
        "Most confident mistakes across every held-out bay", fontsize=12)
    figure.tight_layout()
    figure.savefig(ERRORS_PATH, dpi=110, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {ERRORS_PATH}")


if __name__ == "__main__":
    raise SystemExit(main())
