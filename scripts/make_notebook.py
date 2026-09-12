"""
Generate notebooks/ParkVision_AI_end_to_end.ipynb.

The notebook is generated rather than hand-edited so that it can never drift
out of step with the modules under src/: it imports the same code the scripts
and the Streamlit app use, instead of restating it.

    python scripts/make_notebook.py            # write the notebook
    python scripts/make_notebook.py --execute  # write it and run every cell
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "ParkVision_AI_end_to_end.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


def build() -> nbf.NotebookNode:
    cells = []

    cells.append(md("""
# ParkVision AI — Intelligent Urban Parking Analytics

**Scenario 1 · UrbanFlow AI Pvt. Ltd. · CRS Artificial Intelligence — Machine Learning and Deep Learning**

This notebook walks through the complete system, end to end: the problem, the
data, the model, the insight logic, the evaluation and the deployed dashboard.

It **imports** the project's modules rather than re-implementing them, so every
number below is produced by exactly the code that runs in the Streamlit app.

| Step | Covered in |
|---|---|
| 1 · Understand the problem | §1 · [`docs/01_problem_definition.md`](../docs/01_problem_definition.md) |
| 2 · Data collection and preprocessing | §2 · [`src/data_prep.py`](../src/data_prep.py) |
| 3 · Model development | §3 · [`src/model.py`](../src/model.py) |
| 4 · Insight logic | §4 · [`src/insights.py`](../src/insights.py) |
| 5 · Streamlit dashboard | §5 · [`app.py`](../app.py) |
| 6 · Testing and feedback | §6 · [`scripts/04_robustness_test.py`](../scripts/04_robustness_test.py) |
| 7 · Deployment | §7 · [`docs/06_deployment.md`](../docs/06_deployment.md) |
"""))

    cells.append(code("""
import json, sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

from src import config as C
from src import insights as I
from src import layouts as L
from src import overlay as O
from src import slots as S

pd.set_option("display.width", 120)
plt.rcParams["figure.dpi"] = 110

print("project root:", ROOT)
print("image size  :", C.IMG_SIZE)
print("classes     :", C.CLASS_NAMES)
"""))

    # ---------------------------------------------------------------- step 1
    cells.append(md("""
---
## 1 · Understanding the problem

**Input** → one photograph of a parking lot from a fixed elevated camera, plus
that camera's surveyed bay layout.

**Output** → per-bay status (Occupied / Empty), total / occupied / available
counts, occupancy percentage, congestion level, and a recommendation.

The task is deliberately framed at **slot level**. A model that reports "this
lot looks busy" helps nobody; a driver needs a count, and a city needs a count
it can act on.

Congestion bands are fixed by the brief:

| Occupancy | Level |
|---|---|
| `< 40 %` | Low |
| `40 – 75 %` | Moderate |
| `> 75 %` | High |
"""))

    cells.append(code("""
# The decision rules are pure functions - no model needed to verify them.
for pct in [0, 12.5, 39.9, 40.0, 60.0, 75.0, 75.1, 92.0, 100.0]:
    print(f"{pct:6.1f}% occupancy  ->  {I.classify_congestion(pct)}")
"""))

    # ---------------------------------------------------------------- step 2
    cells.append(md("""
---
## 2 · Data collection and preprocessing

**Dataset — PKLot** (Almeida et al., *Expert Systems with Applications*, 2015):
12,417 photographs of three Brazilian parking lots, every five minutes, across
sunny, overcast and rainy days, with a hand-drawn quadrilateral and an
occupied/empty label for every bay.

Built by `python scripts/01_build_dataset.py`.
"""))

    cells.append(code("""
report = json.loads((C.RESULTS_DIR / "dataset_report.json").read_text(encoding="utf-8"))

print(f"source          : {report['source_dataset']}")
print(f"photographs     : {report['photographs_indexed']:,} indexed")
print(f"distinct scenes : {report['candidate_scenes']:,} kept after de-duplication")
print(f"scenes used     : {report['scenes_downloaded']}")
print(f"lots            : {report['lots']}")
print(f"weather         : {report['weather']}")
print(f"split strategy  : {report['split_strategy']}")
print()
print(pd.DataFrame(report["crops_per_split"]).T.rename_axis("split")
        .assign(total=lambda d: d.sum(axis=1)))
print(f"\\ntotal slot crops: {report['total_crops']:,}  "
      f"({report['total_crops'] // len(C.CLASS_NAMES):,} per class)")
"""))

    cells.append(md("""
### 2.1 Exploratory analysis of the source dataset

`scene_summary.csv` holds one row per photograph, so the whole of PKLot can be
characterised before deciding what to sample from it.
"""))

    cells.append(code("""
summary = pd.read_csv(C.RAW_DIR / "scene_summary.csv")
print(f"{len(summary):,} photographs, {summary['n_spaces'].sum():,} labelled bays\\n")
display(summary.head())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

summary["weather"].value_counts().plot.bar(ax=axes[0], color="#2563EB", rot=0)
axes[0].set_title("Photographs by weather"); axes[0].set_ylabel("photographs")

summary.groupby("lot")["n_spaces"].mean().plot.bar(ax=axes[1], color="#7C3AED", rot=0)
axes[1].set_title("Mean annotated bays per lot"); axes[1].set_ylabel("bays")

axes[2].hist(summary["occupancy_pct"], bins=40, color="#0891B2")
axes[2].set_title("Occupancy across all photographs")
axes[2].set_xlabel("occupancy %"); axes[2].set_ylabel("photographs")

for ax in axes:
    ax.grid(alpha=.25)
fig.suptitle("PKLot — what the source data actually contains", fontsize=13)
fig.tight_layout(); plt.show()
"""))

    cells.append(code("""
# Occupancy follows the working day: near-empty at dawn, full by early afternoon.
# This is exactly why crops are quota-filled class-balanced rather than sampled
# uniformly - uniform sampling would tie "empty" to "early morning light".
summary["hour"] = pd.to_datetime(summary["time"], format="%H:%M:%S").dt.hour
by_hour = summary.groupby(["lot", "hour"])["occupancy_pct"].mean().unstack(0)

ax = by_hour.plot(figsize=(10, 4), linewidth=2, marker="o", markersize=3)
ax.set_title("Mean occupancy through the day, per lot")
ax.set_xlabel("hour of day"); ax.set_ylabel("occupancy %")
ax.axhspan(0, 40, color="#16A34A", alpha=.07)
ax.axhspan(40, 75, color="#D97706", alpha=.07)
ax.axhspan(75, 100, color="#DC2626", alpha=.07)
ax.grid(alpha=.25); plt.show()
"""))

    cells.append(md("""
### 2.2 Perspective-correct cropping

Bays are quadrilaterals, not rectangles: an elevated camera projects a
rectangular bay to a trapezium. Cropping the axis-aligned bounding box would
pull in slices of both neighbours, so a car parked *next door* could make an
empty bay look occupied.

`src/slots.py` warps each quadrilateral to a 224×224 square instead, padded by
8% so an overhanging bumper stays inside the patch.
"""))

    cells.append(code("""
import cv2

demo = json.loads(C.DEMO_INDEX_PATH.read_text(encoding="utf-8"))
scene = max(demo, key=lambda e: e["n_spaces"])
image = cv2.cvtColor(cv2.imread(str(C.DEMO_DIR / scene["image"])), cv2.COLOR_BGR2RGB)
h, w = image.shape[:2]

print(f"{scene['image']}  -  {scene['lot'].upper()}, {scene['weather']}, "
      f"{scene['n_spaces']} bays, {scene['occupancy_pct_truth']:.0f}% occupied")

# Ground-truth geometry drawn on the full scene.
truth = [I.SlotPrediction(space_id=s["space_id"], status=s["label"],
                          confidence=1.0, score=float(s["label"] == "occupied"),
                          polygon=s["points"])
         for s in scene["spaces"]]

fig, ax = plt.subplots(figsize=(13, 7))
ax.imshow(O.annotate(image, truth, I.generate_insight(truth), mark_uncertain=False))
ax.set_title("PKLot ground truth - green = empty, red = occupied", fontsize=12)
ax.axis("off"); plt.show()
"""))

    cells.append(code("""
# The same bays, deskewed exactly as the model receives them.
empties  = [s for s in scene["spaces"] if s["label"] == "empty"][:6]
occupied = [s for s in scene["spaces"] if s["label"] == "occupied"][:6]

fig, axes = plt.subplots(2, 6, figsize=(14, 5))
for row, (group, name) in enumerate([(empties, "EMPTY"), (occupied, "OCCUPIED")]):
    for col, space in enumerate(group):
        polygon = S.polygon_to_pixels(space["points"], w, h)
        axes[row, col].imshow(S.crop_slot(image, polygon, C.IMG_SIZE))
        axes[row, col].axis("off")
        if col == 0:
            axes[row, col].set_ylabel(name)
    axes[row, 0].set_title(name, loc="left", fontsize=11, fontweight="bold")
fig.suptitle("224x224 perspective-corrected bay crops - the model's actual input", fontsize=13)
fig.tight_layout(); plt.show()
"""))

    cells.append(md("""
### 2.3 Augmentation

Applied to the **training split only**, as layers inside the model graph, so
each epoch sees fresh randomisation and the deployed model cannot accidentally
augment at inference time.

Vertical flip is deliberately excluded: it would put a car's roof where its
shadow belongs — something a fixed overhead camera can never produce.
"""))

    cells.append(code("""
from IPython.display import Image as IPyImage
IPyImage(filename=str(C.RESULTS_DIR / "augmentation_examples.png"))
"""))

    cells.append(md("""
### 2.4 The split — and the leakage trap

PKLot frames are five minutes apart, so two frames of the same lot on the same
afternoon are near-identical. Splitting crops **at random** would put a crop and
its near-twin on opposite sides of the train/test boundary; the model could then
score ~99.9% on test by memorising, and the number would mean nothing.

Whole **`(lot, calendar date)`** groups are therefore assigned to a split.
"""))

    cells.append(code("""
manifest = pd.DataFrame(json.loads((C.PROCESSED_DIR / "manifest.json").read_text(encoding="utf-8")))

print("crops per split and class")
display(pd.crosstab(manifest["split"], manifest["label"], margins=True))

print("\\nno calendar day is shared between splits:")
days = manifest.groupby(["lot", "date"])["split"].nunique()
print(f"  {(days > 1).sum()} of {len(days)} (lot, date) groups appear in more than one split")

print("\\ncrops by weather and split")
display(pd.crosstab(manifest["weather"], manifest["split"]))
"""))

    # ---------------------------------------------------------------- step 3
    cells.append(md("""
---
## 3 · Model development

**MobileNetV2** pre-trained on ImageNet, with the classifier head replaced by
global-average-pool → dropout → a single sigmoid unit.

Trained in two phases:

| Phase | Epochs | Learning rate | What trains |
|---|---|---|---|
| 1 | 12 | `1e-3` | The new head only; backbone frozen |
| 2 | 10 | `1e-5` | Head + MobileNetV2 from layer 100 up (BatchNorm stays frozen) |

Fine-tuning immediately would push the randomly initialised head's large
gradients straight into the pre-trained filters and destroy them, so the head
is stabilised first.
"""))

    cells.append(code("""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from src import inference as INF

# The shared loader registers the custom noise layer before loading.
model = INF.load_model(C.MODEL_PATH)
model.summary()

size_mb = C.MODEL_PATH.stat().st_size / 1e6
print(f"\\nsaved model: {size_mb:.1f} MB  -  small enough for Streamlit Community Cloud")
"""))

    cells.append(code("""
history = json.loads(C.HISTORY_PATH.read_text(encoding="utf-8"))
summary_json = json.loads((C.RESULTS_DIR / "training_summary.json").read_text(encoding="utf-8"))

print(json.dumps(summary_json, indent=2))

epochs = range(1, len(history["loss"]) + 1)
switch = summary_json["epochs_head"]

fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
for ax, (tr, va, title) in zip(axes, [("accuracy", "val_accuracy", "Accuracy"),
                                      ("loss", "val_loss", "Loss"),
                                      ("auc", "val_auc", "ROC AUC")]):
    ax.plot(epochs, history[tr], label="train", linewidth=2)
    ax.plot(epochs, history[va], label="validation", linewidth=2)
    ax.axvline(switch + .5, color="#888", linestyle="--", linewidth=1)
    ax.set_title(title); ax.set_xlabel("epoch"); ax.grid(alpha=.25); ax.legend()
fig.suptitle("Two-phase transfer learning (dashed line = fine-tuning starts)", fontsize=13)
fig.tight_layout(); plt.show()
"""))

    cells.append(md("""
### 3.1 Evaluation

The decision threshold is swept over the **validation** split and then applied
**unchanged** to test. Tuning the threshold on test and reporting test accuracy
would be reporting a number we had fitted.
"""))

    cells.append(code("""
metrics = json.loads(C.METRICS_PATH.read_text(encoding="utf-8"))
test = metrics["test"]

print(f"operating threshold : {metrics['operating_threshold']:.2f} (chosen on validation)\\n")
headline = pd.DataFrame({
    "metric": ["Accuracy", "Balanced accuracy", "Precision (occupied)",
               "Recall (occupied)", "F1 (occupied)", "ROC AUC",
               "Average precision", "Matthews corr."],
    "value":  [test["accuracy"], test["balanced_accuracy"], test["precision_occupied"],
               test["recall_occupied"], test["f1_occupied"], test["roc_auc"],
               test["average_precision"], test["matthews_corrcoef"]],
})
display(headline.style.format({"value": "{:.4f}"}).hide(axis="index"))
print(f"errors: {test['errors']} of {test['n_samples']} held-out bays")
print((C.RESULTS_DIR / "classification_report.txt").read_text(encoding="utf-8"))
"""))

    cells.append(code("""
IPyImage(filename=str(C.RESULTS_DIR / "confusion_matrix.png"))
"""))

    cells.append(code("""
IPyImage(filename=str(C.RESULTS_DIR / "roc_pr_curves.png"))
"""))

    cells.append(md("""
### 3.2 Where the model is weak

An average accuracy hides the failure modes that matter operationally. Because
`manifest.json` records the lot, weather and date of every crop, accuracy can be
broken down by slice.
"""))

    cells.append(code("""
by_weather = pd.DataFrame(metrics["by_weather"]).T
by_lot = pd.DataFrame(metrics["by_lot"]).T
print("accuracy by weather condition"); display(by_weather)
print("accuracy by parking lot");       display(by_lot)
IPyImage(filename=str(C.RESULTS_DIR / "accuracy_by_slice.png"))
"""))

    cells.append(code("""
# The most confident mistakes are the informative ones.
IPyImage(filename=str(C.RESULTS_DIR / "misclassified_examples.png"))
"""))

    # ---------------------------------------------------------------- step 4
    cells.append(md("""
---
## 4 · Insight logic — from 100 numbers to one sentence

`src/insights.py` imports neither TensorFlow nor OpenCV: the decision rules are
pure functions over plain numbers, so they can be verified exactly.
"""))

    cells.append(code("""
from src import inference as INF

threshold = INF.load_threshold()
predictions, timing = INF.predict_slots(image, scene["spaces"], model=model, threshold=threshold)
insight = I.generate_insight(predictions)

print(f"{timing['n_slots']} bays analysed in {timing['total_seconds']:.2f}s "
      f"({timing['ms_per_slot']:.1f} ms per bay)\\n")
print(f"Total slots      {insight.total_slots}")
print(f"Occupied         {insight.occupied_slots}")
print(f"Available        {insight.available_slots}")
print(f"Occupancy        {insight.occupancy_pct:.1f} %")
print(f"Congestion       {insight.congestion_level}  -  {insight.headline}")
print(f"Recommendation   {insight.recommendation}")
print(f"Mean confidence  {insight.mean_confidence * 100:.1f} %")
for note in insight.notes:
    print(f"  note: {note}")
"""))

    cells.append(code("""
# How close is that to the truth? This scene comes from a day the model never saw.
check = INF.accuracy_against_truth(predictions, scene["spaces"])
display(pd.DataFrame({
    "": ["Occupied bays", "Available bays"],
    "predicted":    [check["predicted_occupied"], check["predicted_available"]],
    "ground truth": [check["true_occupied"],      check["true_available"]],
}))
print(f"slot-level accuracy on this unseen photograph: "
      f"{check['n_correct']}/{check['n_compared']} = {check['accuracy'] * 100:.1f}%")
"""))

    cells.append(code("""
# Zone guidance - "twelve free" is not useful without "which way do I turn?"
zones = I.row_summary(predictions)
display(pd.DataFrame(zones))
target = I.best_zone(zones)
print(f"-> Head for {target['zone']} ({target['position'].lower()}): "
      f"{target['available']} of {target['total']} bays free"
      if target else "-> No zone has a free bay; try another lot.")
"""))

    cells.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(17, 6))
axes[0].imshow(image); axes[0].set_title("Input photograph", fontsize=12); axes[0].axis("off")
axes[1].imshow(O.annotate(image, predictions, insight))
axes[1].set_title(f"ParkVision AI - {insight.available_slots} of {insight.total_slots} bays free "
                  f"({insight.congestion_level} congestion)", fontsize=12)
axes[1].axis("off")
fig.tight_layout(); plt.show()
"""))

    # ---------------------------------------------------------------- step 5/6/7
    cells.append(md("""
---
## 5 · Streamlit dashboard

```bash
streamlit run app.py
```

The dashboard wraps everything above in an interface aimed at a non-technical
driver: upload or pick a scene, see the annotated lot, read four metric cards
and a colour-coded recommendation, then drill into zones, confidence, a
bay-by-bay table with CSV export, and an accuracy panel.

Where the bays are, for an uploaded image, is resolved in three steps —
automatic camera recognition by visual fingerprint, manual camera selection,
or a clearly-labelled approximate grid.
"""))

    cells.append(code("""
library = L.load_library() or L.build_library()
for lot, layout in sorted(library.items()):
    print(f"{lot:<8} {layout.n_spaces:>4} surveyed bays   {layout.label}")

matched, score, all_scores = L.match_layout(image, library)
print(f"\\nfingerprint match for this scene: {matched.lot.upper()} at {score * 100:.1f}%")
print("all camera scores:", {k: round(v, 3) for k, v in all_scores})
"""))

    cells.append(md("""
---
## 6 · Testing and feedback

Two kinds of test, because "does it work?" has two answers — see
[`results/robustness_report.md`](../results/robustness_report.md).

1. **Scene level.** The complete pipeline over every held-out photograph,
   comparing the free-bay count a driver would act on against ground truth.
2. **Stress testing.** The test split re-run through simulated dusk, night,
   glare, flat light, harsh light, a knocked camera, motion blur, heavy blur,
   sensor noise and aggressive JPEG compression.
"""))

    cells.append(code("""
robust = json.loads((C.RESULTS_DIR / "robustness.json").read_text(encoding="utf-8"))

scenes_df = pd.DataFrame(robust["scene_test"])[[
    "image", "lot", "weather", "predicted_available", "true_available",
    "available_error", "predicted_congestion", "true_congestion",
    "congestion_correct", "slot_accuracy", "seconds",
]]
display(scenes_df)

print(f"mean slot accuracy across scenes : {scenes_df['slot_accuracy'].mean() * 100:.2f}%")
print(f"mean free-bay counting error     : {scenes_df['available_error'].mean():.2f} bays")
print(f"congestion level correct         : {scenes_df['congestion_correct'].sum()}"
      f"/{len(scenes_df)} scenes")
print(f"mean latency                     : {scenes_df['seconds'].mean():.2f} s per photograph")
"""))

    cells.append(code("""
stress = pd.DataFrame(robust["stress_test"])
display(stress[["group", "condition", "meaning", "accuracy", "delta_vs_baseline"]])
IPyImage(filename=str(C.RESULTS_DIR / "robustness_chart.png"))
"""))

    cells.append(md("""
---
## 7 · Deployment

Pushed to GitHub and deployed on Streamlit Community Cloud —
see [`docs/06_deployment.md`](../docs/06_deployment.md).

`tensorflow-cpu` and `opencv-python-headless` are pinned deliberately: the full
TensorFlow wheel can exceed the free tier's build limits, and non-headless
OpenCV fails on a container with no X server.

---

## Conclusion

| Requirement | Delivered |
|---|---|
| Slot-level occupancy detection | MobileNetV2 over perspective-corrected bay crops |
| Real-time availability insights | Total / occupied / available / occupancy %, in one batched prediction |
| Visual smart overlays | Green / red / amber polygons with fill, outline and legend |
| Parking utilisation analytics | Low / Moderate / High congestion, plus per-zone breakdown |
| Intelligent recommendations | Proceed, proceed promptly, or try another area — with situational notes |
| Interactive Streamlit interface | Four-section dashboard, exports, accuracy panel, adjustable threshold |

**What makes the reported accuracy trustworthy:** the train/validation/test
split is grouped by whole calendar days, so every figure in this notebook was
measured on photographs taken on days the model has never seen.

**Known limits, stated plainly:** bay geometry must be surveyed once per camera;
night-time performance is the weakest measured condition and PKLot contains no
true night imagery; a car straddling two bays is counted in both.
"""))

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    }
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                        help="run every cell and store the outputs")
    args = parser.parse_args()

    notebook = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    if args.execute:
        from nbclient import NotebookClient

        client = NotebookClient(
            notebook,
            timeout=1800,
            kernel_name="python3",
            resources={"metadata": {"path": str(OUT.parent)}},
        )
        print("executing notebook…")
        client.execute()

    nbf.write(notebook, str(OUT))
    print(f"wrote {OUT}  ({len(notebook.cells)} cells)")


if __name__ == "__main__":
    main()
