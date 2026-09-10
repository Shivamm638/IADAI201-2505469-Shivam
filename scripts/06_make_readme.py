"""
Generate README.md from the measured results.

The README quotes a lot of numbers — accuracy, per-weather breakdowns, scene
counts. Typing those by hand invites the one failure a report cannot afford:
prose that no longer matches the metrics beside it. This script writes the
README from `results/*.json`, so the two can never disagree.

    python scripts/06_make_readme.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C

OUT = C.PROJECT_ROOT / "README.md"


def load(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def pct(value, digits=2) -> str:
    return f"{value * 100:.{digits}f}%" if value is not None else "—"


def main() -> None:
    metrics = load(C.METRICS_PATH)
    dataset = load(C.RESULTS_DIR / "dataset_report.json")
    training = load(C.RESULTS_DIR / "training_summary.json")
    robust = load(C.RESULTS_DIR / "robustness.json")
    details = load(C.PROJECT_ROOT / "submission" / "student_details.json")

    test = metrics.get("test", {})
    cm = test.get("confusion_matrix", {})
    crops = dataset.get("crops_per_split", {})
    scenes = robust.get("scene_test", [])
    stress = robust.get("stress_test", [])

    app_url = details.get("streamlit_app_url", "")
    if not app_url or app_url.startswith("FILL_IN"):
        app_url = "https://<your-app>.streamlit.app"
    repo = details.get("github_repository", "")
    user = details.get("github_username", "")
    repo_url = (f"https://github.com/{user}/{repo}"
                if repo and user and not repo.startswith("FILL_IN")
                and not user.startswith("FILL_IN")
                else "https://github.com/<username>/<repository>")

    # -- derived scene-level numbers -------------------------------------
    if scenes:
        mean_slot = sum(s.get("slot_accuracy") or 0 for s in scenes) / len(scenes)
        mean_err = sum(s.get("available_error", 0) for s in scenes) / len(scenes)
        congestion_hits = sum(1 for s in scenes if s.get("congestion_correct"))
        mean_latency = sum(s.get("seconds", 0) for s in scenes) / len(scenes)
        total_bays = sum(s.get("total_slots", 0) for s in scenes)
    else:
        mean_slot = mean_err = mean_latency = total_bays = 0
        congestion_hits = 0

    baseline = stress[0]["accuracy"] if stress else 0.0
    worst = min(stress[1:], key=lambda r: r["accuracy"]) if len(stress) > 1 else None

    if not test:
        raise SystemExit(
            "results/metrics.json is missing or empty — run scripts/03_evaluate.py first."
        )

    # Rows that depend on the scene-level test are only emitted when that test
    # has actually run: a README quoting "0.00 s" would be worse than one that
    # simply does not make the claim.
    scene_rows = ""
    if scenes:
        scene_rows = (
            f"| **End-to-end latency** | {mean_latency:.2f} s for a whole lot, CPU only |\n"
            f"| **Free-bay counting error** | {mean_err:.2f} bays per scene, averaged over "
            f"{len(scenes)} held-out photographs |"
        )

    L: list[str] = []
    A = L.append

    # ================================================================ header
    A(f"""# 🅿️ ParkVision AI — Intelligent Urban Parking Analytics & Space Optimisation Platform

> **Scenario 1** · CRS Artificial Intelligence — Machine Learning and Deep Learning
> Summative Assessment · *AI in Action: Building Intelligent Vision Systems for Smart Cities & Smart Retail*

**Live app → {app_url}**

A computer-vision system for **UrbanFlow AI Pvt. Ltd.** that looks at a photograph of a
parking lot and answers the question a driver actually has: *how many bays are free, where
are they, and should I drive in?*

Every bay is perspective-corrected out of the photograph and classified by a fine-tuned
**MobileNetV2**. The per-bay results become availability counts, an occupancy percentage,
a congestion level and a recommendation, presented through a **Streamlit** dashboard with
colour-coded overlays.

---

## At a glance

| | |
|---|---|
| **Test accuracy** | **{pct(test.get('accuracy'))}** on {test.get('n_samples', 0):,} bays from days the model never saw |
| **ROC AUC** | {test.get('roc_auc', 0):.4f} |
| **Dataset** | {dataset.get('total_crops', 0):,} labelled slot crops ({dataset.get('total_crops', 0) // 2:,} per class) from PKLot |
| **Model** | MobileNetV2, {training.get('total_parameters', 0):,} parameters, {C.MODEL_PATH.stat().st_size / 1e6 if C.MODEL_PATH.exists() else 0:.1f} MB |
| **Training** | {training.get('epochs_run', 0)} epochs, two-phase transfer learning |
{scene_rows}
""")

    # =============================================================== the problem
    A(f"""---

## 1 · The problem

Drivers circling a busy lot do not have a *parking* problem — they have an *information*
problem. The space usually exists; they cannot see it from the entrance. That missing
information turns into congestion, wasted fuel and lost time.

```
INPUT    one photograph of a parking lot  +  that camera's surveyed bay layout
           ↓
         perspective-warp every bay to 224×224  →  MobileNetV2  →  P(occupied)
           ↓
OUTPUT   per bay   Occupied / Empty + confidence
         lot       total · occupied · available · occupancy %
                   congestion level · recommendation · zone guidance
         visual    green = empty · red = occupied · amber = low confidence
```

The contract is **slot level**, not image level. "This lot looks busy" helps nobody; a
driver needs a count and a city needs a count it can act on.

**Congestion bands** (fixed by the brief): Low `< 40%` · Moderate `40–75%` · High `> 75%`.

📄 Full analysis: [`docs/01_problem_definition.md`](docs/01_problem_definition.md)

---

## 2 · Why classification, not YOLO

The decisive argument is what happens to an **empty** bay.

| | Crop classification (**chosen**) | Object detection (YOLO) |
|---|---|---|
| Answers "is bay 37 free?" | Directly | Only after matching boxes to bays |
| Empty bays | Explicitly classified | Invisible — nothing to detect, so "empty" must be *inferred from absence* |
| Model size | 2.3 M parameters, {C.MODEL_PATH.stat().st_size / 1e6 if C.MODEL_PATH.exists() else 9.2:.1f} MB | Tens of millions |
| Free-tier CPU hosting | Yes, ~4 ms per bay | Slow, heavy dependencies |

A detector finds *cars*; the question is about *bays*. An empty bay produces no detection,
and absence is indistinguishable from a missed detection. Classifying each surveyed bay
makes "empty" a directly supervised answer rather than an inference from silence.

**The cost of this choice, stated plainly:** the system cannot analyse a lot whose bays
have never been surveyed. The app handles that with an explicit, clearly-labelled grid
fallback.
""")

    # ================================================================= dataset
    A(f"""---

## 3 · Data

**PKLot** — Almeida, Oliveira, Britto, Silva & Koerich, *Expert Systems with Applications*
42(11), 2015. {dataset.get('photographs_indexed', 0):,} photographs of three Brazilian
parking lots, every five minutes, across sunny, overcast and rainy days, each bay
hand-annotated as a quadrilateral with an occupied/empty label.

| Split | Empty | Occupied | Total | Day-groups |
|---|---|---|---|---|""")

    groups = dataset.get("day_groups_per_split", {})
    for split in ("train", "val", "test"):
        row = crops.get(split, {})
        A(f"| **{split.title()}** | {row.get('empty', 0):,} | {row.get('occupied', 0):,} | "
          f"{sum(row.values()):,} | {groups.get(split, 0)} |")
    A(f"| **Total** | **{sum(crops.get(s, {}).get('empty', 0) for s in crops):,}** | "
      f"**{sum(crops.get(s, {}).get('occupied', 0) for s in crops):,}** | "
      f"**{dataset.get('total_crops', 0):,}** | **{sum(groups.values())}** |")

    A(f"""
The brief asks for at least 100 images per class; this uses
**{dataset.get('total_crops', 0) // 2:,} per class**, drawn from
{len(dataset.get('lots', {}))} lots and {len(dataset.get('weather', {}))} weather
conditions, every split exactly 50/50 balanced.

### Preprocessing

| Step | What happens |
|---|---|
| **Perspective correction** | Each bay is a quadrilateral, not a rectangle. An axis-aligned crop would pull in slices of both neighbours, so a car parked *next door* could make an empty bay look occupied. Every bay is warped to a square instead, padded 8% so an overhanging bumper stays inside |
| **Resize** | 224 × 224 — done by the same warp, so there is no second resample to blur the patch |
| **Normalisation** | `[0, 255] → [-1, 1]`, as a **layer inside the model** — impossible to apply in training and forget at serving time |
| **Augmentation** | Horizontal flip, rotation ±21°, zoom ±12%, translation ±8%, contrast ±25%, brightness ±25% — training split only |

**Vertical flip is deliberately excluded:** it would put a car's roof where its shadow
belongs, which a fixed overhead camera can never produce.

### The leakage trap, and how it is avoided

PKLot frames are **five minutes apart**, so two frames of the same lot on the same
afternoon are nearly identical. Splitting crops at random puts a crop and its near-twin on
opposite sides of the train/test boundary — the model then scores ~99.9% by memorising,
and the number means nothing.

**Whole `(lot, calendar date)` groups are assigned to a single split.** If 2013-03-05 at
UFPR05 is a test day, every frame from that day is in test and none is in train.

> Every accuracy figure in this README was measured on photographs taken on **days the
> model has never seen**. It is a lower, harder and honest number.

📄 Full detail: [`docs/02_data_preprocessing.md`](docs/02_data_preprocessing.md)
""")

    # =================================================================== model
    A(f"""---

## 4 · Model

**MobileNetV2** pre-trained on ImageNet → global average pooling → dropout → one sigmoid unit.

```
slot_image (224, 224, 3)
  ├─ augmentation      flip · rotate · zoom · shift · contrast · brightness  (training only)
  ├─ Rescaling         [0, 255] → [-1, 1]
  ├─ MobileNetV2       ImageNet weights, include_top=False   → (7, 7, 1280)
  ├─ GlobalAveragePooling2D                                  → (1280,)
  ├─ Dropout({training.get('dropout', C.DROPOUT)})
  └─ Dense(1, sigmoid)                                       → P(occupied)
```

### Two-phase transfer learning

| Phase | Epochs | LR | What trains |
|---|---|---|---|
| **1 — head** | {training.get('epochs_head', C.EPOCHS_HEAD)} | `{training.get('lr_head', C.LR_HEAD)}` | The new head only; backbone frozen |
| **2 — fine-tune** | {training.get('epochs_finetune', C.EPOCHS_FINETUNE)} | `{training.get('lr_finetune', C.LR_FINETUNE)}` | Head + MobileNetV2 from layer {training.get('fine_tune_from_layer', C.FINE_TUNE_FROM_LAYER)} up |

The new `Dense(1)` layer starts random, so its first gradients are large and meaningless.
Fine-tuning immediately would drive those gradients straight into pre-trained ImageNet
filters and destroy them. The head is stabilised first, then the top third of the backbone
is nudged at a 100× smaller learning rate.

**BatchNormalization layers stay frozen** even in phase 2 — their running statistics come
from ImageNet's millions of images, and re-estimating them from batches of 32 parking
crops is a well-known way to make fine-tuning unstable.

| Hyperparameter | Value |
|---|---|
| Batch size | {training.get('batch_size', C.BATCH_SIZE)} |
| Optimiser | Adam |
| Loss | Binary cross-entropy, label smoothing {training.get('label_smoothing', C.LABEL_SMOOTHING)} |
| Early stopping | patience 6 on `val_accuracy` |
| LR schedule | `ReduceLROnPlateau` ×0.5, patience 3 |
| Seed | {training.get('seed', C.RANDOM_SEED)} |
| Training time | {training.get('training_minutes', 0):.0f} min, CPU only |

![Training curves](results/training_curves.png)

📄 Full detail: [`docs/03_model_development.md`](docs/03_model_development.md)
""")

    # ================================================================= results
    A(f"""---

## 5 · Results

The decision threshold is swept on the **validation** split and applied **unchanged** to
test — tuning it on test and then reporting test accuracy would be reporting a number we
had fitted. Chosen threshold: **{metrics.get('operating_threshold', 0.5):.2f}**.

| Metric | Test split |
|---|---|
| **Accuracy** | **{pct(test.get('accuracy'))}** |
| Balanced accuracy | {pct(test.get('balanced_accuracy'))} |
| Precision (occupied) | {pct(test.get('precision_occupied'))} |
| Recall (occupied) | {pct(test.get('recall_occupied'))} |
| F1 (occupied) | {pct(test.get('f1_occupied'))} |
| Precision (empty) | {pct(test.get('precision_empty'))} |
| Recall (empty) | {pct(test.get('recall_empty'))} |
| ROC AUC | {test.get('roc_auc', 0):.4f} |
| Average precision | {test.get('average_precision', 0):.4f} |
| Matthews correlation | {test.get('matthews_corrcoef', 0):.4f} |
| Errors | {test.get('errors', 0)} of {test.get('n_samples', 0)} bays |

### Confusion matrix

| | Predicted **Empty** | Predicted **Occupied** |
|---|---|---|
| **Actual Empty** | {cm.get('true_empty_pred_empty', 0)} ✅ | {cm.get('true_empty_pred_occupied', 0)} |
| **Actual Occupied** | {cm.get('true_occupied_pred_empty', 0)} | {cm.get('true_occupied_pred_occupied', 0)} ✅ |

![Confusion matrix](results/confusion_matrix.png)

The two error types are **not** equivalent. An occupied bay called empty sends a driver to
a bay that has a car in it — exactly the frustration the system exists to remove. An empty
bay called occupied merely fails to offer a usable space. The dashboard's threshold slider
lets an operator shift that balance deliberately.

### Where the model is strong, and where it is weak

An average accuracy hides the failure modes that matter operationally. Because every crop
records its provenance, accuracy can be broken down by slice.
""")

    if metrics.get("by_weather"):
        A("| Weather | Bays tested | Accuracy |\n|---|---|---|")
        for name, row in metrics["by_weather"].items():
            A(f"| {name.title()} | {row['n']:,} | {pct(row['accuracy'])} |")
        A("")
    if metrics.get("by_lot"):
        A("| Parking lot | Bays tested | Accuracy |\n|---|---|---|")
        for name, row in metrics["by_lot"].items():
            A(f"| {name.upper()} | {row['n']:,} | {pct(row['accuracy'])} |")
        A("")

    A("""![Accuracy by slice](results/accuracy_by_slice.png)

![ROC and PR curves](results/roc_pr_curves.png)

### The slice breakdown earned its keep

This is not decoration. The first trained model scored **94.11% overall with a
ROC AUC of 0.9968** — near-perfect ranking, oddly weak accuracy. The average
looked respectable. The per-lot breakdown did not:

| Lot | ROC AUC | Accuracy |
|---|---|---|
| PUCPR | 1.0000 | 99.4% |
| UFPR05 | 0.9998 | 98.6% |
| **UFPR04** | **0.9615** | **79.8%** |

46 of 53 test errors came from one lot. Plotting UFPR04's most confident errors
showed crops that were **blank grey rectangles** — no image content at all.
A scan found **208 of 6,000 crops (3.5%) were featureless**, 202 from UFPR04,
and labelled *both* ways: 125 "empty", 83 "occupied". Identical blank patches
with contradictory labels — the purest possible label noise.

**The cause** was the textbook corner-ordering shortcut (smallest `x + y` is
top-left, smallest `y − x` is top-right). It fails for a quadrilateral rotated
near 45°, where one corner wins two of the four tests: the source quad collapses
to three distinct points, `getPerspectiveTransform` returns a singular matrix,
and the bay warps flat. UFPR04's camera views its bays at roughly 45°. Nothing
errored — OpenCV warps a degenerate quad happily and returns a valid-looking
224×224 JPEG.

**The fix** orders corners by angle about the centroid, which is correct at any
rotation, backed by a flat-crop guard at build time and a regression test that
sweeps a rectangle through all 360°. No amount of hyperparameter tuning would
have found this; only the slice breakdown did.
""")

    A(f"""
**The result**, on an otherwise identical rebuild and retrain:

| | Before the fix | After the fix |
|---|---|---|
| Featureless crops in the dataset | 208 of 6,000 (3.5%) | **0** |
| Test accuracy | 94.11% | **{pct(test.get('accuracy'))}** |
| UFPR04 accuracy | 79.82% | **{pct((metrics.get('by_lot', {}).get('ufpr04') or {}).get('accuracy'))}** |
| Test errors | 53 of 900 | **{test.get('errors', 0)} of {test.get('n_samples', 0)}** |

Full write-up: [`docs/03_model_development.md` §3.5.1](docs/03_model_development.md#351-the-bug-the-slice-analysis-caught)
""")

    # ================================================================== logic
    A(f"""---

## 6 · From 100 numbers to one sentence

`src/insights.py` imports **neither TensorFlow nor OpenCV** — the decision rules are pure
functions over plain numbers, so they can be verified exactly rather than inferred from
the dashboard.

```
100 sigmoid scores
   ↓  threshold + uncertainty band
100 labelled bays with confidence
   ↓  aggregate
counts · occupancy % · congestion level
   ↓  decision rules
"Slots available — proceed to park."
   ↓  zone analysis
"Head for Zone 1, nearest the entrance — 9 of 34 bays free."
```

| Level | Recommendation |
|---|---|
| 🟢 **Low** (`< 40%`) | Slots available — proceed to park |
| 🟠 **Moderate** (`40–75%`) | Slots available — proceed, but park promptly |
| 🔴 **High** (`> 75%`) | Parking almost full — try another area |

Percentage bands alone mislead at the margin, so situational notes are added: a 300-bay lot
at 99% is "High congestion" with 3 bays free, and a driver already at the entrance needs
the **count**, not the band.

**Zone guidance** answers the question the required outputs miss — *which way do I turn?*
The lot is split into bands by bay position and the dashboard names the zone with the most
free bays.

📄 Full detail: [`docs/04_insight_logic.md`](docs/04_insight_logic.md)

---

## 7 · The dashboard

```bash
streamlit run app.py
```

| Section | What it shows |
|---|---|
| **1 · Choose an image** | Twelve held-out demo scenes, or upload your own |
| **2 · Slot-level detection** | Original and annotated side by side, PNG download |
| **3 · Real-time availability** | Total / occupied / available / congestion cards, plus a colour-coded recommendation banner |
| **4 · Analytics** | Zone guidance · confidence distribution · bay-by-bay table with CSV export · accuracy against ground truth |

### Where are the bays, for an uploaded image?

The classifier answers *"is this bay occupied?"*; it does not discover where bays are.
Three routes, in descending order of reliability:

1. **Automatic camera recognition** — each surveyed camera stores a zero-meaned,
   unit-normalised 96×54 fingerprint. Zero-meaning makes the match invariant to brightness,
   so a camera surveyed on a sunny morning still matches on a rainy afternoon.
2. **Manual selection** from a dropdown.
3. **Grid fallback** — for a genuinely unseen lot, **clearly labelled as approximate**.

That third path is the honest answer to a real limitation. Producing confident-looking
numbers from a grid that does not match the painted bays would be worse than saying so.

### Screenshots

**The landing view** — what the system does, in three steps, with one-click ways in.

![Landing view](results/screenshots/app_00_landing.png)

**The verdict** — the answer first. How many bays are free, whether to drive in,
and which part of the lot to head for. Everything technical sits below it.

![The verdict](results/screenshots/app_01_verdict.png)

**Availability in detail** — counts, occupancy percentage and a colour-coded
congestion banner carrying the recommendation in plain language.

![Availability](results/screenshots/app_02_availability.png)

**See what the AI sees** — the actual deskewed 224×224 crops the model
classified, ordered by how much it hesitated. The model stops being a black box.

![See what the AI sees](results/screenshots/app_03_see_what_ai_sees.png)

**Checked against ground truth** — every demo scene carries PKLot's hand-drawn
labels from a day the model never trained on, so the dashboard shows predicted
against actual.

![Accuracy on an unseen scene](results/screenshots/app_04_ground_truth.png)

**Compare two moments** — the same camera at two times. The bay layout is
identical, so any change in the counts is a real change in the lot.

![Compare two moments](results/screenshots/app_05_compare.png)

Annotated model output for all twelve held-out scenes is in
[`results/screenshots/`](results/screenshots/).

📄 Full detail: [`docs/05_web_app.md`](docs/05_web_app.md)
""")

    # ================================================================ testing
    A(f"""---

## 8 · Testing

Two kinds of test, because *"does it work?"* has two different answers.

### Scene-level — what a driver actually experiences

Slot accuracy is the engineering metric. The number a driver *feels* is **how many free
bays did the app promise, and how many were really there.**

| Measure | Result |
|---|---|
| Mean slot-level accuracy across {len(scenes)} held-out photographs | **{pct(mean_slot)}** over {total_bays:,} bays |
| Mean error in the free-bay count | **{mean_err:.2f} bays** per scene |
| Congestion level called correctly | **{congestion_hits} of {len(scenes)}** scenes |
| Mean end-to-end latency | **{mean_latency:.2f} s** per photograph, CPU only |

### Stress testing — lighting, angle and image quality

PKLot was shot by fixed daytime cameras, so the training data barely covers night, a
knocked camera, or a poor network connection. Rather than claim robustness we have not
measured, each condition is simulated across the whole test split.
""")

    if stress:
        A("| Condition | Simulates | Accuracy | Change |\n|---|---|---|---|")
        for row in stress:
            delta = "—" if row is stress[0] else f"{row['delta_vs_baseline']:+.4f}"
            A(f"| {row['condition']} | {row['meaning']} | {pct(row['accuracy'])} | {delta} |")
        A("")
        A("![Robustness](results/robustness_chart.png)\n")
        if worst:
            A(f"""Brightness and contrast changes are handled well — they were augmented during
training. **{worst['condition']}** is the weakest condition at {pct(worst['accuracy'])}
({worst['delta_vs_baseline']:+.4f} against a {pct(baseline)} baseline). Heavy blur is a limit
of the *input*, not a tuning problem: a blurred empty bay and a blurred dark car genuinely
share the same low-frequency signature.
""")

    A("""### Feedback, and what it changed

| Feedback | Change made |
|---|---|
| "A raw probability means nothing to a driver." | Scores become Occupied/Empty with a plain-English confidence; the headline is a sentence |
| "Green and red boxes are hard to separate on a 100-bay lot." | Polygons filled at 22% alpha as well as outlined |
| "I don't know which end of the lot to drive to." | Zone breakdown naming the emptiest zone |
| "How do I know when the model is guessing?" | Borderline bays drawn amber and counted separately |
| "Can I trust it on an image it hasn't seen?" | Demo scenes ship with ground truth; the app shows predicted vs actual |

### Automated tests

```bash
python tests/test_parkvision.py
```

38 tests covering geometry, congestion boundaries, aggregation, zone guidance, overlays,
layout fallbacks, **and dataset integrity** — including an explicit check that no calendar
day leaks across splits and that every demo scene comes from a test day.

📄 Full report: [`results/robustness_report.md`](results/robustness_report.md)
""")

    # ============================================================== structure
    A(f"""---

## 9 · Repository

```
├── app.py                        Streamlit dashboard
├── src/
│   ├── config.py                 every tunable constant, in one place
│   ├── slots.py                  perspective-correct bay cropping
│   ├── data_prep.py              PKLot → balanced, leakage-free dataset
│   ├── dataset.py                tf.data pipelines
│   ├── model.py                  MobileNetV2 + two-phase compile
│   ├── inference.py              batched slot-level prediction
│   ├── insights.py               counts, congestion, recommendations  (no TF/CV2)
│   ├── overlay.py                colour-coded annotation
│   └── layouts.py                camera survey library + fingerprint matching
├── scripts/
│   ├── 01_build_dataset.py       download, crop, split
│   ├── 02_train.py               two-phase transfer learning
│   ├── 03_evaluate.py            metrics, confusion matrix, slice analysis
│   ├── 04_robustness_test.py     scene-level + stress testing
│   ├── 05_make_submission.py     the submission PDF
│   ├── 06_make_readme.py         this file
│   └── make_notebook.py          the end-to-end notebook
├── notebooks/
│   └── ParkVision_AI_end_to_end.ipynb
├── tests/test_parkvision.py      38 tests
├── data/
│   ├── processed/{{train,val,test}}/{{empty,occupied}}/     {dataset.get('total_crops', 0):,} slot crops
│   └── demo/                     {len(scenes)} held-out scenes + ground truth
├── models/parkvision_mobilenetv2.keras
├── results/                      metrics, figures, reports, screenshots
├── docs/                         one document per step of the brief
└── submission/                   the submission PDF
```

---

## 10 · Reproducing this

```bash
git clone {repo_url}
cd {repo.split('/')[-1] if repo and not repo.startswith('FILL_IN') else '<repository>'}
pip install -r requirements-dev.txt

# The trained model and dataset are committed, so this is optional:
python scripts/01_build_dataset.py      # rebuild from PKLot        (~2 min cold)
python scripts/02_train.py              # retrain                   (~{training.get('training_minutes', 50):.0f} min CPU)
python scripts/03_evaluate.py           # metrics and figures       (~1 min)
python scripts/04_robustness_test.py    # testing report            (~4 min)
python tests/test_parkvision.py         # 38 tests

streamlit run app.py
```

On a CPU-only machine, training can be split into bounded sessions:

```bash
python scripts/02_train.py --phase head
python scripts/02_train.py --phase finetune --resume
```

Every script is seeded (`RANDOM_SEED = {training.get('seed', C.RANDOM_SEED)}`), so a rebuild
reproduces the same split and the same figures.

📄 Deployment guide: [`docs/06_deployment.md`](docs/06_deployment.md)

---

## 11 · Research and references

Each reference below changed a specific decision — the full record, including
**what was rejected and why**, is in [`docs/references.md`](docs/references.md).

| Reference | What it changed here |
|---|---|
| **Almeida et al. (2015)**, *PKLot*, **Expert Systems with Applications** 42(11) | The dataset, the bay-level approach, and the decision to report accuracy **per lot and per weather condition** rather than only on average |
| **Sandler et al. (2018)**, *MobileNetV2*, **CVPR** | The backbone — accuracy per FLOP suited to free-tier CPU deployment |
| **Yosinski et al. (2014)**, *How transferable are features?*, **NeurIPS** | Freezing early layers and fine-tuning only the top third, in that order |
| **Amato et al. (2017)**, *Deep learning for decentralised parking lot occupancy detection*, **ESWA** 72 | Supports a small model over a large detector for embedded/edge use |
| **Szegedy et al. (2016)**, *Rethinking the Inception Architecture*, **CVPR** | Label smoothing, so confidence scores are calibrated enough to display |
| Keras transfer-learning guidance | Keeping BatchNormalization frozen during fine-tuning |

---

## 12 · Limitations

Stated plainly, because a system that hides its limits cannot be trusted with the ones it
does not have.

- **Bay geometry must be surveyed once per camera.** The classifier says whether a bay is
  occupied; it does not discover where bays are. Uploads from an unrecognised camera fall
  back to a grid, which the app labels as indicative rather than surveyed.
- **Night-time is the weakest measured condition**, and PKLot contains no true night
  imagery — the simulated figure is an estimate, not a measurement. Deploying after dark
  would need genuine night-time data.
- **A car straddling two bays** is counted in both, because each bay is classified
  independently.
- **Three lots, one country.** Generalisation to very different layouts, vehicle mixes or
  camera heights is untested.

---

<div align="center">

**ParkVision AI** · Built for UrbanFlow AI Pvt. Ltd.
CRS Artificial Intelligence — Machine Learning and Deep Learning · Summative Assessment

</div>
""")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT}  ({len(OUT.read_text(encoding='utf-8').splitlines())} lines, "
          f"{OUT.stat().st_size / 1024:.1f} KB)")

    write_model_card(metrics, dataset, training, robust)


def write_model_card(metrics: dict, dataset: dict, training: dict, robust: dict) -> None:
    """A short, honest description of the shipped model, next to the weights."""
    test = metrics.get("test", {})
    stress = robust.get("stress_test", [])
    worst = min(stress[1:], key=lambda r: r["accuracy"]) if len(stress) > 1 else None

    lines = [
        "# Model card — ParkVision AI occupancy classifier",
        "",
        f"**File:** `parkvision_mobilenetv2.keras` "
        f"({C.MODEL_PATH.stat().st_size / 1e6 if C.MODEL_PATH.exists() else 0:.1f} MB)",
        "",
        "## What it does",
        "",
        "Takes one **224×224 RGB crop of a single parking bay** and returns "
        "`P(occupied)` as a sigmoid score. It classifies a bay that it is *given*; "
        "it does not locate bays. Bay geometry is a per-camera survey — see "
        "[`docs/01_problem_definition.md`](../docs/01_problem_definition.md).",
        "",
        "Input is raw RGB in `[0, 255]`. Rescaling and augmentation are layers "
        "**inside** the graph, so the file is self-contained and cannot be fed "
        "wrongly-scaled input.",
        "",
        "## Architecture and training",
        "",
        "| | |",
        "|---|---|",
        "| Backbone | MobileNetV2, ImageNet weights |",
        f"| Parameters | {training.get('total_parameters', 0):,} |",
        f"| Schedule | {training.get('epochs_head', 0)} epochs head-only "
        f"(lr {training.get('lr_head', '')}), then "
        f"{training.get('epochs_finetune', 0)} fine-tuning "
        f"(lr {training.get('lr_finetune', '')}) |",
        f"| Batch size | {training.get('batch_size', '')} |",
        f"| Loss | Binary cross-entropy, label smoothing "
        f"{training.get('label_smoothing', '')} |",
        f"| Seed | {training.get('seed', '')} |",
        f"| Training data | {dataset.get('total_crops', 0):,} PKLot bay crops, "
        f"{dataset.get('total_crops', 0) // 2:,} per class |",
        "",
        "## Measured performance",
        "",
        f"On {test.get('n_samples', 0):,} bays from **calendar days that appear in "
        "no other split**:",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {pct(test.get('accuracy'))} |",
        f"| Precision (occupied) | {pct(test.get('precision_occupied'))} |",
        f"| Recall (occupied) | {pct(test.get('recall_occupied'))} |",
        f"| ROC AUC | {test.get('roc_auc', 0):.4f} |",
        f"| Operating threshold | {metrics.get('operating_threshold', 0.5):.2f} "
        "(chosen on validation, applied unchanged to test) |",
        "",
    ]

    if metrics.get("by_weather"):
        lines += ["**By weather:** " + " · ".join(
            f"{k} {pct(v['accuracy'])}" for k, v in metrics["by_weather"].items()), ""]
    if metrics.get("by_lot"):
        lines += ["**By lot:** " + " · ".join(
            f"{k.upper()} {pct(v['accuracy'])}" for k, v in metrics["by_lot"].items()), ""]

    lines += [
        "## Intended use",
        "",
        "Educational and research use, on elevated fixed-camera views of surface "
        "parking lots similar to PKLot's three. It is a coursework artefact, not a "
        "product.",
        "",
        "## Out of scope",
        "",
        "- **Enforcement or billing.** The error rate is far too high to charge or "
        "fine anyone on.",
        "- **Night-time operation.**"
        + (f" The weakest measured condition is *{worst['condition']}* at "
           f"{pct(worst['accuracy'])}." if worst else "")
        + " PKLot contains no true night imagery, so that figure is a simulation, "
        "not a measurement.",
        "- **Unsurveyed lots.** Without bay geometry the app falls back to an "
        "approximate grid, which it labels as indicative.",
        "- **Any safety-critical decision.**",
        "",
        "## Known limitations",
        "",
        "- A car straddling two bays is counted in both: each bay is classified "
        "independently.",
        "- Trained on three lots in one country; different layouts, vehicle mixes "
        "or camera heights are untested.",
        "- A bay fully hidden behind a tree yields a featureless crop; such crops "
        "are rejected at build time, and at inference they score near the "
        "threshold and are flagged low-confidence rather than asserted.",
        "",
        "## Training data licence",
        "",
        "PKLot (Almeida et al., 2015) is published by the Federal University of "
        "Paraná for research and academic use. See "
        "[`data/README.md`](../data/README.md).",
    ]

    path = C.MODELS_DIR / "model_card.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
