# Step 2 — Data Collection and Preprocessing

*Everything below is produced by `python scripts/01_build_dataset.py`.*

---

## 2.1 The dataset: PKLot

**PKLot — A robust dataset for parking lot classification**
Almeida, Oliveira, Britto, Silva & Koerich, *Expert Systems with Applications*
42(11), 2015. Mirrored on the Hugging Face Hub by Voxel51 as
[`Voxel51/PKLot`](https://huggingface.co/datasets/Voxel51/PKLot).

PKLot is the standard public benchmark for exactly this task, and it matches
the brief's description of the required data almost word for word — thousands
of parking images captured under sunny, cloudy and rainy conditions, each bay
labelled occupied or empty.

| Property | Value |
|---|---|
| Photographs | 12,417 (12,142 usable, with at least one labelled bay) |
| Parking lots | 3 — PUCPR, UFPR04, UFPR05 |
| Weather conditions | sunny, cloudy (overcast), rainy |
| Capture interval | one frame every 5 minutes, daylight hours |
| Period | 2012–2013 |
| Annotation per bay | rotated quadrilateral + `occupied` flag |
| Labelled bays in total | ≈ 695,000 |
| Image resolution | 1280 × 720 |
| Licence | free for academic and research use |

**Why PKLot rather than a generic car dataset.** A car-detection dataset
teaches a model what a car looks like. PKLot teaches it what an *occupied bay*
looks like — including the hard negative cases that break naive systems: an
empty bay in deep shadow, a wet bay reflecting a car from the next row, a bay
partly hidden behind a lamp post. Those cases are the entire difficulty of the
problem, and they only exist in a dataset annotated at bay level.

---

## 2.2 Selecting a subset — and why not all of it

The brief asks for **at least 100 images per class**. Using all ~695,000
labelled bays would be neither necessary nor honest: consecutive PKLot frames
are five minutes apart and nearly identical, so most of that volume is
redundant rather than informative.

The selection is therefore built around **diversity, not volume**:

1. **One or two frames per (lot, date, weather) bucket.** Where two are kept,
   they are forced to come from different hours of the day, so morning and
   afternoon lighting are both represented. This alone reduces 12,142
   photographs to **274 genuinely distinct scenes**.
2. **Round-robin across (lot × weather × split) buckets** when capping the
   working set, so every lot and every weather condition survives the cap
   instead of whichever one sorts first alphabetically.
3. **Capped contribution per scene.** No single photograph may dominate a
   class quota, so a nearly-full lot cannot supply most of the "occupied"
   examples.

**Result — 3,000 crops per class, 6,000 in total**, thirty times the required
minimum, drawn from all three lots and all three weather conditions.

| | Candidate scenes | Share |
|---|---|---|
| PUCPR | 86 | 31.4% |
| UFPR04 | 88 | 32.1% |
| UFPR05 | 100 | 36.5% |
| **Sunny** | 137 | 50.0% |
| **Cloudy** | 83 | 30.3% |
| **Rainy** | 54 | 19.7% |

---

## 2.3 Preprocessing pipeline

### a. Perspective-correct cropping — not a bounding box

Each bay is annotated as a quadrilateral. Cropping its axis-aligned bounding
box would include large slices of the neighbouring bays, so a car parked next
door could make an empty bay look occupied.

Instead (`src/slots.py`):

1. Order the four corners **by their angle about the centroid**, then roll the
   ordering to start at the corner nearest the top-left bound.
2. Expand the quadrilateral by **8%** about its centre, so a car's roof or
   bumper overhanging the painted line still falls inside the patch.
3. Compute the perspective transform onto a square and warp with `INTER_AREA`.
4. Reject the crop if its pixel standard deviation is below `MIN_CROP_STD` — a
   bay hidden behind a tree yields a featureless patch that is identical
   whatever its label says, so it is noise rather than a hard example.

Every bay therefore arrives at the model **deskewed and at a consistent scale**,
whether it sits in the front row or eighty metres away at the back.

> **Step 1 is not the obvious implementation, and the obvious one is wrong.**
> The usual shortcut — smallest `x + y` is top-left, smallest `y − x` is
> top-right — breaks for quadrilaterals rotated near 45°, where one corner wins
> two of the four tests. The source quad collapses to three distinct points,
> `getPerspectiveTransform` returns a singular matrix, and the bay warps to a
> flat grey rectangle that still looks like a valid 224×224 JPEG. It poisoned
> 3.5% of the first dataset build before the per-lot accuracy breakdown exposed
> it. The full story is in
> [Step 3, §3.5.1](03_model_development.md#351-the-bug-the-slice-analysis-caught).

### b. Resize to 224 × 224

The warp targets 224 × 224 directly, the size the brief specifies and the native
input resolution of MobileNetV2. There is no second resample: the single warp
does both the deskew and the resize, which avoids compounding interpolation blur.

### c. Normalisation

Pixel values are rescaled from `[0, 255]` to `[-1, 1]`, the range MobileNetV2's
ImageNet weights expect. **This rescaling is a layer inside the model**
(`src/model.py`), not a step in the data script. That is deliberate: it makes it
impossible to normalise during training and forget to at serving time — the
single most common source of silent train/serve skew in transfer-learning
projects.

### d. Augmentation

Applied to the **training split only**, as layers inside the model graph so
each epoch sees fresh randomisation:

| Technique | Setting | Real-world condition it simulates |
|---|---|---|
| Horizontal flip | p = 0.5 | Cars parked nose-in versus reverse-in |
| Rotation | ±6% of 2π (≈ ±21°) | Camera bracket knocked out of alignment |
| Zoom | ±12% | Slight differences in survey box size |
| Translation | ±8% both axes | Camera drift, imprecise survey |
| Contrast | ±25% | Overcast flatness versus harsh midday sun |
| Brightness | ±25% | Dawn, dusk, glare, shadow |

**Vertical flip is deliberately excluded.** Flipping a bay upside down would put
a car's roof where its shadow belongs — a configuration a fixed overhead camera
can never produce. Augmenting toward images that cannot occur wastes capacity
and teaches invariances that are not real.

Rendered examples: [`results/augmentation_examples.png`](../results/augmentation_examples.png)

---

## 2.4 The split — and the leakage trap

The brief asks for 70% train, 15% validation, 15% test. **How** those splits are
drawn matters more than the ratio.

### The trap

PKLot frames are five minutes apart. Two frames of the same lot on the same
afternoon are near-identical: the same cars, in the same bays, under the same
light. Splitting 6,000 crops at random puts a crop and its near-twin on opposite
sides of the train/test boundary. The model can then score ~99.9% on test
without having generalised at all — it has simply memorised, and the number is
meaningless.

### The fix: grouped splitting

**Whole `(lot, calendar date)` groups are assigned to a split.** If
2013-03-05 at UFPR05 is a test day, then *every* frame from that day is in test
and none is in train. The ratio is applied within each lot, so all three lots
appear in all three splits.

| Split | Day-groups | Crops per class | Total |
|---|---|---|---|
| Train | 69 | 2,100 | 4,200 |
| Validation | 15 | 450 | 900 |
| Test | 14 | 450 | 900 |
| **Total** | **98** | **3,000** | **6,000** |

Split by day-groups: **70.4% / 15.3% / 14.3%** — the requested 70/15/15 to within
a rounding error, while guaranteeing no calendar day is shared.

> **Why this matters for the reported accuracy.** Every number in
> [`results/metrics.json`](../results/metrics.json) is measured on photographs
> taken on days the model has never seen. It is a lower, harder, and honest
> figure. A random split would have produced a higher number that meant less.

### Class balance

Every split is **exactly 50/50** occupied/empty, filled by quota rather than by
taking whatever a scene happened to contain. Inverse-frequency class weights are
still computed and passed to `model.fit` — they come out at ≈1.0 here, but they
mean a rebuild with different quotas is handled correctly instead of silently
biasing the model.

---

## 2.5 Folder structure

Exactly the layout the brief asks for, and the layout
`image_dataset_from_directory` expects:

```
data/
├── raw/
│   ├── samples.json          PKLot annotations (267 MB, re-downloaded on demand)
│   ├── scene_summary.csv     one row per photograph — for exploratory analysis
│   └── images/               the 274 whole-lot photographs actually used
├── processed/
│   ├── manifest.json         provenance for every crop: lot, weather, date, bay id
│   ├── train/
│   │   ├── empty/            2,100 crops
│   │   └── occupied/         2,100 crops
│   ├── val/
│   │   ├── empty/              450 crops
│   │   └── occupied/           450 crops
│   └── test/
│       ├── empty/              450 crops
│       └── occupied/           450 crops
└── demo/
    ├── demo_index.json       bay geometry + ground truth for the dashboard
    └── *.jpg                 12 whole-lot scenes, all from test days
```

**`manifest.json` is what makes the slice analysis possible.** Every crop
records the lot, weather, date, time and bay id it came from, so
`scripts/03_evaluate.py` can report accuracy *by weather* and *by lot* rather
than only in aggregate — which is how you find out that a model is fine on
average and poor in the rain.

**Demo scenes are drawn only from test day-groups**, and are excluded from the
crop pool. Every prediction the dashboard shows is therefore on a photograph
from a day the model never trained on, spread deliberately across lots, weather
and occupancy bands so that low, moderate and high congestion can all be
demonstrated.

---

## 2.6 Reproducibility

- One seed (`RANDOM_SEED = 42` in `src/config.py`) drives day-group assignment,
  scene ordering, bay sampling and demo selection.
- `python scripts/01_build_dataset.py` rebuilds the entire dataset from scratch
  in **~13 seconds** once the annotations are cached (~2 minutes from cold,
  including the download).
- The build writes [`results/dataset_report.json`](../results/dataset_report.json)
  recording exactly what was produced.

---

**Next:** [Step 3 — Model Development](03_model_development.md)
