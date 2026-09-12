# Step 3 — Model Development

*Produced by `python scripts/02_train.py` and `python scripts/03_evaluate.py`.*

---

## 3.1 Choosing the architecture

The brief names MobileNet, EfficientNet and YOLO as candidates.
[Step 1](01_problem_definition.md#14-choosing-the-approach) already settled that
this is a **classification** problem over surveyed bays, not a detection
problem, which leaves the choice of backbone.

| Backbone | Parameters | Compute per image | Fit for this task |
|---|---|---|---|
| **MobileNetV2** ✅ | 2.26 M | ~0.3 GFLOPs | Depthwise-separable convolutions are designed for exactly this: small, low-resolution inputs on constrained hardware |
| EfficientNetB0 | 5.3 M | ~0.4 GFLOPs | Stronger on 1000-class ImageNet; that capacity is wasted on a 2-class problem, and its depthwise and squeeze-excite layers run slower on CPU than the FLOP count suggests |
| ResNet50 | 25.6 M | ~4.1 GFLOPs | 11× the parameters and ~13× the compute for a binary decision |
| Training from scratch | — | — | 4,200 training crops is far too few to learn edges, texture and shape from nothing |

Only MobileNetV2 was benchmarked end to end: **~10 ms per bay** on CPU, measured
over the 695 bays of the twelve held-out scenes and including the perspective
crop. The other rows compare published compute costs; they were not trained or
timed here.

**MobileNetV2 with ImageNet weights** was chosen because:

1. **The task is easier than ImageNet, not harder.** Distinguishing "car
   present" from "empty tarmac" needs colour, texture and shape — features the
   early and middle ImageNet layers already encode well. The capacity of a
   larger backbone would mostly go unused.
2. **A 100-bay lot must be analysed in one user-visible moment.** At ~10 ms per
   bay end to end, PUCPR's 100 bays take about a second on CPU.
3. **Free-tier deployment is a hard constraint.** The saved model is **24 MB**
   (9 MB of weights plus the optimizer state that `--resume` needs)
   and runs on Streamlit Community Cloud's CPU container with no GPU.
4. **Small models overfit less on small datasets.** With 4,200 training crops,
   fewer parameters is a feature, not a compromise.

### Architecture

```
slot_image                 (224, 224, 3)   float32 in [0, 255]
   │
   ├─ augmentation          flip · rotate · zoom · shift · contrast · brightness
   │                        (training only — inactive at inference)
   ├─ Rescaling             [0, 255] → [-1, 1]      MobileNetV2's expected range
   │
   ├─ MobileNetV2           ImageNet weights, include_top=False
   │                        → (7, 7, 1280) feature map
   │
   ├─ GlobalAveragePooling2D → (1280,)
   ├─ Dropout(0.30)
   └─ Dense(1, sigmoid)     → P(occupied)

Total parameters      2,259,265
Trainable, phase 1        1,281   (the head alone)
Trainable, phase 2    1,840,897   (head + top of the backbone)
```

**Augmentation and rescaling live inside the model**, not in the data script.
This means the exported `.keras` file is completely self-contained: the
Streamlit app hands it raw pixels and cannot get the preprocessing wrong. It
removes an entire class of silent train/serve skew bugs.

**GlobalAveragePooling instead of Flatten.** Flattening the 7×7×1280 feature map
would create a 62,720-wide vector and a Dense layer with ~62,000 parameters —
fifty times the size of the rest of the head, and a direct route to overfitting
4,200 images. Average pooling gives a 1,280-wide vector and 1,281 parameters.

---

## 3.2 Two-phase transfer learning

Training runs in two phases, for a specific reason.

### Phase 1 — train the head, backbone frozen (12 epochs, lr = 1e-3)

The new `Dense(1)` layer starts from random weights. Its first gradients are
large and essentially meaningless. If the backbone were trainable at that
moment, those gradients would flow straight into carefully pre-trained ImageNet
filters and destroy them — the well-documented failure mode of fine-tuning too
early. Freezing the backbone lets the head reach a sensible solution first.

### Phase 2 — fine-tune the top blocks (10 epochs, lr = 1e-5)

Once the head is stable, MobileNetV2 is unfrozen **from layer 100 upward**
(roughly the last third) at a learning rate **100× smaller**. The early layers
keep their general edge and texture detectors; the later, more semantic layers
adapt to what parking bays actually look like from an elevated camera.

**BatchNormalization layers stay frozen throughout phase 2.** Their running mean
and variance were estimated over ImageNet's millions of images. Re-estimating
them from batches of 32 parking crops is a well-known way to make fine-tuning
unstable, and produces a model whose training and inference behaviour diverge.

**Total: 22 epochs** — inside the 15–30 band the brief specifies.

---

## 3.3 Training configuration

| Hyperparameter | Value | Reasoning |
|---|---|---|
| Batch size | 32 | Standard for this scale; large enough for stable gradients, small enough for a CPU's memory |
| Optimiser | Adam | Robust default; no learning-rate schedule tuning required beyond the two phases |
| LR (phase 1) | 1e-3 | Standard for a freshly initialised head |
| LR (phase 2) | 1e-5 | 100× smaller so fine-tuning nudges pre-trained weights rather than overwriting them |
| Loss | Binary cross-entropy, label smoothing 0.02 | Smoothing discourages the over-confident 0.0/1.0 outputs that make the confidence display meaningless |
| Dropout | 0.30 | Regularisation before the final layer |
| Class weights | inverse frequency (≈1.0) | Splits are balanced by construction; the weights make an unbalanced rebuild safe |
| Early stopping | patience 6 on `val_accuracy` | Stops fine-tuning once validation stops improving |
| ReduceLROnPlateau | ×0.5, patience 3 | Lets the model settle into a minimum instead of bouncing around it |
| Seed | 42 | Set for Python, NumPy and TensorFlow |

### Two correctness details worth calling out

**1. The checkpoint cannot be overwritten by a worse model.** `ModelCheckpoint`
starts from `-inf` when it is constructed. A fresh instance for phase 2 would
therefore save the first fine-tuning epoch unconditionally — even if it were
worse than the best phase-1 model — because it has no memory of phase 1. Phase
2's checkpoint is explicitly seeded with phase 1's best score
(`initial_value_threshold`), so the file on disk always holds the best model
across *both* phases.

**2. The training pipeline genuinely reshuffles, without holding the split in
RAM.** The obvious implementation — `image_dataset_from_directory(...).cache()`
— caches an *already batched* dataset, which freezes the first epoch's batches
and replays them unchanged forever, silently cancelling the reshuffle.

The fix is not simply to cache earlier. Shuffling is done at the **file-path**
level, before any pixels are decoded, and tf.data reshuffles that list every
epoch; a small 512-image buffer adds local mixing on top. The training split is
then *streamed* rather than cached.

That last choice was forced by measurement. Holding the 4,200 decoded crops
resident costs ~630 MB as uint8, and on a machine with roughly a gigabyte free
that pushed the process into paging: **one epoch went from ~150 seconds to over
four hours, with the CPU sitting near-idle the whole time.** Re-decoding the
JPEGs each epoch costs about four seconds and no resident memory. Validation and
test *are* cached — they are a quarter of the size, are read identically every
epoch, and must not vary between runs.

The general lesson: caching is not free, and on a constrained machine the
"optimisation" can be two orders of magnitude slower than the thing it replaced.

---

## 3.4 Evaluation methodology

### The decision threshold is chosen on validation, never on test

A sigmoid outputs a probability; turning it into a decision needs a threshold.
The default 0.5 is arbitrary. `scripts/03_evaluate.py` sweeps 91 thresholds from
0.05 to 0.95, picks the one that maximises accuracy **on the validation split**,
then applies it **unchanged** to test.

Tuning the threshold on test and then reporting test accuracy would be fitting
to the test set and reporting the fit as a result. Keeping the two separate is
what makes the reported number trustworthy. The sweep is plotted in
[`results/threshold_sweep.png`](../results/threshold_sweep.png).

### Metrics reported

| Metric | Why it is included |
|---|---|
| Accuracy | The brief's headline metric |
| Confusion matrix | The brief's requirement; also shows *which direction* errors go |
| Precision / recall / F1, per class | An occupied-bay error and an empty-bay error have different real-world costs |
| ROC AUC and average precision | Threshold-independent measures of how well the two classes separate |
| Balanced accuracy, Matthews correlation | Robust to any class imbalance introduced by a rebuild |
| **Accuracy by weather** | A model that is 98% overall but 85% in rain is not deployable in a city with rain |
| **Accuracy by parking lot** | Shows whether the model generalises across camera angles or has memorised one lot |

The last two are the ones that matter most operationally, and they are only
possible because `data/processed/manifest.json` records the provenance of every
crop.

### The two error types are not equivalent

| Error | What the driver experiences | Cost |
|---|---|---|
| **Occupied bay called empty** (false negative) | Driver is sent to a bay that has a car in it | High — the exact frustration the system exists to remove |
| **Empty bay called occupied** (false positive) | A usable bay is not offered | Lower — the driver simply parks elsewhere |

Recall on the occupied class is therefore reported alongside accuracy, and the
threshold slider in the dashboard lets an operator shift the balance
deliberately rather than accept whichever trade-off the default 0.5 happens to
give.

---

## 3.5 How the model was improved

The brief asks for iterative improvement — adding data, fixing labels, tuning
parameters. Each change below was made in response to a measured problem:

| # | Problem observed | Change made | Effect |
|---|---|---|---|
| 1 | An axis-aligned crop of an angled bay included half of each neighbour, so a car next door made an empty bay look occupied | Perspective-warp the annotated quadrilateral instead of its bounding box | Cleanly separated bays; removed a systematic label-noise source |
| 2 | Cars overhanging the painted line were being cut off at the bay edge | Expand the quadrilateral by 8% before warping | Full vehicle visible in the crop |
| 3 | Quotas filled from the earliest scenes first, over-representing one lot and one time of day | Round-robin across (lot × weather × split) buckets, with a per-scene cap | All three lots and all three weather conditions represented in every split |
| 4 | Validation and test quotas fell short (246/450 and 187/450) because a tight per-scene cap starves splits with fewer days | Multi-round cap, relaxed only for classes still short of quota | Exactly 450/450 per class in both splits, spread preserved |
| 5 | A random split would let a frame's five-minute-later near-twin sit in the test set | Group whole (lot, date) pairs into a single split | Honest generalisation measurement rather than an inflated one |
| 6 | Phase 2's checkpoint could overwrite a better phase-1 model | Seed phase 2's checkpoint with phase 1's best score | The shipped model is the best across both phases |
| 7 | Caching after batching froze the epoch order, cancelling the reshuffle | Shuffle file paths before decoding; small in-memory buffer on top | Genuine per-epoch reshuffling |
| 7b | Caching the decoded training split (630 MB) pushed a memory-constrained machine into paging — one epoch went from ~150 s to **over four hours** at near-zero CPU | Stream the training split; cache only the small validation and test splits; carry pixels as uint8 throughout | Epochs back to ~150 s; peak memory down roughly 1 GB |
| 8 | Over-confident 0.00/1.00 outputs made the confidence display meaningless | Label smoothing 0.02 | Calibrated scores, so "low confidence" flags real uncertainty |
| 9 | **UFPR04 scored 79.8% while the other two lots scored 99%+** | Fixed the corner-ordering bug described below, and added a flat-crop guard | See §3.5.1 |
| 10 | **Under sigma=18 sensor noise the model called 29 empty bays occupied**, reading grain on bare tarmac as car texture | Added a random-strength sensor-noise layer to the in-graph augmentation | See §3.5.2 |

### 3.5.1 The bug the slice analysis caught

This one is worth describing in full, because average accuracy would never have
surfaced it and because it is the clearest example of why the per-lot breakdown
was built in the first place.

**The symptom.** The first trained model scored 94.11% overall, with a ROC AUC
of 0.9968 — near-perfect *ranking* but a much weaker *accuracy*. The per-lot
breakdown showed why the average was hiding something:

| Lot | ROC AUC | Accuracy |
|---|---|---|
| PUCPR | 1.0000 | 99.4% |
| UFPR05 | 0.9998 | 98.6% |
| **UFPR04** | **0.9615** | **79.8%** |

46 of the 53 test errors came from one lot.

**The investigation.** Plotting UFPR04's most confident errors showed something
no amount of hyperparameter tuning would have fixed: the crops were **blank
grey rectangles**. Not hard examples — no image content at all. A scan of the
whole dataset found **208 of 6,000 crops (3.5%) were featureless**, 202 of them
from UFPR04, and split across *both* labels: 125 "empty" and 83 "occupied".
Identical blank patches carrying contradictory labels are the purest possible
form of label noise.

**The cause.** `_order_quad()` used the textbook shortcut for putting four
corners in order — smallest `x + y` is top-left, smallest `y − x` is top-right.
That heuristic silently fails for a quadrilateral rotated near 45°, where the
*same* corner wins two of the four tests. The source quad then contains only
three distinct points, `cv2.getPerspectiveTransform` returns a singular matrix,
and the warp collapses the bay to a flat patch.

UFPR04's camera views its bays at roughly 45°, so it was hit far harder than
the other two lots. Nothing raised an error: OpenCV warps a degenerate quad
happily, and the result is a valid-looking 224×224 JPEG.

```
bay id  raw corners (pixels)                        ordered corners                              collapsed?
     1  (641,570) (726,671) (650,708) (577,602)     (577,602) (641,570) (726,671) (650,708)      no
     2  (698,505) (782,586) (715,629) (629,541)     (629,541) (782,586) (782,586) (715,629)      YES
     3  (719,427) (799,499) (754,540) (666,461)     (666,461) (799,499) (799,499) (666,461)      YES
```

**The fix.** Order corners by their angle about the centroid, which is correct
for any convex quadrilateral at any rotation, then roll the ordering to start
at the corner nearest the top-left bound so the same bay always deskews to the
same orientation at training and inference time.

**Defence in depth.** A crop whose pixel standard deviation falls below
`MIN_CROP_STD` is now rejected outright at build time. A bay completely hidden
behind a tree produces a genuinely featureless patch through no fault of the
geometry, and admitting it into training is noise whatever the label says. After
the geometry fix, this guard rejects only 8 bays across the whole build — but it
is what turns "we fixed the bug we found" into "this class of bug cannot
silently return".

**Regression tests.** `tests/test_parkvision.py` now sweeps a rectangle through
all 360° and asserts that (a) the ordering never reuses a corner, and (b) a
textured bay never warps to a flat patch. Both fail against the old code.

---


### 3.5.2 The weakness the stress test caught

The same principle as §3.5.1, applied to the robustness suite rather than the
slice breakdown: the average was fine, one row was not.

Across twelve simulated field conditions the model lost at most four bays in
900 — except one. Under **sigma=18 Gaussian noise it lost 33**, and the errors
were lopsided: **29 of them were empty bays called occupied**. Inspecting the
scores showed why a threshold tweak could not rescue it. The mistaken bays were
not sitting near the boundary; many scored P(occupied) between 0.8 and 0.97.
The model was confidently wrong. Grain on bare tarmac has the high-frequency
texture of car bodywork, and nothing in training had ever told it otherwise:
the augmentation covered flips, rotation, zoom, shift, contrast and brightness,
but **no noise at all**.

Two candidate fixes were measured before either was adopted.

**Test-time augmentation, rejected.** Averaging the prediction over horizontal
and vertical flips is the cheap, no-retraining option. Measured, it does not
work here: noisy errors moved 32 -> 30 with two views and 32 -> **35** with
four, for two to four times the inference cost. The reason is obvious in
hindsight — flipping a noisy crop flips the noise with it, so the views are not
independent and nothing averages out.

**Noise augmentation, adopted.** `RandomSensorNoise` (in `src/model.py`) draws a
standard deviation per image from `U(0, 25)` on the 0-255 scale and applies it
to half of each batch. Keras's built-in `GaussianNoise` was not suitable: it
applies one fixed strength to every image, so the model would never see a clean
one. The layer sits last in the augmentation block, clips back into [0, 255],
and is the identity at inference.

The range is deliberately *not* tuned to the test. sigma=18 sits inside
`U(0, 25)` but so does everything from a clean daylight sensor upward, and the
stress suite gained a **sigma=30** row that lies outside the training range
specifically to check that the robustness generalises instead of being
memorised for one value.

#### Result

Retrained with one variable changed — same data, same split, same schedule,
same seed:

| Condition | Before | After | Errors / 900 |
|---|---|---|---|
| Clean test crops | 99.89% | **100.00%** | 1 -> 0 |
| Sensor noise sigma=10 | 95.78% | **99.78%** | 38 -> 2 |
| Sensor noise sigma=18 | 96.33% | **99.33%** | 33 -> 6 |
| Sensor noise sigma=30 *(outside training range)* | 96.78% | **98.78%** | 29 -> 11 |
| Harsh light (+50% contrast) | 99.67% | **100.00%** | 3 -> 0 |
| Low bandwidth (JPEG q=25) | 99.67% | **99.78%** | 3 -> 2 |
| Glare (+45% brightness) | 99.56% | 99.56% | 4 -> 4 |
| **All 14 conditions** | | | **117 -> 26** |

**No condition got worse.** Total errors across the whole suite fell by
78%.

One detail is worth more than the headline. The old model made *more* mistakes
at mild noise (38 at sigma=10) than at heavy noise (29 at sigma=30) — its
response was not even monotonic, which is what an untrained-for input
distribution looks like. The new model degrades in order: 2, then 6, then 11.
That ordering, more than any single number, is the evidence that it learned
something general about grain rather than a trick for one setting.

#### What did *not* change, and why that is honest

The clean test went from 1 error to 0. **That is not a real improvement.** One
bay in 900 is inside the noise of the measurement: the exact binomial interval
for 1/900 is [99.38%, 100.00%] and for 0/900 it is [99.59%, 100.00%], which
overlap almost entirely. The clean split was already at the ceiling and could
not have shown a difference either way. The stress suite could, which is why
the argument for this change rests there.

The scene-level test is also unchanged: 11 of 12 photographs perfect, 12 of 12
congestion levels correct, mean free-bay error 0.08. Both models miss the same
single bay, and it is worth looking at — PKLot labels bay 77 of the PUCPR
2012-10-09 scene **empty**, while a car is caught halfway into it. The model
calls it occupied at P=0.94. For a driver being told where to park, the model's
answer is the more useful one. It is counted as an error throughout this report
regardless, because silently re-labelling ground truth to flatter a model is
exactly the habit these checks exist to prevent.

## 3.6 Results

Headline numbers, the confusion matrix, per-weather and per-lot breakdowns are
in the [main README](../README.md#results) and, in machine-readable form, in
[`results/metrics.json`](../results/metrics.json).

Generated artefacts:

| File | Contents |
|---|---|
| [`results/confusion_matrix.png`](../results/confusion_matrix.png) | Counts and row-normalised percentages |
| [`results/training_curves.png`](../results/training_curves.png) | Accuracy, loss and AUC across both phases |
| [`results/roc_pr_curves.png`](../results/roc_pr_curves.png) | ROC and precision-recall curves |
| [`results/threshold_sweep.png`](../results/threshold_sweep.png) | How the operating threshold was chosen |
| [`results/accuracy_by_slice.png`](../results/accuracy_by_slice.png) | Accuracy per weather condition and per lot |
| [`results/misclassified_examples.png`](../results/misclassified_examples.png) | The most confident errors, for discussion |
| [`results/classification_report.txt`](../results/classification_report.txt) | Per-class precision, recall and F1 |

---

**Next:** [Step 4 — Integration and Parking Insight Logic](04_insight_logic.md)
