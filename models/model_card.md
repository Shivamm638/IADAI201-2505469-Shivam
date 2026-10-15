# Model card — ParkVision AI occupancy classifier

**File:** `parkvision_mobilenetv2.keras` (24.4 MB)

## What it does

Takes one **224×224 RGB crop of a single parking bay** and returns `P(occupied)` as a sigmoid score. It classifies a bay that it is *given*; it does not locate bays. Bay geometry is a per-camera survey — see [`docs/01_problem_definition.md`](../docs/01_problem_definition.md).

Input is raw RGB in `[0, 255]`. Rescaling and augmentation are layers **inside** the graph, so the file is self-contained and cannot be fed wrongly-scaled input.

## Architecture and training

| | |
|---|---|
| Backbone | MobileNetV2, ImageNet weights |
| Parameters | 2,259,265 |
| Augmentation | Flip, rotation, zoom, shift, contrast, brightness, and random-strength sensor noise — all inside the model graph, inactive at inference |
| Schedule | 12 epochs head-only (lr 0.001), then 10 fine-tuning (lr 1e-05) |
| Batch size | 32 |
| Loss | Binary cross-entropy, label smoothing 0.02 |
| Seed | 42 |
| Training data | 6,000 PKLot bay crops, 3,000 per class |

## Measured performance

On 900 bays from **calendar days that appear in no other split**:

| Metric | Value |
|---|---|
| Accuracy | 100.00% (95% CI 99.59–100.00%) |
| Precision (occupied) | 100.00% |
| Recall (occupied) | 100.00% |
| ROC AUC | 1.0000 |
| Operating threshold | 0.55 (chosen on validation, applied unchanged to test) |

### Measured again, at 115x the size

The table above is a balanced 900-bay sample. Scored against **every** bay from the same held-out days — 103,465 of them across 1,773 photographs, in their natural distribution:

| Metric | Value |
|---|---|
| Accuracy | **99.77%** (95% CI 99.735–99.794%) |
| Errors | 242 (160 empty called occupied, 82 the reverse) |
| Congestion level correct | 1,769 / 1,773 scenes |
| Mean free-bay error | 0.133 bays (worst scene: 34) |

**Treat this as a floor.** A seeded random sample of 24 disagreements found roughly a third to be cases where PKLot's label contradicts the image — bays labelled empty that visibly hold a car. The true accuracy is higher; no corrected figure is quoted, because re-labelling ground truth to improve a score stops it being a measurement.

Weakest slices, from that sweep: rainy 99.40% · sunny 99.79% · cloudy 99.98% · UFPR05 99.54% · PUCPR 99.82% · UFPR04 99.93%

**By weather:** cloudy 100.00% · rainy 100.00% · sunny 100.00%

**By lot:** PUCPR 100.00% · UFPR04 100.00% · UFPR05 100.00%

## Intended use

Educational and research use, on elevated fixed-camera views of surface parking lots similar to PKLot's three. It is a coursework artefact, not a product.

## Out of scope

- **Enforcement or billing.** A perfect score on a 900-bay test does not mean a zero error rate: the same measurement only bounds the true accuracy at 99.59% with 95% confidence, which still permits several wrong calls per thousand bays. Nobody should be charged or fined on that.
- **Night-time operation.** PKLot contains no true night imagery. The dusk and night rows in the robustness report are simulated by dimming daytime frames, which is not the same as a real low-light sensor.
- **Heavy sensor noise.** The weakest measured condition is *Sensor noise (σ=30)* at 98.78%; it is set outside the range the model trains on deliberately, as a generalisation check.
- **UFPR05 in direct sun.** The one weakness the large sweep isolated: dense tree shadow and pedestrians both read as vehicles there. It is the lowest-scoring lot in the full held-out evaluation.
- **Unsurveyed lots.** Without bay geometry the app falls back to an approximate grid, which it labels as indicative.
- **Any safety-critical decision.**

## Known limitations

- A car straddling two bays is counted in both: each bay is classified independently.
- Trained on three lots in one country; different layouts, vehicle mixes or camera heights are untested.
- A bay fully hidden behind a tree yields a featureless crop; such crops are rejected at build time, and at inference they score near the threshold and are flagged low-confidence rather than asserted.

## Training data licence

PKLot (Almeida et al., 2015) is published by the Federal University of Paraná for research and academic use. See [`data/README.md`](../data/README.md).