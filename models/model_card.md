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
| Schedule | 12 epochs head-only (lr 0.001), then 10 fine-tuning (lr 1e-05) |
| Batch size | 32 |
| Loss | Binary cross-entropy, label smoothing 0.02 |
| Seed | 42 |
| Training data | 6,000 PKLot bay crops, 3,000 per class |

## Measured performance

On 900 bays from **calendar days that appear in no other split**:

| Metric | Value |
|---|---|
| Accuracy | 99.89% |
| Precision (occupied) | 99.78% |
| Recall (occupied) | 100.00% |
| ROC AUC | 1.0000 |
| Operating threshold | 0.50 (chosen on validation, applied unchanged to test) |

**By weather:** cloudy 99.55% · rainy 100.00% · sunny 100.00%

**By lot:** PUCPR 100.00% · UFPR04 100.00% · UFPR05 99.71%

## Intended use

Educational and research use, on elevated fixed-camera views of surface parking lots similar to PKLot's three. It is a coursework artefact, not a product.

## Out of scope

- **Enforcement or billing.** The error rate is far too high to charge or fine anyone on.
- **Night-time operation.** The weakest measured condition is *Sensor noise (σ=18)* at 96.33%. PKLot contains no true night imagery, so that figure is a simulation, not a measurement.
- **Unsurveyed lots.** Without bay geometry the app falls back to an approximate grid, which it labels as indicative.
- **Any safety-critical decision.**

## Known limitations

- A car straddling two bays is counted in both: each bay is classified independently.
- Trained on three lots in one country; different layouts, vehicle mixes or camera heights are untested.
- A bay fully hidden behind a tree yields a featureless crop; such crops are rejected at build time, and at inference they score near the threshold and are flagged low-confidence rather than asserted.

## Training data licence

PKLot (Almeida et al., 2015) is published by the Federal University of Paraná for research and academic use. See [`data/README.md`](../data/README.md).