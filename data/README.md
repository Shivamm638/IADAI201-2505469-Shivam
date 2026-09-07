# Data

## Provenance

Every image in this project comes from **PKLot**, used for academic study.

> Almeida, P. R. L., Oliveira, L. S., Britto, A. S., Silva, E. J., & Koerich, A. L.
> (2015). *PKLot — A robust dataset for parking lot classification.*
> **Expert Systems with Applications, 42**(11), 4937–4949.
> https://doi.org/10.1016/j.eswa.2015.02.009

- Original release: <https://web.inf.ufpr.br/vri/databases/parking-lot-database/>
- Mirror used here: <https://huggingface.co/datasets/Voxel51/PKLot>

PKLot is published by the Federal University of Paraná for research and
academic use. It is redistributed here only as the derived 224×224 bay crops
needed to reproduce this assignment, with attribution. Anyone building on this
work should cite the paper above and obtain the dataset from the original
source.

## What is in this folder

| Path | Committed | Contents |
|---|---|---|
| `processed/train\|val\|test/{empty,occupied}/` | yes | 6,000 perspective-corrected 224×224 bay crops — the dataset the model was trained and evaluated on |
| `processed/manifest.json` | yes | Provenance of every crop: lot, weather, date, time, source photograph, bay id |
| `demo/` | yes | 12 whole-lot photographs plus `demo_index.json` (bay geometry + ground truth) for the dashboard |
| `raw/samples.json` | **no** | The 267 MB PKLot annotation document |
| `raw/images/` | **no** | The 274 full-resolution source photographs |
| `raw/scene_summary.csv` | yes | One row per PKLot photograph — lot, weather, timestamp, bay counts |

The two excluded paths are re-downloaded on demand:

```bash
python scripts/01_build_dataset.py
```

That rebuilds everything above from scratch in about two minutes from cold, or
thirteen seconds if the annotations are already cached. The build is seeded, so
it reproduces the same split and the same figures.

## How the crops were made

1. Each bay's annotated quadrilateral is expanded 8% about its centre.
2. It is perspective-warped to a 224×224 square, which deskews the bay and
   resizes it in a single resample.
3. Crops with a pixel standard deviation below `MIN_CROP_STD` are rejected —
   a bay hidden behind a tree yields a featureless patch that is identical
   whatever its label says.

## The split

Whole `(lot, calendar date)` groups are assigned to a single split. PKLot
frames are five minutes apart, so a random per-crop split would place a crop
and its near-twin on opposite sides of the train/test boundary and report
memorisation as generalisation.

| Split | Day-groups | Empty | Occupied |
|---|---|---|---|
| train | 69 | 2,100 | 2,100 |
| val | 15 | 450 | 450 |
| test | 14 | 450 | 450 |

Demo scenes are drawn only from test day-groups and are excluded from the crop
pool, so every prediction the dashboard shows is on an unseen day.
