# Working on ParkVision AI

## Layout

| Path | Role |
|---|---|
| `src/` | Library code. Imported by the scripts, the notebook and the app — never duplicated between them. |
| `scripts/` | Numbered pipeline stages, each runnable on its own. |
| `tests/` | Fast checks over the parts whose behaviour must be exact. |
| `app.py` | The Streamlit dashboard. Presentation only; all logic lives in `src/`. |

## Order of operations

```bash
python scripts/01_build_dataset.py      # PKLot -> balanced, leakage-free crops
python scripts/02_train.py              # two-phase transfer learning
python scripts/03_evaluate.py           # metrics, confusion matrix, slice analysis
python scripts/04_robustness_test.py    # scene-level + stress testing
python scripts/06_make_readme.py        # README, generated from the results
python scripts/make_notebook.py --execute
python scripts/05_make_submission.py    # submission PDF
python scripts/07_verify_submission.py  # pre-submission checklist
```

`03` must run before `04` and before the app: it writes the operating threshold
that both read from `results/metrics.json`.

## Conventions

**Every constant lives in `src/config.py`.** If a number appears in two files,
it belongs in config.

**`src/insights.py` must not import TensorFlow or OpenCV.** The decision rules
are pure functions over plain numbers so they can be tested exactly. Keep it
that way.

**Crop geometry has exactly one implementation.** `src/slots.py::crop_slot` is
used by the dataset builder *and* by inference. A second implementation would
reintroduce train/serve skew; a change here invalidates the trained model.

**Preprocessing lives inside the model graph.** Augmentation and rescaling are
layers, not data-pipeline steps, so the exported `.keras` file is
self-contained and cannot be fed wrongly-scaled input.

**The README is generated.** Edit `scripts/06_make_readme.py`, not `README.md`
— the README quotes measured numbers and hand-editing lets prose drift away
from the metrics beside it.

## Before committing

```bash
python tests/test_parkvision.py         # 42 tests, ~40 s
python scripts/07_verify_submission.py  # checks every deliverable is present
```

## Retraining on a constrained machine

Training is CPU-bound and memory-sensitive. If a machine has under ~2 GB free,
split the run so each phase fits in one sitting:

```bash
python scripts/02_train.py --phase head
python scripts/02_train.py --phase finetune --resume
```

Do not re-enable caching of the training split in `src/dataset.py`. It looks
like an optimisation and is one on a machine with spare memory; on a
constrained one it caused an epoch to take over four hours instead of ~150
seconds, with the CPU near-idle throughout.
