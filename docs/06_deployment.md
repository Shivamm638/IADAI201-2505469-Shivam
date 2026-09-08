# Step 7 — Deployment on Streamlit Community Cloud

---

## 7.1 What gets deployed

Streamlit Community Cloud clones the GitHub repository and runs `app.py`.
Everything the app needs at runtime is therefore committed:

| Path | Size | Why it must be in the repository |
|---|---|---|
| `app.py` | ~19 KB | The application itself |
| `src/` | ~60 KB | Inference, insight logic, overlays, layouts |
| `models/parkvision_mobilenetv2.keras` | 24 MB | The trained model — there is no training step at deploy time |
| `models/layouts.json` | ~200 KB | Surveyed bay geometry per camera |
| `data/demo/` | ~2 MB | Twelve held-out scenes plus their ground truth |
| `results/metrics.json` | ~3 KB | Supplies the tuned decision threshold and the accuracy panel |
| `requirements.txt` | — | Python dependencies |
| `packages.txt` | — | System libraries for OpenCV |

Deliberately **not** committed (see `.gitignore`):

- `data/raw/samples.json` — the 267 MB PKLot annotation document
- `data/raw/images/` — full-resolution source photographs

Both are re-downloaded on demand by `scripts/01_build_dataset.py`, so the
pipeline stays fully reproducible without bloating the clone.

---

## 7.2 Dependency choices

```
streamlit==1.62.0
tensorflow-cpu==2.17.1
keras==3.15.1
numpy==1.26.4
opencv-python-headless==4.11.0.86
pillow==11.3.0
pandas==2.2.3
```

Every version is pinned to exactly the environment the app was built and tested
against. Four of those pins are deliberate rather than incidental:

**`tensorflow-cpu`, not `tensorflow`.** The deployed app only runs inference.
The full package pulls in CUDA libraries that will never be used and can push
the image past Community Cloud's build limits; the CPU wheel is roughly a third
of the size and starts faster.

**`opencv-python-headless`, not `opencv-python`.** The headless build omits the
GUI bindings, which need an X server that a container does not have. Using the
regular build is the single most common cause of `ImportError: libGL.so.1` on
Streamlit Cloud.

**`packages.txt` with `libgl1` and `libglib2.0-0`.** Even the headless OpenCV
build links against these system libraries, and Debian slim images do not ship
them.

**`streamlit` at 1.62, not 1.40.** `st.image(..., use_container_width=...)` only
exists from 1.41; on 1.40 the same call raises `TypeError` and the app dies on
its first render. Pinning below the feature you use is worse than not pinning at
all, because it fails only in deployment.

**`keras` pinned alongside TensorFlow.** The saved `.keras` file must be loaded
by a compatible Keras version, and TensorFlow's own requirement (`keras>=3.2`)
is loose enough to resolve to something that cannot read it.

An unpinned `streamlit` or `tensorflow` will silently upgrade on a later rebuild
and can break an app that has not been touched in months.

---

## 7.3 Deployment steps

### 1 — Push to GitHub

This folder is **already a git repository with the project committed**, so
there is nothing to initialise. Create an empty repository on GitHub — no
README, no .gitignore, no licence, or the first push will be rejected as
diverged — then attach it and push:

```bash
cd "SA 1 Output"
git branch -M main
git remote add origin https://github.com/<username>/IADAI201-<student-id>-<name>.git
git push -u origin main
```

Repository name follows the format the brief specifies:
`IADAI201(Student_id)-studentname`.

The push moves roughly 112 MB — the 6,000 dataset crops and the 24 MB model —
so expect it to take a few minutes. If it is rejected for a file over 100 MB,
something outside `.gitignore` has crept in; `git ls-files -s | sort -k4` will
show the largest tracked paths.

### 2 — Grant assessor access

Settings → Collaborators → add **`ai.assignments@wacpinternational.org`**
(or make the repository public).

### 3 — Deploy

1. Go to <https://streamlit.io/cloud> and sign in with GitHub.
2. **New app** → select the repository.
3. Branch `main`, main file path `app.py`.
4. **Deploy**.

The first build takes roughly 3–6 minutes, mostly installing TensorFlow.

### 4 — Verify and share

Once the build finishes, check that:

- the app loads without an error banner;
- the sidebar shows the model status with test accuracy;
- a demo scene produces an annotated image with green and red bays;
- the metric cards and congestion banner populate;
- the CSV and PNG downloads work.

Then copy the public URL (`https://<app-name>.streamlit.app`) into the README
and the submission document.

---

## 7.4 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: libGL.so.1` | `opencv-python` instead of the headless build | Use `opencv-python-headless` **and** keep `packages.txt` |
| Build exceeds resource limits | Full `tensorflow` package | Use `tensorflow-cpu` |
| `FileNotFoundError: ...keras` | Model not committed, or blocked by `.gitignore` | Confirm `git ls-files models/` lists the `.keras` file |
| App boots then sleeps | Community Cloud idles inactive apps | Normal; the first visitor wakes it in ~30 s |
| Model file rejected by GitHub | File over 100 MB | Not applicable — the model is 24 MB |
| Slow first prediction | TensorFlow graph warm-up | Only affects the first request per session |

---

## 7.5 Reproducing the whole project from a clone

```bash
git clone https://github.com/<username>/IADAI201-<student-id>-<name>.git
cd IADAI201-<student-id>-<name>
pip install -r requirements-dev.txt

# Optional — the trained model and dataset are already committed.
python scripts/01_build_dataset.py      # rebuild from PKLot   (~2 min cold)
python scripts/02_train.py              # retrain              (~70 min CPU)
python scripts/03_evaluate.py           # metrics and figures  (~1 min)
python scripts/04_robustness_test.py    # testing report       (~4 min)

streamlit run app.py
```

On a CPU-only machine the training step can be split into bounded sessions:

```bash
python scripts/02_train.py --phase head
python scripts/02_train.py --phase finetune --resume
```

Every script is seeded (`RANDOM_SEED = 42`), so a rebuild reproduces the same
dataset split and the same reported figures.

---

**Back to:** [README](../README.md)
