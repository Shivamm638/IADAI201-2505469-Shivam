# Step 5 — Web App Development using Streamlit

*Implemented in [`app.py`](../app.py). Run with `streamlit run app.py`.*

---

## 5.1 Design goal

The brief's requirement is precise: the interface must be usable *"for
non-technical users"*. That single constraint drove every decision below.

The test applied throughout: **could a driver waiting at a barrier get the
answer in three seconds, without knowing what a sigmoid is?**

---

## 5.2 Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│  🅿️ ParkVision AI                                    (gradient hero)  │
│  Intelligent Urban Parking Analytics & Space Optimisation Platform   │
├────────────┬─────────────────────────────────────────────────────────┤
│ SIDEBAR    │  1 · Choose a parking-lot image                         │
│            │     [ 📁 Demo scenes ]  [ ⬆️ Upload your own ]           │
│ Model      │                                                         │
│  status    │  2 · Slot-level detection                               │
│  accuracy  │     ┌───────────────┬───────────────┐                   │
│            │     │   original    │   annotated   │  ← side by side   │
│ Detection  │     └───────────────┴───────────────┘                   │
│  threshold │     [ ⬇️ Download annotated image ]                      │
│            │                                                         │
│ Overlay    │  3 · Real-time availability                             │
│  bay no.   │     ┌──────┬──────────┬───────────┬────────────┐        │
│  confidence│     │Total │ Occupied │ Available │ Congestion │        │
│  amber flag│     │ 100  │    77    │    23     │    High    │        │
│  legend    │     └──────┴──────────┴───────────┴────────────┘        │
│            │     ╔═══════════════════════════════════════════╗       │
│ Colour key │     ║ 🔴 High congestion — Heavy congestion      ║       │
│ Bands      │     ║ Recommendation: try another area          ║       │
│            │     ╚═══════════════════════════════════════════╝       │
│            │                                                         │
│            │  4 · Parking utilisation analytics                      │
│            │     [Where to park][Confidence][Bay-by-bay][Accuracy]   │
└────────────┴─────────────────────────────────────────────────────────┘
```

The four numbered sections mirror how a person actually reasons: *what am I
looking at → what did the system see → what does it mean → tell me more*.

---

## 5.3 Features against the brief

| Requirement | Implementation |
|---|---|
| Users can upload parking images | `st.file_uploader` accepting JPG/PNG, 15 MB cap |
| Display uploaded image | Shown side by side with the annotated output |
| Annotated output with clear colour coding | Green empty · red occupied · amber low-confidence, filled at 22% alpha with a legend stamped in |
| Show total slots, available slots, occupancy % | Four large metric cards above the fold |
| Clean, responsive, easy to use | Custom CSS cards, a colour-coded banner, and a wide two-column layout that reflows on narrow screens |
| Non-technical audience | No raw probability appears in the primary view — the headline is a sentence, not a number |

### Beyond the requirements

| Feature | Why it was added |
|---|---|
| **Demo scenes with ground truth** | A visitor with no parking photograph can still evaluate the system, and sees predictions checked against real labels |
| **Zone guidance** | Answers "which way do I turn?", not just "how many are free?" |
| **Adjustable threshold** | Lets an operator trade false "full" readings against false "free" readings, with the tuned default pre-selected |
| **Confidence tab** | Shows the score distribution, so a user can see whether the model was decisive or hedging |
| **Bay-by-bay table + CSV export** | Municipal operators need the underlying data, not only the picture |
| **Annotated image download** | Lets a result be attached to a report or shared |
| **Accuracy panel** | Overall test metrics plus per-weather and per-lot breakdowns, in the app rather than buried in a file |
| **Latency display** | Shows the system is fast enough for live use |

---

## 5.4 Handling uploads: where are the bays?

The classifier answers *"is this bay occupied?"*. It does not discover where
bays are — that is a survey, done once per fixed camera
([Step 1](01_problem_definition.md#13-understanding-parking-structure)).

An uploaded photograph therefore needs geometry from somewhere, and the app
offers three routes in descending order of reliability:

**1. Automatic camera recognition.** Each surveyed camera stores a visual
fingerprint: a 96×54 greyscale thumbnail, zero-meaned and unit-normalised, taken
as the *median* across that camera's scenes. An upload is fingerprinted the same
way and compared by normalised cross-correlation. Zero-meaning makes the match
invariant to brightness and contrast, so a camera surveyed on a sunny morning
still matches on a rainy afternoon. Above 0.80 similarity the surveyed layout is
reused automatically.

**2. Manual selection.** The operator picks the camera from a dropdown.

**3. Grid fallback.** For a genuinely unseen lot, a rows × cols grid is laid over
the frame, with an aisle gap so adjacent crops do not overlap (overlapping crops
would let one car be counted in two bays). **The app labels this clearly as
approximate**, both in a warning box and in the caption under the annotated
image.

That third path is the honest answer to a real limitation. Silently producing
confident-looking numbers from a grid that does not match the painted bays would
be worse than saying so.

---

## 5.5 Performance

| Measure | Value |
|---|---|
| Model load | Once per session, `@st.cache_resource` |
| Demo images, layouts, metrics | `@st.cache_data`, loaded once |
| Inference | One batched `predict` call for all bays, not a loop |
| 100-bay lot, CPU | Well under a second for the classification step |
| Model size | 24 MB — comfortably inside Streamlit Community Cloud's free tier |

Bays are cropped into a single stacked array and passed to `model.predict` as
one batch. Classifying bays one at a time would pay the per-call graph overhead
a hundred times over and turn a sub-second response into a visible wait.

---

## 5.6 Accessibility and robustness

- **Colour is never the only signal.** Every count that the colours encode also
  appears as a number in the metric cards and in the bay-by-bay table, so the
  dashboard is usable with colour-vision deficiency.
- **Dark mode.** The CSS defines a `prefers-color-scheme: dark` block so cards
  and notes stay readable in either theme.
- **Missing model.** If `models/parkvision_mobilenetv2.keras` is absent the app
  explains which script to run instead of raising a stack trace.
- **Missing metrics.** The threshold falls back to 0.5 and the sidebar says so,
  rather than crashing on a missing file.
- **No bays detected.** `generate_insight([])` returns a valid result with a
  clear message instead of dividing by zero.
- **Upload cap.** 15 MB, set in `.streamlit/config.toml`.

---

## 5.7 Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens on `http://localhost:8501`. The app needs `models/parkvision_mobilenetv2.keras`
and `data/demo/`, both committed to the repository, so a fresh clone runs
without retraining.

---

**Next:** [Step 7 — Deployment on Streamlit Cloud](06_deployment.md)
*(Step 6, testing and feedback, is generated by `scripts/04_robustness_test.py`
into [`results/robustness_report.md`](../results/robustness_report.md).)*
