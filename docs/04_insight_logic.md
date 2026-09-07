# Step 4 — Integration and Parking Insight Logic

*Implemented in [`src/insights.py`](../src/insights.py) and
[`src/overlay.py`](../src/overlay.py).*

---

## 4.1 What this step is for

A classifier produces 100 numbers between 0 and 1. A driver at the barrier
needs one sentence. This step is the whole distance between those two things —
the layer that turns predictions into a decision, and the layer that makes the
project a smart-city solution rather than a model demo.

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

`src/insights.py` deliberately imports **neither TensorFlow nor OpenCV**. The
decision rules are pure functions over plain numbers, which means they can be
tested exactly and reasoned about without a model in the loop.

---

## 4.2 From score to labelled bay

```python
score_to_prediction(space_id, score, threshold, uncertain_band)
```

| Step | Rule |
|---|---|
| Label | `occupied` if `P(occupied) ≥ threshold`, else `empty` |
| Confidence | `score` if occupied, `1 − score` if empty — always the probability *of the reported answer*, which is the number a user can interpret |
| Uncertainty flag | `abs(score − threshold) < 0.15` |

**Every bay still gets a definite label.** A driver cannot act on "maybe". But a
borderline call is *flagged*, drawn amber on the overlay, and counted separately
in the summary — so the dashboard is honest about what it does not know without
refusing to answer.

The threshold is not hard-coded: it is read from `results/metrics.json`, where
`scripts/03_evaluate.py` recorded the value chosen on the validation split.

---

## 4.3 Counting and occupancy

```python
total_slots     = len(predictions)
occupied_slots  = count of predictions labelled occupied
available_slots = total_slots − occupied_slots
occupancy_pct   = 100 × occupied_slots / total_slots
```

`available_slots` is computed as a **complement, never as a second count**. If
it were counted independently, a bay that failed to classify could vanish from
both totals and the two numbers would silently stop summing to the total the
user was shown.

---

## 4.4 Congestion levels

Exactly the bands the brief specifies:

| Occupancy | Level | Meaning shown to the driver |
|---|---|---|
| `< 40%` | 🟢 **Low** | Plenty of space available |
| `40% – 75%` | 🟠 **Moderate** | Filling up — park soon |
| `> 75%` | 🔴 **High** | Heavy congestion |

The boundaries are inclusive-at-the-top for Moderate (`40 ≤ x ≤ 75`), so
exactly 40% is Moderate and exactly 75% is Moderate, with High beginning above
75%. This is verified by an explicit boundary test rather than left to
inspection:

```python
classify_congestion(39.9) == "Low"       classify_congestion(75.0)  == "Moderate"
classify_congestion(40.0) == "Moderate"  classify_congestion(75.1)  == "High"
```

Thresholds live in `src/config.py` (`CONGESTION_LOW_MAX`,
`CONGESTION_MODERATE_MAX`) so a city with different policy can change them in
one place.

---

## 4.5 Recommendations

Each congestion level carries a headline, a recommendation and an explanation:

| Level | Recommendation | Explanation shown |
|---|---|---|
| Low | **Slots available — proceed to park.** | "The lot is lightly used. Drive in and pick any highlighted green slot." |
| Moderate | **Slots available — proceed, but park promptly.** | "Availability is shrinking. Head to the nearest green slot before it is taken." |
| High | **Parking almost full — try another area.** | "Very few slots remain. Consider an alternative lot to avoid circling traffic." |

### Conditional notes

Percentage bands alone miss cases that matter, so the engine adds situational
notes when they apply:

| Condition | Note raised |
|---|---|
| `available == 0` | "Every annotated bay is occupied — no space is available right now." |
| `available ≤ 3` | "Only N bay(s) left; they may be taken within minutes." |
| Any uncertain bays | "N of M bays (X%) were borderline calls — usually caused by shadow, glare or a partly occluded car." |
| Mean confidence `< 0.80` | "Average confidence is below 80%; consider a clearer or better-lit image." |

The `available ≤ 3` note exists because the percentage bands are misleading at
the margin: a 300-bay lot at 99% occupancy is "High congestion" with 3 free
bays, and a driver who is already at the entrance should be told the count, not
just the band.

---

## 4.6 Zone guidance — beyond the brief

User testing surfaced a gap the required outputs do not cover:

> *"Twelve bays are free — but which way do I turn?"*

`row_summary()` splits the lot into horizontal bands by the vertical centre of
each bay's polygon, so Zone 1 is the row nearest the camera (typically the
entrance) and Zone 3 the far end. For each zone it reports total, available,
occupied and occupancy percentage; `best_zone()` then picks the zone with the
most free bays, breaking ties toward the nearest.

The dashboard turns that into a sentence:

> **Head for Zone 1 — near the entrance.** 9 of 34 bays are free there (74% occupied).

This is the difference between an availability *number* and actual navigational
guidance.

---

## 4.7 Visual smart overlays

[`src/overlay.py`](../src/overlay.py) draws the colour-coded output the brief
requires:

| Colour | RGB | Meaning |
|---|---|---|
| 🟩 Green | `(22, 163, 74)` | Bay is empty |
| 🟥 Red | `(220, 38, 38)` | Bay is occupied |
| 🟨 Amber | `(234, 179, 8)` | Low-confidence call |

Three details that came directly out of usability testing:

1. **Fill as well as outline.** Outlines alone are hard to separate on a
   100-bay lot. Polygons are filled at 22% alpha, so colour reads at a glance
   while the cars underneath stay visible.
2. **True polygons, not rectangles.** The overlay draws the same quadrilateral
   the classifier was given, so the boxes sit on the painted bays instead of
   floating over them.
3. **Text with a dark halo.** Bay numbers and confidence percentages are drawn
   twice — a thick dark pass, then the coloured text — so labels stay readable
   over both pale tarmac and dark cars.

A translucent legend with the live free/total count is stamped into the corner,
so a downloaded screenshot is self-explanatory without the surrounding page.

---

## 4.8 Worked example

From a held-out PUCPR scene:

```
Input   1280 × 720 photograph · 100 surveyed bays

Model   100 sigmoid scores, 0.38 s total (3.8 ms per bay)

Counts  Total slots      100
        Occupied          77
        Available         23
        Occupancy      77.0 %

Bands   77.0 % > 75 %  →  HIGH congestion

Output  🔴 High congestion — Heavy congestion
        "Very few slots remain. Consider an alternative lot to
         avoid circling traffic."
        Recommendation: Parking almost full — try another area.

Notes   4 of 100 bays (4%) were borderline calls — usually caused by
        shadow, glare or a partly occluded car.

Zones   Zone 1 (near entrance)  34 bays · 11 free · 67.6 % occupied
        Zone 2 (middle)         33 bays ·  8 free · 75.8 % occupied
        Zone 3 (far end)        33 bays ·  4 free · 87.9 % occupied
        → Head for Zone 1, nearest the entrance.
```

*(Illustrative structure; the live numbers for all twelve held-out scenes,
with ground-truth comparison, are in
[`results/scene_level_results.csv`](../results/scene_level_results.csv).)*

---

## 4.9 Verification

Because the insight layer is pure Python, its behaviour is checked directly
rather than inferred from the dashboard:

- **Congestion boundaries** — all six edge cases (0, 39.9, 40, 75, 75.1, 100)
  assert to the expected band.
- **Aggregation** — a fixed vector of five scores produces the expected total,
  occupied, available, percentage, level and uncertain count.
- **Empty input** — zero bays returns a valid `ParkingInsight` with a clear
  message rather than dividing by zero.
- **End to end** — `scripts/04_robustness_test.py` compares predicted free-bay
  counts and congestion levels against PKLot's ground truth across all twelve
  held-out scenes.

---

**Next:** [Step 5 — Web App Development using Streamlit](05_web_app.md)
