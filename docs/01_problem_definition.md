# Step 1 — Understanding the Problem and Defining Requirements

*ParkVision AI · UrbanFlow AI Pvt. Ltd. · Scenario 1*

---

## 1.1 The problem, stated precisely

Drivers looking for a space in a busy urban lot do not have a *parking*
problem — they have an *information* problem. The space usually exists; the
driver simply cannot see it from the entrance. That missing information is what
turns into measurable harm:

| Consequence | Why it happens |
|---|---|
| Congestion inside and around the lot | Cars circulate slowly while searching, blocking through-traffic and the lot's own aisles |
| Wasted fuel and avoidable emissions | Time spent cruising is time spent burning fuel while moving almost nowhere |
| Lost time and driver frustration | Minutes per trip, repeated across every driver, every day |
| Poor use of infrastructure a city already owns | A lot can be 30% empty while drivers queue at the entrance believing it is full |

Studies of urban cruising consistently find that a substantial share of traffic
in dense districts is made up of drivers searching for parking rather than
travelling to a destination (see [references](references.md)). The lot is not
the bottleneck. **Knowing which bay is free** is the bottleneck.

**So the system's job is not to detect cars. It is to answer, for a specific
lot, at a specific moment: _how many bays are free, where are they, and should
I drive in?_**

---

## 1.2 System definition — input and output

The contract is deliberately narrow and testable.

```
INPUT
  One photograph of a parking lot, taken from an elevated fixed camera.
  Format:  JPG or PNG, any resolution (internally normalised)
  Plus:    the surveyed bay layout for that camera
           (a list of quadrilaterals, one per painted bay)

           ↓

PROCESSING
  For every bay in the layout:
     perspective-warp the quadrilateral to a 224 x 224 patch
     classify that patch  ->  P(occupied) in [0, 1]
     apply the tuned decision threshold  ->  Occupied | Empty

           ↓

OUTPUT
  Per bay      : status (Occupied / Empty) + confidence
  Whole lot    : total slots, occupied slots, available slots
                 occupancy percentage
                 congestion level  (Low / Moderate / High)
                 recommendation    (proceed to park / try another area)
                 zone guidance     (which part of the lot to drive to)
  Visual       : the original photograph with every bay outlined —
                 green = empty, red = occupied, amber = low confidence
```

This is a **slot-level** contract, not an image-level one. A system that
returned "this lot looks busy" would satisfy no one: a driver needs a count,
and a municipality needs a count it can act on. Every design decision below
follows from committing to per-bay output.

---

## 1.3 Understanding parking structure

Real lots are not tidy grids, and the three lots in PKLot demonstrate the range:

| Lot | Bays annotated | Structure |
|---|---|---|
| PUCPR | 100 | Large lot seen from a tall building; bays are small, densely packed, and heavily foreshortened at the far end |
| UFPR04 | ~28 | Shot from a 4th floor; bays are larger in frame and moderately angled |
| UFPR05 | ~40 | Shot from a 5th floor; steeper viewing angle, stronger perspective distortion |

Two structural facts drove the architecture:

**Bays are quadrilaterals, not rectangles.** From an elevated camera, a
rectangular bay projects to a trapezium, and its shape differs across the frame.
Cropping an axis-aligned box around such a bay pulls in a slice of the two
neighbouring bays, so a car parked *next door* can leak into the crop. The
system therefore perspective-warps each quadrilateral to a square
(`src/slots.py`), which deskews the bay and keeps the neighbours out.

**The camera does not move.** A parking camera is bolted to a building. Bay
geometry is therefore a one-time survey per camera, not something to
re-discover in every frame — the same assumption PKLot's own annotations make.
This is what makes a lightweight classifier viable where full object detection
would be overkill.

---

## 1.4 Choosing the approach

The brief allows either crop-based classification or full-image detection.

| | Crop classification (**chosen**) | Full-image detection (YOLO) |
|---|---|---|
| Output | One label per surveyed bay | Boxes around whatever it finds |
| Answers "is bay 37 free?" | Directly | Only after matching boxes to bays |
| Empty bays | Explicitly classified | Invisible — an empty bay has nothing to detect, so absence of a box must be *inferred* |
| Labels needed | Per-bay occupied/empty (PKLot ships these) | Per-car bounding boxes |
| Model size | MobileNetV2, 2.3 M parameters, ~9 MB | YOLO, tens of millions of parameters |
| Runs on free-tier CPU hosting | Yes — ~10 ms per bay end to end, measured | Slow, and a heavy dependency install |
| Handles a car straddling two bays | Each bay judged on its own | Ambiguous box-to-bay assignment |

The decisive argument is the third row. Detection finds cars; the question is
about *bays*. An empty bay produces no detection, so a detector must reason
about absence — and absence is indistinguishable from a missed detection.
Classifying each surveyed bay makes "empty" a first-class, directly supervised
answer rather than an inference from silence.

This also matches the brief's guidance that "Models like MobileNet or
EfficientNet are suitable for classification", and it is the approach the PKLot
authors themselves used.

**Cost of this choice, stated honestly:** the system cannot analyse a lot whose
bays have never been surveyed. The app handles that with an explicit grid
fallback that is labelled as approximate — see
[Step 5](05_web_app.md).

---

## 1.5 Real-world challenges identified

These were identified before modelling and each one is answered by a specific
design decision, then measured in
[the testing report](../results/robustness_report.md).

| Challenge | Why it is hard | How the system answers it |
|---|---|---|
| **Shadows and lighting** | A dark shadow filling an empty bay looks like a dark car; low sun creates glare that washes out bay markings | Training data spans sunny, overcast and rainy days across many hours; brightness and contrast augmentation during training; stress-tested at ±45% brightness and ±50% contrast |
| **Occlusion** | A car in a nearer row hides part of a farther bay; trees and lamp posts cut across bays | The 8% padded warp keeps the bay centred, where occlusion is least; low-confidence bays are flagged amber rather than asserted |
| **Weather** | Rain darkens tarmac and adds reflections; wet ground can mimic a car's sheen | Rainy days form ~20% of the candidate scenes and appear in all three splits; accuracy is reported *per weather condition*, not only on average |
| **Perspective distortion** | A bay at the far end of PUCPR is a fraction of the size of a near one | Every bay is perspective-warped to the same 224×224 square before classification |
| **Camera drift** | A bracket knocked out of alignment shifts every bay | Rotation/translation augmentation; stress-tested at ±8° and −12° |
| **Class imbalance by time of day** | Lots are near-empty at 06:00 and near-full at 14:00 | Crops are quota-filled class-balanced across scenes, so neither class is dominated by one time of day |
| **Near-duplicate frames** | PKLot photographs are five minutes apart and nearly identical | Splits are grouped by whole (lot, calendar date), so a frame's near-twin can never sit across the train/test boundary — see [Step 2](02_data_preprocessing.md) |

---

## 1.6 Final outputs the system must produce

Directly from the brief, all implemented in `src/insights.py` and surfaced in
`app.py`:

1. **Total slots** — number of bays analysed.
2. **Occupied slots** — bays classified as occupied.
3. **Available slots** — `total − occupied`.
4. **Occupancy percentage** — `100 × occupied / total`.
5. **Congestion level** — Low `< 40%`, Moderate `40–75%`, High `> 75%`.
6. **Recommendation** — "Slots available — proceed to park" or
   "Parking almost full — try another area".

Two outputs go beyond the required list because testing showed users asked for
them (see the [feedback section](../results/robustness_report.md)):

7. **Zone guidance** — which part of the lot has the most free bays, so the
   driver knows which way to turn, not merely that space exists.
8. **Confidence reporting** — bays the model was unsure about are shown in
   amber and counted separately, so the dashboard never presents a coin-flip
   as a fact.

---

## 1.7 Success criteria set before building

| Criterion | Target | Where it is verified |
|---|---|---|
| Slot-level accuracy on unseen days | ≥ 95% | [`results/metrics.json`](../results/metrics.json) |
| Accuracy holds across sunny / cloudy / rainy | No condition below 90% | `by_weather` in `metrics.json` |
| Free-bay count error on a whole scene | ≤ 3 bays | [`results/robustness_report.md`](../results/robustness_report.md) |
| Congestion level called correctly | ≥ 10 of 12 held-out scenes | same |
| End-to-end latency for a 100-bay lot | < 3 s on CPU | same |
| Usable by a non-technical driver | No raw probabilities in the primary view | [Step 5](05_web_app.md) |

---

**Next:** [Step 2 — Data Collection and Preprocessing](02_data_preprocessing.md)
