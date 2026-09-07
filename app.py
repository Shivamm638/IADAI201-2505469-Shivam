"""
ParkVision AI — Intelligent Urban Parking Analytics & Space Optimisation Platform

Streamlit dashboard for UrbanFlow AI Pvt. Ltd.

Upload a parking-lot photograph (or pick one of the held-out demo scenes) and
the app crops every bay, classifies it with the fine-tuned MobileNetV2 model,
draws colour-coded overlays, and turns the result into availability metrics,
a congestion level and a recommendation.

Run locally:   streamlit run app.py
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config as C
from src import insights as I
from src import layouts as L
from src import overlay as O
from src import inference as INF

# ==========================================================================
# Page setup
# ==========================================================================
st.set_page_config(
    page_title="ParkVision AI — Smart Parking Analytics",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
  .block-container { padding-top: 2.0rem; padding-bottom: 3rem; max-width: 1400px; }

  .pv-hero {
      background: linear-gradient(120deg, #0F2C4C 0%, #17457A 55%, #1E63A8 100%);
      color: #fff; padding: 1.5rem 1.75rem; border-radius: 14px; margin-bottom: 1.25rem;
  }
  .pv-hero h1 { margin: 0 0 .3rem 0; font-size: 1.85rem; font-weight: 700; letter-spacing: -.02em; }
  .pv-hero p  { margin: 0; opacity: .88; font-size: .97rem; }
  .pv-hero .pv-tag {
      display: inline-block; margin-top: .7rem; padding: .18rem .6rem; border-radius: 999px;
      background: rgba(255,255,255,.16); font-size: .76rem; letter-spacing: .02em;
  }

  .pv-metric {
      background: #fff; border: 1px solid #E4E8EF; border-radius: 12px;
      padding: 1rem 1.1rem; height: 100%;
  }
  .pv-metric .pv-label {
      font-size: .75rem; text-transform: uppercase; letter-spacing: .07em;
      color: #64748B; font-weight: 600; margin-bottom: .3rem;
  }
  .pv-metric .pv-value { font-size: 2.05rem; font-weight: 700; line-height: 1.05; color: #0F172A; }
  .pv-metric .pv-sub   { font-size: .8rem; color: #64748B; margin-top: .25rem; }

  .pv-banner { border-radius: 12px; padding: 1.15rem 1.3rem; color: #fff; margin: .4rem 0 1rem 0; }
  .pv-banner h3 { margin: 0 0 .3rem 0; font-size: 1.25rem; font-weight: 700; }
  .pv-banner p  { margin: 0; font-size: .95rem; opacity: .95; }
  .pv-banner .pv-rec {
      margin-top: .7rem; padding: .55rem .8rem; border-radius: 8px;
      background: rgba(255,255,255,.18); font-weight: 600; font-size: 1rem;
  }

  .pv-bar { height: 13px; border-radius: 999px; background: #E8ECF3; overflow: hidden; margin-top: .5rem; }
  .pv-bar > div { height: 100%; border-radius: 999px; }

  .pv-note {
      background: #FFF8E6; border-left: 4px solid #E0A800; padding: .6rem .85rem;
      border-radius: 6px; font-size: .88rem; margin-bottom: .45rem; color: #4A3B00;
  }
  .pv-foot { color: #64748B; font-size: .82rem; line-height: 1.6; }

  @media (prefers-color-scheme: dark) {
      .pv-metric { background: #111827; border-color: #263041; }
      .pv-metric .pv-value { color: #F1F5F9; }
      .pv-note { background: #2A2410; color: #F5E9C6; }
  }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ==========================================================================
# Cached resources
# ==========================================================================
@st.cache_resource(show_spinner="Loading the MobileNetV2 occupancy model…")
def get_model():
    return INF.load_model()


@st.cache_resource
def get_layouts():
    library = L.load_library()
    if not library and C.DEMO_INDEX_PATH.exists():
        library = L.build_library()
    return library


@st.cache_data
def get_demo_index():
    if not C.DEMO_INDEX_PATH.exists():
        return []
    return json.loads(C.DEMO_INDEX_PATH.read_text(encoding="utf-8"))


@st.cache_data
def get_metrics():
    if not C.METRICS_PATH.exists():
        return {}
    return json.loads(C.METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_demo_image(name: str) -> np.ndarray:
    return np.array(Image.open(C.DEMO_DIR / name).convert("RGB"))


# ==========================================================================
# Small render helpers
# ==========================================================================
def metric_card(label: str, value, sub: str = "") -> str:
    return (
        f'<div class="pv-metric"><div class="pv-label">{label}</div>'
        f'<div class="pv-value">{value}</div>'
        f'<div class="pv-sub">{sub}</div></div>'
    )


def congestion_banner(insight: I.ParkingInsight) -> str:
    return (
        f'<div class="pv-banner" style="background:{insight.colour};">'
        f'<h3>{insight.emoji} {insight.congestion_level} congestion — {insight.headline}</h3>'
        f'<p>{insight.detail}</p>'
        f'<div class="pv-rec">Recommendation: {insight.recommendation}</div>'
        f'<div class="pv-bar" style="background:rgba(255,255,255,.28);">'
        f'<div style="width:{min(100.0, insight.occupancy_pct):.1f}%;background:#fff;"></div></div>'
        f'<p style="margin-top:.35rem;font-size:.82rem;">'
        f'{insight.occupancy_pct:.1f}% of bays occupied</p>'
        f"</div>"
    )


def to_png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


# ==========================================================================
# Header
# ==========================================================================
st.markdown(
    '<div class="pv-hero">'
    "<h1>🅿️ ParkVision AI</h1>"
    "<p>Intelligent Urban Parking Analytics &amp; Space Optimisation Platform "
    "— UrbanFlow AI Pvt. Ltd.</p>"
    '<span class="pv-tag">Slot-level occupancy detection · MobileNetV2 transfer learning · '
    "Real-time availability insights</span>"
    "</div>",
    unsafe_allow_html=True,
)

metrics = get_metrics()
demo_index = get_demo_index()

# ==========================================================================
# Sidebar
# ==========================================================================
with st.sidebar:
    st.header("⚙️ Controls")

    if metrics:
        test = metrics.get("test", {})
        st.success(
            f"**Model ready**\n\n"
            f"Test accuracy **{test.get('accuracy', 0) * 100:.2f}%**  \n"
            f"ROC AUC **{test.get('roc_auc', 0):.4f}**  \n"
            f"Errors: {test.get('errors', 0)} / {test.get('n_samples', 0)} bays"
        )
    else:
        st.info("Model metrics not found — run `python scripts/03_evaluate.py`.")

    st.subheader("Detection")
    default_threshold = float(metrics.get("operating_threshold", C.DEFAULT_THRESHOLD))
    threshold = st.slider(
        "Decision threshold — P(occupied)",
        0.05, 0.95, default_threshold, 0.01,
        help=(
            "Raise it to call a bay occupied only when the model is very sure "
            "(fewer false 'full' readings). Lower it to be cautious about "
            "sending drivers to a bay that is actually taken."
        ),
    )
    if abs(threshold - default_threshold) > 1e-6:
        st.caption(f"Tuned default is {default_threshold:.2f} (chosen on the validation split).")

    st.subheader("Overlay")
    show_ids = st.checkbox("Show bay numbers", value=False)
    show_conf = st.checkbox("Show per-bay confidence", value=False)
    mark_uncertain = st.checkbox(
        "Flag low-confidence bays in amber", value=True,
        help="Bays whose score sits close to the threshold are shown in amber.",
    )
    show_legend = st.checkbox("Show legend on image", value=True)

    st.divider()
    st.caption(
        "**Colour key**  \n"
        "🟩 Green — bay is empty  \n"
        "🟥 Red — bay is occupied  \n"
        "🟨 Amber — low-confidence call"
    )
    st.divider()
    st.caption(
        "Congestion bands  \n"
        f"Low < {C.CONGESTION_LOW_MAX:.0f}% · "
        f"Moderate {C.CONGESTION_LOW_MAX:.0f}–{C.CONGESTION_MODERATE_MAX:.0f}% · "
        f"High > {C.CONGESTION_MODERATE_MAX:.0f}%"
    )

# ==========================================================================
# Input
# ==========================================================================
st.subheader("1 · Choose a parking-lot image")

library = get_layouts()
tab_demo, tab_upload = st.tabs(["📁 Demo scenes (held-out)", "⬆️ Upload your own"])

image_rgb = None
spaces = None
scene_meta = None
geometry_source = ""

with tab_demo:
    if not demo_index:
        st.warning("No demo scenes found. Run `python scripts/01_build_dataset.py` first.")
    else:
        options = {
            f"{e['lot'].upper()} · {e['weather']} · {e['date']} {e['time'][:5]} "
            f"· {e['n_spaces']} bays · {e['occupancy_pct_truth']:.0f}% full": e
            for e in demo_index
        }
        choice = st.selectbox(
            "These scenes come from calendar days the model never trained on:",
            list(options),
        )
        entry = options[choice]
        image_rgb = load_demo_image(entry["image"])
        spaces = entry["spaces"]
        scene_meta = entry
        geometry_source = f"Surveyed layout for camera **{entry['lot'].upper()}**"

with tab_upload:
    uploaded = st.file_uploader(
        "Upload a parking-lot photograph (JPG or PNG)",
        type=["jpg", "jpeg", "png"],
        help="Best results with an elevated view of the whole lot.",
    )
    if uploaded is not None:
        image_rgb = np.array(Image.open(uploaded).convert("RGB"))
        scene_meta = None

        matched, score, all_scores = L.match_layout(image_rgb, library)
        st.markdown("**Bay geometry**")

        if matched is not None:
            st.success(
                f"Recognised camera **{matched.lot.upper()}** "
                f"(match {score * 100:.1f}%). Using its surveyed layout of "
                f"{matched.n_spaces} bays."
            )
        else:
            st.info(
                "This view does not match any surveyed camera"
                + (f" (best match {score * 100:.1f}%)." if all_scores else ".")
                + " Pick a camera below, or lay a grid over the image."
            )

        mode_options = ["Auto / surveyed camera", "Choose a camera", "Lay a grid over the image"]
        default_mode = 0 if matched is not None else (1 if library else 2)
        mode = st.radio("How should bays be located?", mode_options,
                        index=default_mode, horizontal=True)

        if mode == "Auto / surveyed camera" and matched is not None:
            spaces = matched.spaces
            geometry_source = f"Auto-matched surveyed layout — {matched.label}"
        elif mode == "Choose a camera" and library:
            choices = L.layout_choices(library)
            picked = st.selectbox(
                "Camera", [k for k, _ in choices],
                format_func=lambda k: dict(choices)[k],
            )
            spaces = library[picked].spaces
            geometry_source = f"Manually selected layout — {library[picked].label}"
        else:
            col_a, col_b = st.columns(2)
            rows = col_a.slider("Rows of bays", 1, 10, 4)
            cols = col_b.slider("Bays per row", 2, 20, 10)
            spaces = L.grid_layout(rows, cols)
            geometry_source = (
                f"Approximate {rows}×{cols} grid — not a surveyed layout, "
                "so counts are indicative only"
            )
            st.warning(
                "Grid mode divides the frame evenly. It is a fallback for an "
                "unsurveyed lot: bays will not line up with the painted bays, "
                "so treat the counts as indicative."
            )

# ==========================================================================
# Analysis
# ==========================================================================
if image_rgb is None:
    st.info("👆 Pick a demo scene or upload an image to run the analysis.")
    st.stop()

if not C.MODEL_PATH.exists():
    st.error(
        f"No trained model at `{C.MODEL_PATH.name}`. "
        "Run `python scripts/02_train.py` before starting the app."
    )
    st.stop()

model = get_model()

with st.spinner("Analysing every parking bay…"):
    predictions, timing = INF.predict_slots(
        image_rgb, spaces, model=model, threshold=threshold
    )
    insight = I.generate_insight(predictions)
    annotated = O.annotate(
        image_rgb, predictions, insight,
        show_ids=show_ids,
        show_confidence=show_conf,
        mark_uncertain=mark_uncertain,
        legend=show_legend,
    )

st.subheader("2 · Slot-level detection")

if not predictions:
    st.error(
        "No parking bays could be read from this image. The bay layout may not "
        "line up with the photograph — try selecting a camera manually, or use "
        "grid mode."
    )
    st.stop()

st.caption(f"{geometry_source} · {timing['n_slots']} bays analysed in "
           f"{timing['total_seconds']:.2f}s ({timing['ms_per_slot']:.1f} ms per bay)")

left, right = st.columns(2)
with left:
    st.markdown("**Demo scene**" if scene_meta else "**Uploaded image**")
    st.image(image_rgb, use_container_width=True)
with right:
    st.markdown("**Annotated output** — 🟩 empty · 🟥 occupied")
    st.image(annotated, use_container_width=True)

st.download_button(
    "⬇️ Download annotated image (PNG)",
    data=to_png_bytes(annotated),
    file_name="parkvision_annotated.png",
    mime="image/png",
)

# ==========================================================================
# Metrics + insight
# ==========================================================================
st.subheader("3 · Real-time availability")

c1, c2, c3, c4 = st.columns(4)
c1.markdown(metric_card("Total slots", insight.total_slots, "bays in this lot"),
            unsafe_allow_html=True)
c2.markdown(metric_card("Occupied", insight.occupied_slots,
                        f"{insight.occupancy_pct:.1f}% of the lot"),
            unsafe_allow_html=True)
c3.markdown(metric_card("Available", insight.available_slots,
                        "ready to park in now"),
            unsafe_allow_html=True)
c4.markdown(metric_card("Congestion", insight.congestion_level,
                        f"avg confidence {insight.mean_confidence * 100:.1f}%"),
            unsafe_allow_html=True)

st.markdown(congestion_banner(insight), unsafe_allow_html=True)

for note in insight.notes:
    st.markdown(f'<div class="pv-note">ℹ️ {note}</div>', unsafe_allow_html=True)

# ==========================================================================
# Deeper analytics
# ==========================================================================
st.subheader("4 · Parking utilisation analytics")

zones = I.row_summary(predictions)
tab_zone, tab_conf, tab_table, tab_truth = st.tabs(
    ["🧭 Where to park", "📊 Confidence", "📋 Bay-by-bay", "✅ Accuracy on this scene"]
)

with tab_zone:
    if zones:
        target = I.best_zone(zones)
        if target:
            st.success(
                f"**Head for {target['zone']} — {target['position'].lower()}.** "
                f"{target['available']} of {target['total']} bays are free there "
                f"({target['occupancy_pct']:.0f}% occupied)."
            )
        else:
            st.error("No zone has a free bay. Try another parking location.")
        zone_df = pd.DataFrame(zones).rename(
            columns={
                "zone": "Zone", "position": "Position", "total": "Bays",
                "available": "Available", "occupied": "Occupied",
                "occupancy_pct": "Occupancy %",
            }
        )
        st.dataframe(zone_df, use_container_width=True, hide_index=True)
        st.bar_chart(zone_df.set_index("Zone")[["Available", "Occupied"]],
                     color=["#16A34A", "#DC2626"])
    else:
        st.caption("Zone analysis needs bay geometry with at least three rows.")

with tab_conf:
    scores = np.array([p.score for p in predictions])
    st.caption(
        "Scores cluster at 0 and 1 when the model is decisive. Bays near the "
        f"threshold ({threshold:.2f}) are the ones flagged amber."
    )
    counts, edges = np.histogram(scores, bins=20, range=(0, 1))
    st.bar_chart(
        pd.DataFrame({"bays": counts},
                     index=[f"{edges[i]:.2f}" for i in range(len(counts))])
    )
    d1, d2, d3 = st.columns(3)
    d1.metric("Mean confidence", f"{insight.mean_confidence * 100:.1f}%")
    d2.metric("Low-confidence bays", insight.uncertain_slots)
    d3.metric("Decisive bays (>90%)",
              int(sum(1 for p in predictions if p.confidence > 0.90)))

with tab_table:
    table = pd.DataFrame(
        [
            {
                "Bay": p.space_id,
                "Status": p.status.title(),
                "P(occupied)": round(p.score, 4),
                "Confidence %": round(p.confidence * 100, 1),
                "Low confidence": "Yes" if p.uncertain else "",
            }
            for p in predictions
        ]
    ).sort_values("Bay")
    st.dataframe(table, use_container_width=True, hide_index=True, height=340)
    st.download_button(
        "⬇️ Download bay-level results (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="parkvision_slot_results.csv",
        mime="text/csv",
    )

with tab_truth:
    if scene_meta:
        truth = INF.accuracy_against_truth(predictions, scene_meta["spaces"])
        if truth:
            st.caption(
                "This demo scene carries PKLot's hand-drawn ground truth, and its "
                "calendar day was never used in training — so this is an honest "
                "read of unseen-image performance."
            )
            t1, t2, t3 = st.columns(3)
            t1.metric("Bays correct", f"{truth['n_correct']} / {truth['n_compared']}",
                      f"{truth['accuracy'] * 100:.1f}% accuracy")
            t2.metric("Available — predicted vs actual",
                      f"{truth['predicted_available']} vs {truth['true_available']}")
            t3.metric("Counting error", f"{truth['count_error']} bays",
                      delta_color="inverse")
            comparison = pd.DataFrame(
                {
                    "Metric": ["Occupied bays", "Available bays"],
                    "Predicted": [truth["predicted_occupied"], truth["predicted_available"]],
                    "Ground truth": [truth["true_occupied"], truth["true_available"]],
                }
            )
            st.dataframe(comparison, use_container_width=True, hide_index=True)
        else:
            st.info("This scene has no ground-truth labels to compare against.")
    else:
        st.info(
            "Ground truth is only available for the built-in demo scenes. "
            "An uploaded photograph has no labels to compare against."
        )

# ==========================================================================
# Footer
# ==========================================================================
st.divider()
if metrics:
    with st.expander("📈 How the model performs overall (held-out test split)"):
        test = metrics["test"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{test['accuracy'] * 100:.2f}%")
        m2.metric("Precision (occupied)", f"{test['precision_occupied'] * 100:.2f}%")
        m3.metric("Recall (occupied)", f"{test['recall_occupied'] * 100:.2f}%")
        m4.metric("ROC AUC", f"{test['roc_auc']:.4f}")

        if metrics.get("by_weather"):
            st.markdown("**Accuracy by weather condition**")
            st.dataframe(
                pd.DataFrame(metrics["by_weather"]).T.reset_index().rename(
                    columns={"index": "Weather", "n": "Bays",
                             "accuracy": "Accuracy", "share_occupied": "Share occupied"}
                ),
                use_container_width=True, hide_index=True,
            )
        if metrics.get("by_lot"):
            st.markdown("**Accuracy by parking lot**")
            st.dataframe(
                pd.DataFrame(metrics["by_lot"]).T.reset_index().rename(
                    columns={"index": "Lot", "n": "Bays",
                             "accuracy": "Accuracy", "share_occupied": "Share occupied"}
                ),
                use_container_width=True, hide_index=True,
            )

st.markdown(
    '<div class="pv-foot">'
    "<b>ParkVision AI</b> · Slot-level occupancy classification with MobileNetV2 "
    "transfer learning (ImageNet weights, two-phase fine-tuning). "
    "Trained on the PKLot dataset (Almeida et al., <i>Expert Systems with "
    "Applications</i>, 2015) with train/validation/test split by whole "
    "(lot, calendar-date) groups to prevent near-duplicate leakage.<br>"
    "Built for the CRS Artificial Intelligence summative assessment — "
    "<i>AI in Action: Building Intelligent Vision Systems for Smart Cities &amp; "
    "Smart Retail</i>, Scenario 1."
    "</div>",
    unsafe_allow_html=True,
)
