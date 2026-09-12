"""
ParkVision AI — Intelligent Urban Parking Analytics & Space Optimisation Platform

Streamlit dashboard for UrbanFlow AI Pvt. Ltd.

Pick a parking lot (or upload a photograph) and the app crops every bay,
classifies it with the fine-tuned MobileNetV2 model, draws colour-coded
overlays, and answers the question a driver actually has: how many bays are
free, where are they, and should I drive in?

The layout leads with that answer. Everything technical — confidence
distributions, per-bay tables, the model's own uncertainty — sits below it for
whoever wants to look.

Run locally:   python -m streamlit run app.py
"""

from __future__ import annotations

import io
import json
import random
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

# The theme itself (background, surfaces, borders, widget colours) lives in
# .streamlit/config.toml, so native widgets pick it up consistently. This block
# only styles the app's own cards.
#
# Surfaces are translucent slate and text inherits its colour rather than
# being fixed. `prefers-color-scheme` is deliberately not used: it follows the
# operating system, while Streamlit's theme is set independently, and keying
# off it once rendered dark cards on a white page. Translucent surfaces stay
# correct if a user switches the theme from Streamlit's settings menu.
#
# Status colours reach this CSS as custom properties set per element
# (--pv-accent, --pv-tint, --pv-edge, --pv-glow); see `accent_vars`.
CSS = """
<style>
  .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1450px; }

  :root {
      --pv-surface:   rgba(148,163,184,.06);
      --pv-surface-2: rgba(148,163,184,.10);
      --pv-line:      rgba(148,163,184,.20);
  }

  /* Every image — photographs, thumbnails, bay crops — gets the same frame. */
  [data-testid="stImage"] img {
      border-radius: 10px; border: 1px solid var(--pv-line);
  }

  /* ---------------------------------------------------------------- hero */
  .pv-hero {
      position: relative; overflow: hidden;
      background:
          radial-gradient(900px 260px at 88% -30%, rgba(96,165,250,.40), transparent 62%),
          linear-gradient(118deg, #081A36 0%, #122F66 48%, #1D4ED8 100%);
      border: 1px solid rgba(96,165,250,.30);
      box-shadow: 0 18px 50px -24px rgba(59,130,246,.65);
      color: #fff; padding: 1.45rem 1.8rem; border-radius: 16px; margin-bottom: 1.1rem;
  }
  .pv-hero h1 { margin: 0 0 .3rem 0; font-size: 1.85rem; font-weight: 800; letter-spacing: -.02em; color: #fff; }
  .pv-hero p  { margin: 0; opacity: .86; font-size: .95rem; color: #fff; }
  .pv-hero .pv-tag {
      display: inline-block; margin-top: .7rem; padding: .2rem .65rem; border-radius: 999px;
      background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.18);
      font-size: .75rem; letter-spacing: .02em;
  }
  .pv-live {
      display: inline-block; width: .5rem; height: .5rem; border-radius: 50%;
      background: #22C55E; box-shadow: 0 0 0 0 rgba(34,197,94,.7);
      margin-right: .45rem; vertical-align: middle; animation: pv-pulse 2s infinite;
  }
  @keyframes pv-pulse {
      0%   { box-shadow: 0 0 0 0 rgba(34,197,94,.65); }
      70%  { box-shadow: 0 0 0 .5rem rgba(34,197,94,0); }
      100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
  }
  @media (prefers-reduced-motion: reduce) { .pv-live { animation: none; } }

  /* ------------------------------------------------------------- verdict */
  .pv-verdict {
      display: flex; gap: 1.7rem; align-items: stretch; flex-wrap: wrap;
      background: linear-gradient(135deg, var(--pv-tint) 0%, var(--pv-surface) 58%);
      border: 1px solid var(--pv-line); border-left: 8px solid var(--pv-accent);
      border-radius: 16px; padding: 1.35rem 1.55rem; margin: .2rem 0 1.1rem 0;
      box-shadow: 0 18px 50px -28px var(--pv-glow);
  }
  .pv-verdict-num { min-width: 190px; }
  .pv-big {
      font-size: 4.6rem; line-height: .9; font-weight: 800; letter-spacing: -.04em;
      color: var(--pv-accent); text-shadow: 0 0 28px var(--pv-glow);
  }
  .pv-big-label {
      font-size: 1.0rem; font-weight: 700; text-transform: uppercase;
      letter-spacing: .1em; color: inherit; margin-top: .35rem;
  }
  .pv-big-sub { font-size: .84rem; opacity: .62; margin-top: .25rem; }
  .pv-verdict-body { flex: 1 1 340px; }
  .pv-verdict-body h3 { margin: .6rem 0 .3rem 0; font-size: 1.32rem; font-weight: 700; color: inherit; }
  .pv-verdict-body p { margin: 0 0 .3rem 0; font-size: .95rem; opacity: .86; }
  .pv-verdict-meta { font-size: .82rem !important; opacity: .6 !important; margin-top: .6rem !important; }

  .pv-pill {
      display: inline-block; padding: .22rem .78rem; border-radius: 999px;
      background: var(--pv-tint); color: var(--pv-accent); border: 1px solid var(--pv-edge);
      font-size: .8rem; font-weight: 700; letter-spacing: .02em;
  }

  /* -------------------------------------------------------------- metric */
  .pv-metric {
      background: var(--pv-surface); border: 1px solid var(--pv-line);
      border-top: 3px solid var(--pv-accent, rgba(148,163,184,.35));
      border-radius: 12px; padding: .95rem 1.05rem; height: 100%;
  }
  .pv-metric .pv-label {
      font-size: .72rem; text-transform: uppercase; letter-spacing: .08em;
      opacity: .6; font-weight: 700; margin-bottom: .3rem;
  }
  .pv-metric .pv-value { font-size: 1.95rem; font-weight: 800; line-height: 1.05; color: var(--pv-accent, inherit); }
  .pv-metric .pv-sub   { font-size: .78rem; opacity: .6; margin-top: .25rem; }

  /* -------------------------------------------------------------- banner */
  .pv-banner {
      background: linear-gradient(135deg, var(--pv-tint) 0%, rgba(148,163,184,.04) 70%);
      border: 1px solid var(--pv-edge); border-radius: 14px;
      padding: 1.05rem 1.3rem; margin: .3rem 0 1rem 0; color: inherit;
  }
  .pv-banner h3 { margin: 0 0 .3rem 0; font-size: 1.18rem; font-weight: 700; color: var(--pv-accent); }
  .pv-banner p  { margin: 0; font-size: .93rem; opacity: .86; }
  .pv-banner .pv-rec {
      margin-top: .7rem; padding: .55rem .85rem; border-radius: 8px;
      background: var(--pv-surface-2); border-left: 3px solid var(--pv-accent);
      font-weight: 600; font-size: .97rem;
  }
  .pv-bar {
      height: 10px; border-radius: 999px; overflow: hidden; margin-top: .65rem;
      background: rgba(148,163,184,.16);
  }
  .pv-bar > div {
      height: 100%; border-radius: 999px; background: var(--pv-accent);
      box-shadow: 0 0 14px var(--pv-glow);
  }

  .pv-note {
      background: rgba(245,158,11,.10); border: 1px solid rgba(245,158,11,.28);
      border-left: 4px solid #F59E0B; padding: .55rem .85rem; border-radius: 8px;
      font-size: .87rem; margin-bottom: .4rem; color: inherit;
  }

  /* -------------------------------------------------------------- landing */
  .pv-land {
      background: linear-gradient(135deg, rgba(59,130,246,.10) 0%, var(--pv-surface) 55%);
      border: 1px solid var(--pv-line); border-radius: 16px;
      padding: 1.5rem 1.7rem; margin-bottom: 1rem;
  }
  .pv-land h2 { margin: 0 0 .5rem 0; font-size: 1.38rem; font-weight: 800; color: inherit; }
  .pv-land p  { margin: 0 0 .5rem 0; opacity: .8; font-size: .95rem; line-height: 1.6; }
  .pv-flow { display: flex; gap: .6rem; flex-wrap: wrap; margin-top: 1rem; }
  .pv-step {
      flex: 1 1 150px; background: var(--pv-surface); border: 1px solid var(--pv-line);
      border-radius: 12px; padding: .75rem .9rem;
  }
  .pv-step b { display: block; font-size: .86rem; color: #60A5FA; margin-bottom: .2rem; }
  .pv-step span { font-size: .8rem; opacity: .7; line-height: 1.45; }

  /* ---------------------------------------------------------------- crop */
  .pv-crop-cap {
      text-align: center; font-size: .78rem; opacity: .72;
      margin-top: .25rem; line-height: 1.35;
  }
  .pv-crop-cap b { display: block; font-size: .86rem; color: inherit; opacity: 1; }

  .pv-foot { opacity: .58; font-size: .81rem; line-height: 1.6; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ==========================================================================
# Cached resources
# ==========================================================================
@st.cache_resource(show_spinner="Loading the MobileNetV2 occupancy model…")
def get_model():
    # No warm-up predict here. It looks like an obvious win — pay TensorFlow's
    # graph tracing behind the loading spinner instead of during the first
    # analysis — but measured on CPU it cost 2.6 s at batch 1 (4.7 s at batch
    # 64) and moved the first 100-bay scene only from 8.0 s to 6.6 s, while
    # repeat runs of identical work varied between 3.6 s and 5.7 s on their
    # own. The benefit sat inside the noise, so the startup cost is not worth
    # paying. The latency the dashboard reports is always the real measurement.
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
def measured_ms_per_bay() -> float | None:
    """
    End-to-end latency (crop + classify) measured over the held-out scenes by
    scripts/04_robustness_test.py. Read from the results rather than typed in,
    so the landing page can never quote a speed the system has not achieved.
    """
    path = C.RESULTS_DIR / "robustness.json"
    if not path.exists():
        return None
    scenes = json.loads(path.read_text(encoding="utf-8")).get("scene_test", [])
    bays = sum(s.get("total_slots", 0) for s in scenes)
    return 1000.0 * sum(s.get("seconds", 0.0) for s in scenes) / bays if bays else None


@st.cache_data
def load_demo_image(name: str) -> np.ndarray:
    return np.array(Image.open(C.DEMO_DIR / name).convert("RGB"))


@st.cache_data(show_spinner=False)
def scene_thumbnail(name: str, width: int = 320) -> np.ndarray:
    """A small preview so the scene picker shows lots rather than filenames."""
    image = Image.open(C.DEMO_DIR / name).convert("RGB")
    height = max(1, round(image.height * width / image.width))
    return np.array(image.resize((width, height), Image.LANCZOS))


# ==========================================================================
# Plain-language helpers
#
# The brief's audience is explicitly non-technical. These turn the numbers the
# model produces into the words a driver would use.
# ==========================================================================
def scene_mood(occupancy_pct: float) -> str:
    if occupancy_pct >= 95:
        return "Packed"
    if occupancy_pct >= 75:
        return "Busy"
    if occupancy_pct >= 40:
        return "Filling up"
    if occupancy_pct > 5:
        return "Quiet"
    return "Empty"


def time_of_day(clock: str) -> str:
    hour = int(clock[:2])
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    return "evening"


def scene_headline(entry: dict) -> str:
    """'Packed · cloudy morning' rather than a row of raw fields."""
    return (f"{scene_mood(entry['occupancy_pct_truth'])} · "
            f"{entry['weather']} {time_of_day(entry['time'])}")


def scene_detail(entry: dict) -> str:
    return (f"{entry['lot'].upper()} · {entry['date']} {entry['time'][:5]} · "
            f"{entry['n_spaces']} bays")


def ci_text(test: dict) -> str:
    """
    The 95% interval on test accuracy, e.g. '99.59–100.00%'.

    Shown next to every accuracy figure: 0 errors in 900 prints as 100.00%, but
    900 samples can only bound the true accuracy from below, not prove it.
    """
    ci = test.get("accuracy_ci95")
    if not ci:
        return ""
    return f"{ci[0] * 100:.2f}–{ci[1] * 100:.2f}%"


def confidence_sentence(insight: I.ParkingInsight, predictions) -> str:
    total = insight.total_slots
    decisive = int(sum(1 for p in predictions if p.confidence > 0.90))
    if insight.uncertain_slots == 0:
        return f"The model was confident about all {total} bays."
    return (f"The model was confident about {decisive} of {total} bays and "
            f"hesitated on {insight.uncertain_slots}.")


# ==========================================================================
# Colour
#
# The congestion colours in src/config.py are 600-weight shades, chosen as
# solid fills behind white text. As *text* on this dark theme the red manages
# only 3.96:1 (3.60:1 on a card), under the 4.5:1 WCAG AA minimum. Wherever a
# status colour is the foreground, the app uses the brighter 500-weight shade
# instead — red then reaches 5.08:1. Small labels use the 400 weight (6.9:1).
# ==========================================================================
ACCENT = {
    "#16A34A": "#22C55E",   # green
    "#D97706": "#F59E0B",   # amber
    "#DC2626": "#EF4444",   # red
    "#6B7280": "#9CA3AF",   # grey (no bays)
}
TEXT_EMPTY, TEXT_OCCUPIED = "#4ADE80", "#F87171"     # 400 weight, for small labels
CHART_EMPTY, CHART_OCCUPIED = "#22C55E", "#EF4444"
CHART_BLUE = "#60A5FA"


def rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def accent_vars(colour: str) -> str:
    """The four custom properties the CSS reads a status colour from."""
    accent = ACCENT.get(colour.upper(), colour)
    return (f"--pv-accent:{accent};--pv-tint:{rgba(accent, .14)};"
            f"--pv-edge:{rgba(accent, .45)};--pv-glow:{rgba(accent, .55)};")


# ==========================================================================
# Render helpers
# ==========================================================================
def metric_card(label: str, value, sub: str = "", colour: str | None = None) -> str:
    style = f' style="{accent_vars(colour)}"' if colour else ""
    return (
        f'<div class="pv-metric"{style}><div class="pv-label">{label}</div>'
        f'<div class="pv-value">{value}</div>'
        f'<div class="pv-sub">{sub}</div></div>'
    )


def verdict_card(insight: I.ParkingInsight, predictions, timing: dict,
                 zone_line: str) -> str:
    """
    The answer, before anything technical.

    A driver wants one number and one instruction. Everything else on the page
    exists to justify these two things, so they lead.
    """
    free = insight.available_slots
    return (
        f'<div class="pv-verdict" style="{accent_vars(insight.colour)}">'
        f'  <div class="pv-verdict-num">'
        f'    <div class="pv-big">{free}</div>'
        f'    <div class="pv-big-label">{"bay free" if free == 1 else "bays free"}</div>'
        f'    <div class="pv-big-sub">of {insight.total_slots} · '
        f'{insight.occupancy_pct:.0f}% occupied</div>'
        f'  </div>'
        f'  <div class="pv-verdict-body">'
        f'    <span class="pv-pill">'
        f'{insight.emoji} {insight.congestion_level} congestion</span>'
        f'    <h3>{insight.recommendation}</h3>'
        f'    <p>{zone_line}</p>'
        f'    <p class="pv-verdict-meta">{timing["n_slots"]} bays checked in '
        f'{timing["total_seconds"]:.2f} s · {confidence_sentence(insight, predictions)}</p>'
        f'  </div>'
        f'</div>'
    )


def congestion_banner(insight: I.ParkingInsight) -> str:
    return (
        f'<div class="pv-banner" style="{accent_vars(insight.colour)}">'
        f'<h3>{insight.emoji} {insight.congestion_level} congestion — {insight.headline}</h3>'
        f'<p>{insight.detail}</p>'
        f'<div class="pv-rec">Recommendation: {insight.recommendation}</div>'
        f'<div class="pv-bar">'
        f'<div style="width:{min(100.0, insight.occupancy_pct):.1f}%;"></div></div>'
        f'<p style="margin-top:.4rem;font-size:.82rem;opacity:.7;">'
        f'{insight.occupancy_pct:.1f}% of bays occupied</p>'
        f"</div>"
    )


def to_png_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def analyse(image_rgb: np.ndarray, spaces, model, threshold: float,
            overlay_opts: dict):
    """Run the whole pipeline for one photograph."""
    predictions, timing = INF.predict_slots(
        image_rgb, spaces, model=model, threshold=threshold
    )
    insight = I.generate_insight(predictions)
    annotated = O.annotate(image_rgb, predictions, insight, **overlay_opts)
    return predictions, timing, insight, annotated


def zone_sentence(predictions) -> str:
    """
    Where in the lot to actually drive — not just how many bays are free.

    Returns HTML, not Markdown: this goes inside the verdict card, which is
    injected as raw HTML, and Markdown emphasis is not interpreted there.
    """
    zones = I.row_summary(predictions)
    if not zones:
        return "Bay-level results are shown below."
    target = I.best_zone(zones)
    if not target:
        return "No zone in this lot has a free bay right now."
    return (f"Head for <b>{target['zone']}</b> — {target['position'].lower()}. "
            f"{target['available']} of {target['total']} bays are free there.")


# ==========================================================================
# Header
# ==========================================================================
st.markdown(
    '<div class="pv-hero">'
    "<h1>🅿️ ParkVision AI</h1>"
    "<p>Intelligent Urban Parking Analytics &amp; Space Optimisation Platform "
    "— UrbanFlow AI Pvt. Ltd.</p>"
    '<span class="pv-tag"><span class="pv-live"></span>Slot-level occupancy detection · '
    "MobileNetV2 transfer learning · Real-time availability insights</span>"
    "</div>",
    unsafe_allow_html=True,
)

metrics = get_metrics()
demo_index = get_demo_index()
library = get_layouts()

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
            + (f"95% interval {ci_text(test)}  \n" if ci_text(test) else "")
            + f"ROC AUC **{test.get('roc_auc', 0):.4f}**  \n"
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
    st.caption(
        "Congestion bands  \n"
        f"Low < {C.CONGESTION_LOW_MAX:.0f}% · "
        f"Moderate {C.CONGESTION_LOW_MAX:.0f}–{C.CONGESTION_MODERATE_MAX:.0f}% · "
        f"High > {C.CONGESTION_MODERATE_MAX:.0f}%"
    )

OVERLAY_OPTS = dict(
    show_ids=show_ids,
    show_confidence=show_conf,
    mark_uncertain=mark_uncertain,
    legend=show_legend,
)

# ==========================================================================
# Mode
# ==========================================================================
mode = st.segmented_control(
    "Mode",
    ["🔍 Analyse a lot", "⚖️ Compare two moments"],
    default="🔍 Analyse a lot",
    label_visibility="collapsed",
)
mode = mode or "🔍 Analyse a lot"

if "scene_idx" not in st.session_state:
    st.session_state.scene_idx = None


def pick_scene(index: int) -> None:
    st.session_state.scene_idx = index


# ==========================================================================
# COMPARE MODE — the same lot at two different moments
# ==========================================================================
if mode == "⚖️ Compare two moments":
    st.subheader("⚖️ Compare two moments")
    st.caption(
        "The same camera, two different times or weather conditions. This is "
        "where a fixed-camera system earns its keep: the layout is identical, "
        "so any difference in the counts is a real change in the lot."
    )

    if not demo_index:
        st.warning("No demo scenes found. Run `python scripts/01_build_dataset.py` first.")
        st.stop()

    lots = sorted({e["lot"] for e in demo_index})
    counts = {lot: sum(1 for e in demo_index if e["lot"] == lot) for lot in lots}
    comparable = [lot for lot in lots if counts[lot] >= 2]
    if not comparable:
        st.info("Comparison needs at least two scenes from one camera.")
        st.stop()

    lot = st.selectbox(
        "Camera", comparable,
        format_func=lambda x: f"{x.upper()} — {counts[x]} scenes",
    )
    scenes = [e for e in demo_index if e["lot"] == lot]
    scenes.sort(key=lambda e: (e["date"], e["time"]))
    labels = [f"{e['date']} {e['time'][:5]} · {scene_headline(e)}" for e in scenes]

    col_a, col_b = st.columns(2)
    idx_a = col_a.selectbox("Moment A", range(len(scenes)),
                            format_func=lambda i: labels[i], index=0)
    idx_b = col_b.selectbox("Moment B", range(len(scenes)),
                            format_func=lambda i: labels[i], index=len(scenes) - 1)

    if not C.MODEL_PATH.exists():
        st.error(f"No trained model at `{C.MODEL_PATH.name}`. "
                 "Run `python scripts/02_train.py` first.")
        st.stop()

    model = get_model()
    results = []
    with st.spinner("Analysing both moments…"):
        for idx in (idx_a, idx_b):
            entry = scenes[idx]
            image = load_demo_image(entry["image"])
            results.append((entry,) + analyse(image, entry["spaces"], model,
                                              threshold, OVERLAY_OPTS) + (image,))

    for column, (entry, preds, timing, insight, annotated, _img) in zip(
            (col_a, col_b), results):
        with column:
            st.markdown(
                f"**{entry['date']} {entry['time'][:5]}** · {entry['weather']} · "
                f"{scene_mood(insight.occupancy_pct)}"
            )
            st.image(annotated, width="stretch")
            m1, m2, m3 = st.columns(3)
            m1.metric("Free", insight.available_slots)
            m2.metric("Occupied", insight.occupied_slots)
            m3.metric("Occupancy", f"{insight.occupancy_pct:.0f}%")
            st.markdown(
                f'<span class="pv-pill" style="{accent_vars(insight.colour)}">'
                f'{insight.emoji} {insight.congestion_level}</span>',
                unsafe_allow_html=True,
            )

    entry_a, preds_a, _ta, ins_a, _aa, _ia = results[0]
    entry_b, preds_b, _tb, ins_b, _ab, _ib = results[1]
    delta = ins_b.available_slots - ins_a.available_slots

    st.divider()
    if idx_a == idx_b:
        st.info("Pick two different moments to see the change.")
    elif delta == 0:
        st.info(
            f"**No net change.** Both moments have {ins_a.available_slots} free "
            f"{'bay' if ins_a.available_slots == 1 else 'bays'} — even though the "
            f"conditions differ ({entry_a['weather']} → {entry_b['weather']})."
        )
    else:
        direction = "freed up" if delta > 0 else "filled up"
        st.success(
            f"**The lot {direction}.** Free bays went from "
            f"**{ins_a.available_slots}** at {entry_a['time'][:5]} to "
            f"**{ins_b.available_slots}** at {entry_b['time'][:5]} — a change of "
            f"**{abs(delta)}** {'bay' if abs(delta) == 1 else 'bays'}, "
            f"and the congestion level moved from *{ins_a.congestion_level}* to "
            f"*{ins_b.congestion_level}*."
        )

    trend = pd.DataFrame(
        {
            "Moment": [f"A · {entry_a['time'][:5]}", f"B · {entry_b['time'][:5]}"],
            "Available": [ins_a.available_slots, ins_b.available_slots],
            "Occupied": [ins_a.occupied_slots, ins_b.occupied_slots],
        }
    ).set_index("Moment")
    st.bar_chart(trend, color=[CHART_EMPTY, CHART_OCCUPIED])
    st.stop()

# ==========================================================================
# ANALYSE MODE — input
# ==========================================================================
st.subheader("1 · Pick a parking lot")

image_rgb = None
spaces = None
scene_meta = None
geometry_source = ""

tab_demo, tab_upload = st.tabs(["📁 Demo lots (held-out days)", "⬆️ Upload your own"])

with tab_demo:
    if not demo_index:
        st.warning("No demo scenes found. Run `python scripts/01_build_dataset.py` first.")
    else:
        lots = sorted({e["lot"] for e in demo_index})
        lot = st.segmented_control(
            "Camera", lots, default=lots[0],
            format_func=lambda x: f"📷 {x.upper()}",
            label_visibility="collapsed",
        ) or lots[0]

        in_lot = [(i, e) for i, e in enumerate(demo_index) if e["lot"] == lot]

        st.caption(
            "Every scene below comes from a calendar day the model never trained "
            "on. Pick one to run the full pipeline on it."
        )

        # A strip of thumbnails reads far faster than a dropdown of filenames,
        # and it shows the variety of conditions the model has to survive.
        for row_start in range(0, len(in_lot), 4):
            row = in_lot[row_start:row_start + 4]
            columns = st.columns(4)
            for column, (index, entry) in zip(columns, row):
                with column:
                    st.image(scene_thumbnail(entry["image"]), width="stretch")
                    st.markdown(
                        f'<div class="pv-crop-cap"><b>{scene_headline(entry)}</b>'
                        f'{scene_detail(entry)}</div>',
                        unsafe_allow_html=True,
                    )
                    st.button(
                        "Analyse this" if st.session_state.scene_idx != index
                        else "✓ Selected",
                        key=f"pick_{index}",
                        width="stretch",
                        on_click=pick_scene,
                        args=(index,),
                        type="primary" if st.session_state.scene_idx == index
                        else "secondary",
                    )

        # A chosen scene stays chosen even if the camera tab is switched, so
        # browsing the other cameras never silently discards the analysis.
        if st.session_state.scene_idx is not None:
            entry = demo_index[st.session_state.scene_idx]
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
        geometry_mode = st.radio("How should bays be located?", mode_options,
                                 index=default_mode, horizontal=True)

        if geometry_mode == "Auto / surveyed camera" and matched is not None:
            spaces = matched.spaces
            geometry_source = f"Auto-matched surveyed layout — {matched.label}"
        elif geometry_mode == "Choose a camera" and library:
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
# Landing state — what this is, before anything is selected
# ==========================================================================
if image_rgb is None:
    test = metrics.get("test", {})
    accuracy = f"{test.get('accuracy', 0) * 100:.2f}%" if test else "—"

    st.markdown(
        '<div class="pv-land">'
        "<h2>Find out how many bays are free — from one photograph</h2>"
        "<p>Drivers circling a busy lot do not have a parking problem. They have "
        "an <b>information</b> problem: the space usually exists, they just cannot "
        "see it from the entrance. ParkVision reads a photograph of the lot and "
        "answers bay by bay.</p>"
        '<div class="pv-flow">'
        "<div class=\"pv-step\"><b>1 · Every bay is cut out</b><span>Each surveyed "
        "bay is perspective-corrected into its own 224×224 image.</span></div>"
        "<div class=\"pv-step\"><b>2 · Each one is classified</b><span>A fine-tuned "
        "MobileNetV2 decides occupied or empty, and says how sure it is.</span></div>"
        "<div class=\"pv-step\"><b>3 · Counts become advice</b><span>Free bays, "
        "congestion level, which zone to drive to, and whether to bother.</span></div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.markdown(metric_card("Accuracy", accuracy,
                            f"{test.get('errors', 0)} errors in {test.get('n_samples', 0)} "
                            "unseen bays"
                            + (f" · 95% CI {ci_text(test)}" if ci_text(test) else ""),
                            colour=CHART_EMPTY),
                unsafe_allow_html=True)
    c2.markdown(metric_card("Demo lots", len(demo_index), "real held-out scenes",
                            colour=CHART_BLUE),
                unsafe_allow_html=True)
    speed = measured_ms_per_bay()
    c3.markdown(metric_card("Speed", f"{speed:.0f} ms" if speed else "—",
                            "per bay end to end, CPU only (measured)",
                            colour=CHART_BLUE),
                unsafe_allow_html=True)

    if demo_index:
        st.write("")
        st.markdown("**Not sure where to start?**")
        q1, q2, q3 = st.columns(3)

        busiest = max(range(len(demo_index)),
                      key=lambda i: demo_index[i]["occupancy_pct_truth"])
        emptiest = min(range(len(demo_index)),
                       key=lambda i: demo_index[i]["occupancy_pct_truth"])
        q1.button("🚗 Show me a packed lot", width="stretch",
                  on_click=pick_scene, args=(busiest,))
        q2.button("🅿️ Show me an empty one", width="stretch",
                  on_click=pick_scene, args=(emptiest,))
        q3.button("🎲 Surprise me", width="stretch",
                  on_click=pick_scene, args=(random.randrange(len(demo_index)),))

        st.caption("Or open **📁 Demo lots** above to browse every scene.")
    st.stop()

if not C.MODEL_PATH.exists():
    st.error(
        f"No trained model at `{C.MODEL_PATH.name}`. "
        "Run `python scripts/02_train.py` before starting the app."
    )
    st.stop()

# ==========================================================================
# Analysis
# ==========================================================================
model = get_model()

with st.spinner("Analysing every parking bay…"):
    predictions, timing, insight, annotated = analyse(
        image_rgb, spaces, model, threshold, OVERLAY_OPTS
    )

if not predictions:
    st.error(
        "No parking bays could be read from this image. The bay layout may not "
        "line up with the photograph — try selecting a camera manually, or use "
        "grid mode."
    )
    st.stop()

# ---------------------------------------------------------------- the answer
st.subheader("2 · The verdict")
st.markdown(
    verdict_card(insight, predictions, timing, zone_sentence(predictions)),
    unsafe_allow_html=True,
)

for note in insight.notes:
    st.markdown(f'<div class="pv-note">ℹ️ {note}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------- what it saw
st.subheader("3 · What the camera saw")
st.caption(f"{geometry_source} · {timing['n_slots']} bays analysed in "
           f"{timing['total_seconds']:.2f}s ({timing['ms_per_slot']:.1f} ms per bay)")

view_annotated, view_original, view_both = st.tabs(
    ["🎯 Annotated", "🖼️ Original", "🔀 Side by side"]
)
with view_annotated:
    st.image(annotated, width="stretch")
    st.caption("🟩 green — empty · 🟥 red — occupied"
               + (" · 🟨 amber — the model was unsure" if mark_uncertain else ""))
with view_original:
    st.image(image_rgb, width="stretch")
with view_both:
    left, right = st.columns(2)
    left.markdown("**Original**")
    left.image(image_rgb, width="stretch")
    right.markdown("**Annotated**")
    right.image(annotated, width="stretch")

st.download_button(
    "⬇️ Download annotated image (PNG)",
    data=to_png_bytes(annotated),
    file_name="parkvision_annotated.png",
    mime="image/png",
)

# ==========================================================================
# Metrics + insight
# ==========================================================================
st.subheader("4 · Availability in detail")

c1, c2, c3, c4 = st.columns(4)
c1.markdown(metric_card("Total slots", insight.total_slots, "bays in this lot",
                        colour=CHART_BLUE),
            unsafe_allow_html=True)
c2.markdown(metric_card("Occupied", insight.occupied_slots,
                        f"{insight.occupancy_pct:.1f}% of the lot",
                        colour=CHART_OCCUPIED),
            unsafe_allow_html=True)
c3.markdown(metric_card("Available", insight.available_slots,
                        "ready to park in now", colour=CHART_EMPTY),
            unsafe_allow_html=True)
c4.markdown(metric_card("Congestion", insight.congestion_level,
                        f"avg confidence {insight.mean_confidence * 100:.1f}%",
                        colour=insight.colour),
            unsafe_allow_html=True)

st.markdown(congestion_banner(insight), unsafe_allow_html=True)

# ==========================================================================
# Deeper analytics
# ==========================================================================
st.subheader("5 · Explore the analysis")

zones = I.row_summary(predictions)
tab_zone, tab_see, tab_conf, tab_table, tab_truth = st.tabs(
    ["🧭 Where to park", "🔬 See what the AI sees", "📊 Confidence",
     "📋 Bay-by-bay", "✅ Accuracy on this scene"]
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
        st.dataframe(zone_df, width="stretch", hide_index=True)
        st.bar_chart(zone_df.set_index("Zone")[["Available", "Occupied"]],
                     color=[CHART_EMPTY, CHART_OCCUPIED])
    else:
        st.caption("Zone analysis needs bay geometry with at least three rows.")

with tab_see:
    st.caption(
        "The model never sees the whole photograph. It sees one deskewed "
        "224×224 crop per bay — exactly the images below. This is the input "
        "behind every green and red box above."
    )

    patches, kept = INF.crops_from_spaces(image_rgb, spaces)
    index_of = {int(space.get("space_id", i + 1)): i
                for i, space in enumerate(kept)}

    focus = st.radio(
        "Show me",
        ["Where the model hesitated", "Bays it called occupied",
         "Bays it called empty", "Pick a bay myself"],
        horizontal=True,
    )

    if focus == "Pick a bay myself":
        chosen = st.selectbox(
            "Bay", [p.space_id for p in predictions],
            format_func=lambda b: f"Bay {b}",
        )
        shortlist = [p for p in predictions if p.space_id == chosen]
    elif focus == "Bays it called occupied":
        shortlist = sorted([p for p in predictions if p.status == "occupied"],
                           key=lambda p: -p.confidence)[:8]
    elif focus == "Bays it called empty":
        shortlist = sorted([p for p in predictions if p.status == "empty"],
                           key=lambda p: -p.confidence)[:8]
    else:
        shortlist = sorted(predictions, key=lambda p: p.confidence)[:8]
        if shortlist and shortlist[0].confidence > 0.90:
            st.info(
                "The model was decisive everywhere on this scene — the least "
                "confident bay still scored "
                f"{shortlist[0].confidence * 100:.1f}%. Shown below anyway."
            )

    if not shortlist:
        st.info("No bays in this category.")
    else:
        for row_start in range(0, len(shortlist), 4):
            row = shortlist[row_start:row_start + 4]
            columns = st.columns(4)
            for column, prediction in zip(columns, row):
                position = index_of.get(prediction.space_id)
                if position is None or position >= len(patches):
                    continue
                with column:
                    st.image(patches[position].astype(np.uint8), width="stretch")
                    tint = TEXT_OCCUPIED if prediction.status == "occupied" else TEXT_EMPTY
                    flag = " · unsure" if prediction.uncertain else ""
                    st.markdown(
                        f'<div class="pv-crop-cap">'
                        f'<b style="color:{tint};">Bay {prediction.space_id} — '
                        f'{prediction.status}</b>'
                        f'{prediction.confidence * 100:.1f}% confident{flag}<br>'
                        f'P(occupied) = {prediction.score:.3f}</div>',
                        unsafe_allow_html=True,
                    )

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
    st.dataframe(table, width="stretch", hide_index=True, height=340)
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
            st.dataframe(comparison, width="stretch", hide_index=True)
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
        m1.metric("Accuracy", f"{test['accuracy'] * 100:.2f}%",
                  help=("95% exact (Clopper–Pearson) interval: " + ci_text(test)
                        + ". The test split has a fixed size, so a perfect score "
                        "bounds the true accuracy from below rather than proving it.")
                  if ci_text(test) else None)
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
                width="stretch", hide_index=True,
            )
        if metrics.get("by_lot"):
            st.markdown("**Accuracy by parking lot**")
            st.dataframe(
                pd.DataFrame(metrics["by_lot"]).T.reset_index().rename(
                    columns={"index": "Lot", "n": "Bays",
                             "accuracy": "Accuracy", "share_occupied": "Share occupied"}
                ),
                width="stretch", hide_index=True,
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
