"""
Evaluate the ParkVision AI classifier.

Produces the evidence the brief asks for (accuracy, confusion matrix) plus the
breakdowns that show *where* the model is weak rather than only how strong it
is on average.

Usage
-----
    python scripts/03_evaluate.py

Writes
------
    results/metrics.json                 headline + per-slice metrics
    results/classification_report.txt    precision/recall/F1 per class
    results/confusion_matrix.png         counts and row-normalised percentages
    results/roc_pr_curves.png            ROC and precision-recall curves
    results/threshold_sweep.png          why the operating threshold was chosen
    results/misclassified_examples.png   the errors, so they can be discussed
"""

from __future__ import annotations

import gc
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config as C
from src import dataset as DS


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_confusion(cm: np.ndarray, out_path: Path, title: str) -> None:
    normalised = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    for ax, matrix, fmt, subtitle, cmap in (
        (axes[0], cm, "{:,}", "counts", "Blues"),
        (axes[1], normalised, "{:.1%}", "row-normalised (recall per class)", "Greens"),
    ):
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=matrix.max() or 1)
        ax.set_xticks(range(len(C.CLASS_NAMES)), [c.title() for c in C.CLASS_NAMES])
        ax.set_yticks(range(len(C.CLASS_NAMES)), [c.title() for c in C.CLASS_NAMES])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(subtitle, fontsize=10)
        threshold = (matrix.max() or 1) / 2.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j, i, fmt.format(matrix[i, j]),
                    ha="center", va="center", fontsize=13,
                    color="white" if matrix[i, j] > threshold else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_curves(y_true: np.ndarray, scores: np.ndarray, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot(fpr, tpr, linewidth=2, label=f"AUC = {auc(fpr, tpr):.4f}")
    axes[0].plot([0, 1], [0, 1], "--", color="#999", linewidth=1, label="chance")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve — test split")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(recall, precision, linewidth=2,
                 label=f"AP = {average_precision_score(y_true, scores):.4f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall curve — test split")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.suptitle("Discrimination between occupied and empty bays", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_sweep(y_true, scores, chosen: float, out_path: Path) -> None:
    grid = np.linspace(0.05, 0.95, 91)
    accuracy = [accuracy_score(y_true, (scores >= t).astype(int)) for t in grid]
    f1 = [f1_score(y_true, (scores >= t).astype(int), zero_division=0) for t in grid]

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(grid, accuracy, linewidth=2, label="accuracy")
    ax.plot(grid, f1, linewidth=2, label="F1 (occupied)")
    ax.axvline(chosen, color="#DC2626", linestyle="--", linewidth=1.5,
               label=f"chosen threshold = {chosen:.2f}")
    ax.axvline(0.5, color="#999", linestyle=":", linewidth=1, label="default 0.50")
    ax.set_xlabel("Decision threshold on P(occupied)")
    ax.set_ylabel("Score")
    ax.set_title("Threshold selected on validation, applied unchanged to test")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_misclassified(files, y_true, y_pred, scores, out_path: Path, limit: int = 12) -> None:
    import cv2

    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        fig, ax = plt.subplots(figsize=(7, 2))
        ax.text(0.5, 0.5, "No misclassified test crops", ha="center", va="center", fontsize=14)
        ax.axis("off")
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return

    # Show the most confident mistakes first: those are the informative ones.
    wrong = sorted(wrong, key=lambda i: abs(scores[i] - 0.5), reverse=True)[:limit]

    cols = 6
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, index in zip(axes, wrong):
        image = cv2.cvtColor(cv2.imread(files[index]), cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.set_title(
            f"true: {C.CLASS_NAMES[y_true[index]]}\n"
            f"pred: {C.CLASS_NAMES[y_pred[index]]} ({scores[index]:.2f})",
            fontsize=8,
        )
        ax.axis("off")
    for ax in axes[len(wrong):]:
        ax.axis("off")

    fig.suptitle(f"Most confident errors ({len(np.where(y_true != y_pred)[0])} total on test)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Slice analysis
# --------------------------------------------------------------------------
def load_manifest_lookup() -> dict:
    path = C.PROCESSED_DIR / "manifest.json"
    if not path.exists():
        return {}
    records = json.loads(path.read_text(encoding="utf-8"))
    return {Path(r["file"]).name: r for r in records}


def clopper_pearson(correct: int, n: int, level: float = 0.95) -> tuple:
    """
    Exact binomial confidence interval for an accuracy of ``correct`` / ``n``.

    With errors in single digits, a bare accuracy overstates what the test set
    can prove: 0 errors in 900 prints as 100.00%, but 900 samples cannot rule
    out a true error rate of a few tenths of a percent. The usual normal
    approximation is useless this close to 1 (its interval can exceed 100%), so
    this uses the beta distribution directly.
    """
    from scipy.stats import beta
    alpha = 1.0 - level
    low = 0.0 if correct == 0 else float(beta.ppf(alpha / 2, correct, n - correct + 1))
    high = 1.0 if correct == n else float(beta.ppf(1 - alpha / 2, correct + 1, n - correct))
    return low, high


def slice_metrics(files, y_true, y_pred, lookup, key: str) -> dict:
    """Accuracy broken down by a manifest field (weather, lot, ...)."""
    buckets = defaultdict(lambda: {"n": 0, "correct": 0, "occupied": 0})
    for path, truth, pred in zip(files, y_true, y_pred):
        record = lookup.get(Path(path).name)
        if not record:
            continue
        bucket = buckets[record[key]]
        bucket["n"] += 1
        bucket["correct"] += int(truth == pred)
        bucket["occupied"] += int(truth == 1)

    return {
        name: {
            "n": b["n"],
            "accuracy": round(b["correct"] / b["n"], 4),
            "share_occupied": round(b["occupied"] / b["n"], 4),
        }
        for name, b in sorted(buckets.items())
        if b["n"] > 0
    }


def plot_slices(slices: dict, out_path: Path) -> None:
    panels = [(k, v) for k, v in slices.items() if v]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.2))
    axes = np.atleast_1d(axes)

    for ax, (name, data) in zip(axes, panels):
        labels = list(data)
        values = [data[k]["accuracy"] for k in labels]
        counts = [data[k]["n"] for k in labels]
        bars = ax.bar(labels, values, color="#2563EB", width=0.55)
        for bar, value, count in zip(bars, values, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008,
                    f"{value:.3f}\nn={count}", ha="center", fontsize=8)
        ax.set_ylim(min(0.90, min(values) - 0.04), 1.005)
        ax.set_ylabel("Test accuracy")
        ax.set_title(f"Accuracy by {name}")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Where the model is strong and where it is weak", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    C.ensure_dirs()
    if not C.MODEL_PATH.exists():
        raise SystemExit(f"No model at {C.MODEL_PATH}. Run scripts/02_train.py first.")

    print(f"loading {C.MODEL_PATH}")
    # Through the shared loader, which registers the custom noise layer.
    from src import inference as INF
    model = INF.load_model(C.MODEL_PATH)

    print("scoring the validation split (threshold selection)")
    val_x, val_y, _ = DS.unbatched_arrays("val")
    val_scores = model.predict(val_x, batch_size=64, verbose=0).ravel()

    # Choose the threshold that maximises accuracy on validation, then freeze it.
    # Tuning on test and reporting on test would be reporting a number we fitted.
    grid = np.linspace(0.05, 0.95, 91)
    val_accuracy = [accuracy_score(val_y, (val_scores >= t).astype(int)) for t in grid]
    best_index = int(np.argmax(val_accuracy))
    threshold = float(round(grid[best_index], 2))
    print(f"  validation accuracy @0.50 = {accuracy_score(val_y, (val_scores >= 0.5).astype(int)):.4f}")
    print(f"  best validation accuracy  = {val_accuracy[best_index]:.4f} @ threshold {threshold:.2f}")

    # Free the validation images before loading the test ones: each split is
    # ~540 MB as float32, and holding both at once is needless pressure.
    del val_x
    gc.collect()

    print("scoring the test split")
    test_x, test_y, test_files = DS.unbatched_arrays("test")
    test_scores = model.predict(test_x, batch_size=64, verbose=0).ravel()
    test_pred = (test_scores >= threshold).astype(int)

    cm = confusion_matrix(test_y, test_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    report_text = classification_report(
        test_y, test_pred, target_names=[c.title() for c in C.CLASS_NAMES], digits=4
    )
    (C.RESULTS_DIR / "classification_report.txt").write_text(
        "ParkVision AI — test split classification report\n"
        f"decision threshold on P(occupied): {threshold:.2f}\n\n" + report_text,
        encoding="utf-8",
    )

    lookup = load_manifest_lookup()
    slices = {
        "weather": slice_metrics(test_files, test_y, test_pred, lookup, "weather"),
        "lot": slice_metrics(test_files, test_y, test_pred, lookup, "lot"),
    }

    metrics = {
        "operating_threshold": threshold,
        "test": {
            "accuracy": round(float(accuracy_score(test_y, test_pred)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(test_y, test_pred)), 4),
            "precision_occupied": round(float(precision_score(test_y, test_pred, zero_division=0)), 4),
            "recall_occupied": round(float(recall_score(test_y, test_pred, zero_division=0)), 4),
            "f1_occupied": round(float(f1_score(test_y, test_pred, zero_division=0)), 4),
            "precision_empty": round(float(precision_score(test_y, test_pred, pos_label=0, zero_division=0)), 4),
            "recall_empty": round(float(recall_score(test_y, test_pred, pos_label=0, zero_division=0)), 4),
            "f1_empty": round(float(f1_score(test_y, test_pred, pos_label=0, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(test_y, test_scores)), 4),
            "average_precision": round(float(average_precision_score(test_y, test_scores)), 4),
            "matthews_corrcoef": round(float(matthews_corrcoef(test_y, test_pred)), 4),
            "n_samples": int(len(test_y)),
            "confusion_matrix": {
                "true_empty_pred_empty": int(tn),
                "true_empty_pred_occupied": int(fp),
                "true_occupied_pred_empty": int(fn),
                "true_occupied_pred_occupied": int(tp),
            },
            "errors": int(fp + fn),
            "accuracy_ci95": [
                round(v, 4) for v in clopper_pearson(int(tn + tp), int(len(test_y)))
            ],
        },
        "validation": {
            "accuracy_at_default_threshold": round(
                float(accuracy_score(val_y, (val_scores >= 0.5).astype(int))), 4
            ),
            "accuracy_at_operating_threshold": round(float(val_accuracy[best_index]), 4),
            "roc_auc": round(float(roc_auc_score(val_y, val_scores)), 4),
            "n_samples": int(len(val_y)),
        },
        "by_weather": slices["weather"],
        "by_lot": slices["lot"],
        "class_names": list(C.CLASS_NAMES),
        "note": (
            "Threshold selected on the validation split only. Train/val/test are "
            "split by whole (lot, date) groups, so no calendar day appears in more "
            "than one split."
        ),
    }
    C.METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plot_confusion(cm, C.RESULTS_DIR / "confusion_matrix.png",
                   f"ParkVision AI — test confusion matrix (threshold {threshold:.2f})")
    plot_curves(test_y, test_scores, C.RESULTS_DIR / "roc_pr_curves.png")
    plot_threshold_sweep(val_y, val_scores, threshold, C.RESULTS_DIR / "threshold_sweep.png")
    plot_misclassified(test_files, test_y, test_pred, test_scores,
                       C.RESULTS_DIR / "misclassified_examples.png")
    plot_slices(slices, C.RESULTS_DIR / "accuracy_by_slice.png")

    print("\n=== test-set results ===")
    print(report_text)
    print(f"  accuracy          {metrics['test']['accuracy']:.4f}")
    print(f"  ROC AUC           {metrics['test']['roc_auc']:.4f}")
    print(f"  errors            {metrics['test']['errors']} / {len(test_y)}")
    print(f"  confusion matrix  [[{tn} {fp}] [{fn} {tp}]]  (rows = actual empty/occupied)")
    print("\n  accuracy by weather:")
    for name, row in slices["weather"].items():
        print(f"    {name:<9} {row['accuracy']:.4f}  (n={row['n']})")
    print("  accuracy by lot:")
    for name, row in slices["lot"].items():
        print(f"    {name:<9} {row['accuracy']:.4f}  (n={row['n']})")
    print(f"\nwrote {C.METRICS_PATH}")


if __name__ == "__main__":
    main()
