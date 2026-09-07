"""
Train the ParkVision AI occupancy classifier.

Two-phase transfer learning on MobileNetV2:

    phase 1  frozen backbone, head only          12 epochs @ lr 1e-3
    phase 2  top blocks unfrozen, fine-tuning    10 epochs @ lr 1e-5
                                                 --------------------
                                                 22 epochs total

Usage
-----
    python scripts/02_train.py [--epochs-head N] [--epochs-finetune N]

Writes
------
    models/parkvision_mobilenetv2.keras
    models/class_indices.json
    results/training_history.json
    results/training_curves.png
    results/augmentation_examples.png
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf
from tensorflow import keras

from src import config as C
from src import dataset as DS
from src import model as M


def set_seeds(seed: int = C.RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


STATE_PATH = C.RESULTS_DIR / "training_state.json"


def load_history() -> dict:
    """Read the epoch-by-epoch history accumulated by previous runs."""
    if C.HISTORY_PATH.exists():
        return json.loads(C.HISTORY_PATH.read_text(encoding="utf-8"))
    return {}


def append_history(history: dict, new: dict) -> dict:
    """
    Concatenate a Keras history onto the running one.

    Any metric missing from one of the two runs is padded so that every series
    stays the same length as ``loss`` and the training curves line up.
    """
    merged = {k: list(v) for k, v in history.items()}
    length = len(merged.get("loss", []))
    for key, values in new.items():
        series = merged.setdefault(key, [float("nan")] * length)
        series.extend(float(v) for v in values)
    target = len(merged.get("loss", []))
    for key, series in merged.items():
        if len(series) < target:
            series.extend([float("nan")] * (target - len(series)))
    return merged


def _read_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def save_history(history: dict, args) -> None:
    """Persist the history plus the epoch at which fine-tuning began."""
    C.HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")
    state = _read_state()
    epochs_done = len(history.get("loss", []))
    if args.phase in ("both", "head"):
        state["head_epochs"] = epochs_done
    state.setdefault("head_epochs", epochs_done)
    if args.phase == "both":
        state["head_epochs"] = args.epochs_head
    state["total_epochs"] = epochs_done
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def switch_epoch() -> int:
    """The epoch after which the backbone was unfrozen, for the curve marker."""
    return int(_read_state().get("head_epochs", C.EPOCHS_HEAD))


def save_augmentation_examples(train_ds: tf.data.Dataset, out_path: Path) -> None:
    """Render one crop next to seven augmented copies, as evidence for the report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    augmenter = M.build_augmenter()
    batch = next(iter(train_ds))[0]
    original = batch[0:1]

    fig, axes = plt.subplots(1, 8, figsize=(16, 2.4))
    axes[0].imshow(original[0].numpy().astype("uint8"))
    axes[0].set_title("original", fontsize=9)
    axes[0].axis("off")
    for i in range(1, 8):
        augmented = augmenter(original, training=True)
        axes[i].imshow(np.clip(augmented[0].numpy(), 0, 255).astype("uint8"))
        axes[i].set_title(f"aug {i}", fontsize=9)
        axes[i].axis("off")
    fig.suptitle("Training-time augmentation: flip, rotate, zoom, shift, contrast, brightness",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_history(history: dict, out_path: Path, switch_epoch: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    panels = [
        ("accuracy", "val_accuracy", "Accuracy"),
        ("loss", "val_loss", "Loss"),
        ("auc", "val_auc", "ROC AUC"),
    ]
    for ax, (train_key, val_key, title) in zip(axes, panels):
        if train_key not in history:
            ax.axis("off")
            continue
        ax.plot(epochs, history[train_key], label="train", linewidth=2)
        ax.plot(epochs, history[val_key], label="validation", linewidth=2)
        ax.axvline(switch_epoch + 0.5, color="#888", linestyle="--", linewidth=1)
        ax.annotate("fine-tuning starts", xy=(switch_epoch + 0.6, ax.get_ylim()[0]),
                    fontsize=8, color="#666")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle("ParkVision AI — MobileNetV2 two-phase transfer learning", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ParkVision AI classifier")
    parser.add_argument("--epochs-head", type=int, default=C.EPOCHS_HEAD)
    parser.add_argument("--epochs-finetune", type=int, default=C.EPOCHS_FINETUNE)
    parser.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    parser.add_argument(
        "--phase", choices=["both", "head", "finetune"], default="both",
        help=(
            "Run the whole schedule (default) or one phase at a time. Splitting "
            "the run lets a CPU-only machine train in bounded sessions: "
            "`--phase head`, then `--phase finetune --resume`."
        ),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue from the saved checkpoint and append to the existing history.",
    )
    args = parser.parse_args()

    set_seeds()
    C.ensure_dirs()
    started = time.time()

    counts = DS.split_counts()
    print("dataset:")
    for split in C.SPLITS:
        print(f"  {split:<5} {counts[split]}")
    if counts["train"]["empty"] == 0:
        raise SystemExit("No training data. Run scripts/01_build_dataset.py first.")

    train_ds = DS.load_split("train", batch_size=args.batch_size)
    val_ds = DS.load_split("val", batch_size=args.batch_size)

    history = load_history() if args.resume else {}
    if args.resume:
        if not C.MODEL_PATH.exists():
            raise SystemExit(f"--resume needs a checkpoint at {C.MODEL_PATH}.")
        print(f"\nresuming from {C.MODEL_PATH} "
              f"({len(history.get('loss', []))} epochs already recorded)")
        model = keras.models.load_model(C.MODEL_PATH)
    else:
        print("\nbuilding MobileNetV2 classifier")
        model = M.build_model()
        M.compile_for_head(model)
        print(f"  total parameters:     {model.count_params():,}")
        print(f"  trainable (phase 1):  {M.trainable_parameter_count(model):,}")
        save_augmentation_examples(train_ds, C.RESULTS_DIR / "augmentation_examples.png")

    best_so_far = max(history.get("val_accuracy", [0.0]) or [0.0])

    # ---------------------------------------------------------------- phase 1
    if args.phase in ("both", "head"):
        M.compile_for_head(model)
        callbacks_phase1 = [
            keras.callbacks.ModelCheckpoint(
                str(C.MODEL_PATH), monitor="val_accuracy", mode="max",
                save_best_only=True, verbose=0,
                initial_value_threshold=best_so_far or None,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
            ),
        ]
        done = len(history.get("loss", []))
        print(f"\n=== phase 1/2: training the head "
              f"({args.epochs_head} epochs, frozen backbone) ===")
        history1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=done + args.epochs_head,
            initial_epoch=done,
            class_weight=DS.class_weights(),
            callbacks=callbacks_phase1,
            shuffle=False,   # the tf.data pipeline already reshuffles each epoch
            verbose=2,
        )
        history = append_history(history, history1.history)
        best_so_far = max(history["val_accuracy"])
        save_history(history, args)

    # ---------------------------------------------------------------- phase 2
    if args.phase in ("both", "finetune"):
        print(f"\n=== phase 2/2: fine-tuning ({args.epochs_finetune} epochs, "
              f"backbone unfrozen from layer {C.FINE_TUNE_FROM_LAYER}) ===")
        M.compile_for_finetune(model)
        print(f"  trainable (phase 2):  {M.trainable_parameter_count(model):,}")

        # Seed phase 2's checkpoint with the best score so far. A fresh
        # ModelCheckpoint starts from -inf and would happily overwrite a better
        # phase-1 model with the first fine-tuning epoch, however much worse.
        print(f"  checkpoint must beat:  {best_so_far:.4f}")

        callbacks_phase2 = [
            keras.callbacks.ModelCheckpoint(
                str(C.MODEL_PATH), monitor="val_accuracy", mode="max",
                save_best_only=True, verbose=0,
                initial_value_threshold=best_so_far or None,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", mode="max", patience=C.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True, verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-8, verbose=1
            ),
        ]

        done = len(history.get("loss", []))
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=done + args.epochs_finetune,
            initial_epoch=done,
            class_weight=DS.class_weights(),
            callbacks=callbacks_phase2,
            shuffle=False,   # the tf.data pipeline already reshuffles each epoch
            verbose=2,
        )
        history = append_history(history, history2.history)
        save_history(history, args)

    plot_history(history, C.RESULTS_DIR / "training_curves.png", switch_epoch())

    # The checkpoint holds the best-validation weights; reload so that whatever
    # the last epoch happened to do cannot change what we ship.
    model = keras.models.load_model(C.MODEL_PATH)
    val_metrics = model.evaluate(val_ds, verbose=0, return_dict=True)

    C.CLASS_INDEX_PATH.write_text(
        json.dumps(
            {
                "class_names": list(C.CLASS_NAMES),
                "positive_class": C.POSITIVE_CLASS,
                "note": "sigmoid output is P(occupied); label 0 = empty, label 1 = occupied",
                "image_size": list(C.IMG_SIZE),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    elapsed = time.time() - started
    state = _read_state()
    cumulative_minutes = round(state.get("training_minutes", 0.0) + elapsed / 60, 2)
    state["training_minutes"] = cumulative_minutes
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")

    head_epochs = switch_epoch()
    total_epochs = len(history["loss"])

    print("\n=== training complete ===")
    print(f"  best validation accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  validation AUC:           {val_metrics['auc']:.4f}")
    print(f"  epochs run:               {total_epochs} "
          f"({head_epochs} head + {total_epochs - head_epochs} fine-tune)")
    print(f"  wall clock (this run):    {elapsed / 60:.1f} min")
    print(f"  wall clock (cumulative):  {cumulative_minutes:.1f} min")
    print(f"  model saved to:           {C.MODEL_PATH}")

    (C.RESULTS_DIR / "training_summary.json").write_text(
        json.dumps(
            {
                "backbone": "MobileNetV2 (ImageNet weights)",
                "epochs_head": head_epochs,
                "epochs_finetune": total_epochs - head_epochs,
                "epochs_run": total_epochs,
                "batch_size": args.batch_size,
                "lr_head": C.LR_HEAD,
                "lr_finetune": C.LR_FINETUNE,
                "fine_tune_from_layer": C.FINE_TUNE_FROM_LAYER,
                "dropout": C.DROPOUT,
                "label_smoothing": C.LABEL_SMOOTHING,
                "total_parameters": int(model.count_params()),
                "best_val_accuracy": float(val_metrics["accuracy"]),
                "best_val_auc": float(val_metrics["auc"]),
                "best_val_loss": float(val_metrics["loss"]),
                "training_minutes": cumulative_minutes,
                "seed": C.RANDOM_SEED,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
