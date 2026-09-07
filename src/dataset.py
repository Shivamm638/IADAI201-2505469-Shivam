"""
tf.data input pipelines over ``data/processed/``.

Images are handed to the model as raw float tensors in [0, 255]; augmentation
and MobileNetV2's rescaling live inside the model graph (see :mod:`src.model`),
so the exact same preprocessing is applied at training time, at evaluation time
and inside the Streamlit app.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from . import config as C

AUTOTUNE = tf.data.AUTOTUNE

# Images held in the training shuffle buffer. 512 crops is ~77 MB as uint8;
# spanning the whole split would be ~630 MB, which is the difference between
# training and thrashing on a machine with about a gigabyte free.
SHUFFLE_BUFFER = 512


def _label_mode() -> str:
    return "binary"


def load_split(
    split: str,
    batch_size: int = C.BATCH_SIZE,
    shuffle: bool | None = None,
    seed: int = C.RANDOM_SEED,
) -> tf.data.Dataset:
    """
    Build a batched dataset for one split.

    ``class_names`` is pinned to :data:`config.CLASS_NAMES` so that label 0 is
    always ``empty`` and label 1 always ``occupied``, independent of how the
    filesystem happens to sort the folders.
    """
    directory = C.PROCESSED_DIR / split
    if shuffle is None:
        shuffle = split == "train"

    # Shuffling happens at the *file-path* level, before any pixels are
    # decoded, and tf.data reshuffles that list every epoch. This is what makes
    # the memory-light pipeline below correct: the expensive in-memory shuffle
    # buffer would only be re-deriving an ordering we already have for free.
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode=_label_mode(),
        class_names=list(C.CLASS_NAMES),
        color_mode="rgb",
        batch_size=None,
        image_size=C.IMG_SIZE,
        shuffle=shuffle,
        seed=seed,
        interpolation="bilinear",
    )

    # Hold crops as uint8 rather than float32 wherever they are kept: a split
    # costs a quarter of the memory that way (4,200 crops is ~630 MB instead
    # of ~2.5 GB). Keras casts to the layer dtype at the model's input, so
    # handing it uint8 is safe.
    dataset = dataset.map(
        lambda image, label: (tf.cast(image, tf.uint8), label), num_parallel_calls=AUTOTUNE
    )

    if shuffle:
        # A small buffer for extra local mixing on top of the per-epoch file
        # shuffle. Deliberately small: a buffer spanning the whole split would
        # hold ~630 MB resident, and on a memory-constrained machine that
        # pushed training into paging — one epoch went from ~150 seconds to
        # over four hours with the CPU almost idle throughout.
        dataset = dataset.shuffle(
            buffer_size=SHUFFLE_BUFFER, seed=seed, reshuffle_each_iteration=True
        )
    else:
        # Validation and test are small, are read identically every epoch, and
        # must not vary between runs, so they are worth holding resident.
        dataset = dataset.cache()

    return dataset.batch(batch_size).prefetch(AUTOTUNE)


def split_counts() -> Dict[str, Dict[str, int]]:
    """Count files per split and class — used for the dataset report."""
    return {
        split: {cls: len(list(C.split_dir(split, cls).glob("*.jpg"))) for cls in C.CLASS_NAMES}
        for split in C.SPLITS
    }


def unbatched_arrays(split: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Materialise a split as ``(images, labels, filepaths)``.

    Used by the evaluation and robustness scripts, which need per-image
    predictions lined up with the file each one came from.

    Images come back as **uint8**: a 900-image split is 135 MB that way against
    540 MB as float32, which matters on a machine with about a gigabyte free.
    Keras casts at the model's input layer, so this feeds ``predict`` directly;
    callers doing arithmetic on the pixels should cast explicitly.
    """
    directory = C.PROCESSED_DIR / split
    files: List[str] = []
    labels: List[int] = []
    for index, cls in enumerate(C.CLASS_NAMES):
        for path in sorted(C.split_dir(split, cls).glob("*.jpg")):
            files.append(str(path))
            labels.append(index)

    images = np.zeros((len(files), *C.IMG_SIZE, C.IMG_CHANNELS), dtype=np.uint8)
    for i, path in enumerate(files):
        raw = tf.io.read_file(path)
        decoded = tf.image.decode_jpeg(raw, channels=3)
        decoded = tf.image.resize(decoded, C.IMG_SIZE, method="bilinear")
        images[i] = np.clip(decoded.numpy(), 0, 255).astype(np.uint8)

    return images, np.asarray(labels, dtype=np.int32), files


def class_weights(split: str = "train") -> Dict[int, float]:
    """
    Inverse-frequency class weights. Our splits are built balanced, so these
    come out at ~1.0 each; the function exists so an unbalanced rebuild is
    handled correctly rather than silently biasing the model.
    """
    counts = split_counts()[split]
    total = sum(counts.values())
    n_classes = len(C.CLASS_NAMES)
    return {
        index: total / (n_classes * max(1, counts[cls]))
        for index, cls in enumerate(C.CLASS_NAMES)
    }
