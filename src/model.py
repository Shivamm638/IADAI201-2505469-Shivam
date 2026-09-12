"""
Step 3 of the brief: model development.

Architecture
------------
MobileNetV2 pre-trained on ImageNet, with the classifier head replaced by a
global-average-pool -> dropout -> single sigmoid unit. MobileNetV2 is the right
backbone here for three reasons:

* the brief names MobileNet/EfficientNet as the appropriate classifiers;
* at ~2.3 M trainable parameters it fine-tunes on CPU in minutes, and runs a
  100-bay lot in well under a second on Streamlit Community Cloud's free tier;
* depthwise-separable convolutions cope well with the low-resolution, heavily
  perspective-distorted bay crops PKLot produces.

Training runs in two phases (see :mod:`src.train`): the backbone is frozen
while the new head learns, then the top blocks are unfrozen at a 100x smaller
learning rate. Fine-tuning a randomly initialised head straight away would push
large gradients through pre-trained weights and destroy them.
"""

from __future__ import annotations

from typing import Tuple

import keras as k3
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from . import config as C


# `tensorflow.keras` is a compatibility namespace here that does not expose
# `saving`, `random` or `ops`; the real Keras 3 package does, and its layer
# classes are the very same objects, so the two mix safely.
@k3.saving.register_keras_serializable(package="parkvision")
class RandomSensorNoise(layers.Layer):
    """
    Additive Gaussian noise with a random strength per image — training only.

    Keras's own ``GaussianNoise`` applies one fixed standard deviation to every
    image, so a model trained with it never sees a clean input. This draws a
    fresh strength for each image from ``U(0, max_stddev)`` and applies it to
    only a ``probability`` share of the batch, so the model learns that grain
    is not a car while still seeing plenty of clean bays.

    At inference it is the identity. The layer is registered as serialisable so
    it round-trips through ``.keras`` files; :func:`src.inference.load_model`
    imports this module before loading for exactly that reason.
    """

    def __init__(self, max_stddev: float = C.AUG_NOISE_MAX_STDDEV,
                 probability: float = C.AUG_NOISE_PROBABILITY,
                 value_range: Tuple[float, float] = (0.0, 255.0),
                 seed=None, **kwargs):
        super().__init__(**kwargs)
        self.max_stddev = float(max_stddev)
        self.probability = float(probability)
        self.value_range = (float(value_range[0]), float(value_range[1]))
        self.seed = seed
        self.seed_generator = k3.random.SeedGenerator(seed)

    def call(self, inputs, training=None):
        if not training:
            return inputs
        ops = k3.ops
        batch = ops.shape(inputs)[0]
        strength = k3.random.uniform(
            (batch, 1, 1, 1), 0.0, self.max_stddev, seed=self.seed_generator
        )
        chosen = k3.random.uniform(
            (batch, 1, 1, 1), 0.0, 1.0, seed=self.seed_generator
        ) < self.probability
        noise = k3.random.normal(ops.shape(inputs), seed=self.seed_generator) * strength
        noisy = ops.clip(inputs + noise, self.value_range[0], self.value_range[1])
        return ops.where(chosen, noisy, inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            max_stddev=self.max_stddev,
            probability=self.probability,
            value_range=self.value_range,
            seed=self.seed,
        )
        return config


def build_augmenter() -> keras.Sequential:
    """
    Training-time augmentation: the rotation, flipping and brightness changes
    the brief asks for, plus zoom/translation to mimic camera drift, contrast
    jitter to mimic overcast and rainy days, and sensor noise to mimic cheap or
    low-light cameras.

    Only horizontal flips are used. A vertical flip would put a car's roof
    where its shadow belongs, which never happens in a fixed overhead camera.

    Noise comes last, after the photometric changes, and is clipped back into
    [0, 255] so the rescaling layer that follows still receives valid pixels.
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", name="aug_flip"),
            layers.RandomRotation(C.AUG_ROTATION, fill_mode="reflect", name="aug_rotate"),
            layers.RandomZoom(C.AUG_ZOOM, fill_mode="reflect", name="aug_zoom"),
            layers.RandomTranslation(
                C.AUG_TRANSLATION, C.AUG_TRANSLATION, fill_mode="reflect", name="aug_shift"
            ),
            layers.RandomContrast(C.AUG_CONTRAST, name="aug_contrast"),
            layers.RandomBrightness(C.AUG_BRIGHTNESS, value_range=(0, 255), name="aug_brightness"),
            RandomSensorNoise(seed=C.RANDOM_SEED, name="aug_noise"),
        ],
        name="augmentation",
    )


def build_model(
    input_shape: Tuple[int, int, int] = (*C.IMG_SIZE, C.IMG_CHANNELS),
    dropout: float = C.DROPOUT,
    augment: bool = True,
) -> keras.Model:
    """
    Assemble the full classifier.

    Augmentation and MobileNetV2's own ``preprocess_input`` are baked into the
    graph, so callers feed raw uint8-valued RGB tensors in [0, 255] and never
    have to remember a normalisation step. That removes the single most common
    train/serve skew bug in transfer-learning projects.
    """
    inputs = keras.Input(shape=input_shape, name="slot_image")

    x = inputs
    if augment:
        x = build_augmenter()(x)

    # MobileNetV2 expects inputs scaled to [-1, 1].
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale")(x)

    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        name="mobilenetv2_backbone",
    )
    backbone.trainable = False

    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="occupancy")(x)

    model = keras.Model(inputs, outputs, name="parkvision_mobilenetv2")
    return model


def get_backbone(model: keras.Model) -> keras.Model:
    """Return the MobileNetV2 sub-model inside a built classifier."""
    return model.get_layer("mobilenetv2_backbone")


def compile_for_head(model: keras.Model, lr: float = C.LR_HEAD) -> keras.Model:
    """Phase 1: frozen backbone, only the new head learns."""
    get_backbone(model).trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=C.LABEL_SMOOTHING),
        metrics=_metrics(),
    )
    return model


def compile_for_finetune(
    model: keras.Model,
    lr: float = C.LR_FINETUNE,
    from_layer: int = C.FINE_TUNE_FROM_LAYER,
) -> keras.Model:
    """
    Phase 2: unfreeze the top of the backbone at a much smaller learning rate.

    BatchNormalization layers stay frozen. Their running statistics were
    estimated over ImageNet's millions of images; re-estimating them from our
    comparatively tiny batches is a well-known way to make fine-tuning unstable.
    """
    backbone = get_backbone(model)
    backbone.trainable = True
    for i, layer in enumerate(backbone.layers):
        if i < from_layer or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=C.LABEL_SMOOTHING),
        metrics=_metrics(),
    )
    return model


def _metrics():
    return [
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
    ]


def trainable_parameter_count(model: keras.Model) -> int:
    return int(sum(tf.size(w).numpy() for w in model.trainable_weights))
