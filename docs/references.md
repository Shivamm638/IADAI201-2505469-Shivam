# Research Findings and References

The research below shaped specific decisions in this project. Each entry says
what was taken from it, not merely that it exists.

---

## 1. Research that shaped the project

### 1.1 The problem is information, not capacity

Work on urban cruising established that a significant share of traffic in dense
districts consists of drivers searching for parking rather than travelling to a
destination, and that the delay is concentrated in the *search*, not in the
availability of space.

> **What this changed:** the system's success criterion became *how accurate is
> the free-bay count a driver acts on*, not *how accurate is the classifier*.
> That is why [`scripts/04_robustness_test.py`](../scripts/04_robustness_test.py)
> reports scene-level counting error alongside slot-level accuracy, and why the
> dashboard leads with a recommendation sentence rather than a number.

### 1.2 PKLot: bay-level annotation is what makes the task learnable

**Almeida, P., Oliveira, L. S., Britto, A. S., Silva, E. J., & Koerich, A. L.
(2015).** *PKLot — A robust dataset for parking lot classification.*
**Expert Systems with Applications, 42(11), 4937–4949.**
<https://doi.org/10.1016/j.eswa.2015.02.009>

The founding paper for this task. Three findings were used directly:

1. **Bay-level classification beats whole-image analysis.** The authors classify
   each segmented bay independently — the approach adopted here.
2. **Cross-lot generalisation is the real test.** Models trained on one lot
   degrade on another because camera angle and bay geometry change. This is why
   accuracy is reported **per lot** in
   [`results/metrics.json`](../results/metrics.json), not only in aggregate.
3. **Weather is the dominant nuisance variable.** Rain and overcast conditions
   drive most of the error, which is why weather-balanced sampling and
   **per-weather accuracy reporting** are built into the pipeline.

> **What this changed:** the entire evaluation design. Reporting a single
> average accuracy would have hidden exactly the failure modes this paper
> identifies.

### 1.3 Near-duplicate frames make random splits meaningless

PKLot frames are captured every five minutes. Any evaluation that splits
individual bay crops at random places a crop and its five-minute-later twin on
opposite sides of the train/test boundary, and reports memorisation as
generalisation.

> **What this changed:** the split is **grouped by whole (lot, calendar date)**
> — see [Step 2](02_data_preprocessing.md#24-the-split--and-the-leakage-trap).
> This lowers the headline accuracy and makes it mean something.

### 1.4 Transfer learning, and the order it must happen in

**Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018).**
*MobileNetV2: Inverted residuals and linear bottlenecks.* **CVPR 2018.**
<https://arxiv.org/abs/1801.04381>

**Yosinski, J., Clune, J., Bengio, Y., & Lipson, H. (2014).**
*How transferable are features in deep neural networks?* **NeurIPS 2014.**
<https://arxiv.org/abs/1411.1792>

Yosinski et al. show that early convolutional layers learn general features
(edges, colour, texture) that transfer across tasks, while later layers become
task-specific. Sandler et al. give an architecture whose accuracy-per-FLOP suits
CPU deployment.

> **What this changed:** the two-phase schedule. Early layers stay frozen
> because their features are already right for this task; only the top third is
> fine-tuned, at a 100× smaller learning rate, and only *after* the randomly
> initialised head has stabilised — otherwise the head's large early gradients
> destroy the pre-trained filters.

### 1.5 BatchNorm is the trap in fine-tuning

Standard Keras transfer-learning guidance documents that unfreezing
BatchNormalization layers during fine-tuning destabilises training, because the
running statistics estimated over ImageNet get overwritten by estimates from
very small batches.

> **What this changed:** `compile_for_finetune()` in
> [`src/model.py`](../src/model.py) explicitly keeps every
> `BatchNormalization` layer frozen even inside the unfrozen region.

### 1.6 Perspective distortion is a data problem, not a model problem

An elevated camera projects a rectangular bay to a trapezium whose shape varies
across the frame. Classifying an axis-aligned crop forces the model to spend
capacity undoing a distortion that can simply be removed.

> **What this changed:** `crop_slot()` in [`src/slots.py`](../src/slots.py)
> applies a perspective warp to deskew every bay to the same 224×224 square,
> and pads by 8% so overhanging vehicles stay inside the patch.

---

## 2. Datasets

| Resource | Use |
|---|---|
| **PKLot** — Almeida et al., 2015. [UFPR](https://web.inf.ufpr.br/vri/databases/parking-lot-database/) · [Voxel51 mirror](https://huggingface.co/datasets/Voxel51/PKLot) | The dataset used for training, validation and testing |
| **CNRPark-EXT** — Amato et al. <http://cnrpark.it/> | Consulted as an alternative benchmark; PKLot was chosen for its polygon annotations, which the demo overlays require |

---

## 3. Key papers

| Reference | Relevance |
|---|---|
| Almeida et al. (2015), *PKLot*, **Expert Systems with Applications** 42(11) | The dataset and the bay-level classification approach |
| Sandler et al. (2018), *MobileNetV2*, **CVPR** | The backbone architecture |
| Howard et al. (2017), *MobileNets*, <https://arxiv.org/abs/1704.04861> | Depthwise-separable convolutions for constrained hardware |
| Yosinski et al. (2014), *How transferable are features?*, **NeurIPS** | Justifies freezing early layers and fine-tuning only the top |
| Amato et al. (2017), *Deep learning for decentralised parking lot occupancy detection*, **Expert Systems with Applications** 72 | CNN-based occupancy detection on embedded devices; supports the small-model choice |
| Deng et al. (2009), *ImageNet*, **CVPR** | Source of the pre-trained weights |
| Szegedy et al. (2016), *Rethinking the Inception Architecture*, **CVPR** | Origin of label smoothing, used here for calibration |
| Redmon et al. (2016), *You Only Look Once*, **CVPR** | The detection alternative considered and rejected in [Step 1](01_problem_definition.md#14-choosing-the-approach) |

---

## 4. Technical documentation

| Resource | Link |
|---|---|
| TensorFlow — transfer learning and fine-tuning | <https://www.tensorflow.org/tutorials/images/transfer_learning> |
| Keras — image classification from scratch | <https://keras.io/examples/vision/image_classification_from_scratch/> |
| Keras Applications — MobileNetV2 | <https://keras.io/api/applications/mobilenet/> |
| OpenCV — geometric image transformations | <https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html> |
| Streamlit — API reference | <https://docs.streamlit.io/develop/api-reference> |
| Streamlit — caching | <https://docs.streamlit.io/develop/concepts/architecture/caching> |
| Streamlit Community Cloud — deployment | <https://docs.streamlit.io/deploy/streamlit-community-cloud> |
| Ultralytics YOLO documentation | <https://docs.ultralytics.com/> |
| scikit-learn — classification metrics | <https://scikit-learn.org/stable/modules/model_evaluation.html> |
| Roboflow — annotation and dataset management | <https://docs.roboflow.com/> |

---

## 5. What was rejected, and why

Recording the discarded options is part of the research record.

| Option | Why it was rejected |
|---|---|
| **YOLO object detection** | Detects cars, not bays. An empty bay produces no detection, so "empty" would have to be inferred from absence — and absence is indistinguishable from a missed detection. Also far heavier than free-tier CPU hosting allows |
| **Training a CNN from scratch** | 4,200 training crops is nowhere near enough to learn general visual features; transfer learning reaches higher accuracy in a fraction of the time |
| **EfficientNetB0** | ~2.3× slower on CPU for a binary task that does not need the extra capacity |
| **Random per-crop split** | Leaks near-identical frames across the train/test boundary and inflates the reported accuracy |
| **Axis-aligned bounding-box crops** | Pull in slices of neighbouring bays, so a car parked next door corrupts the label |
| **Vertical-flip augmentation** | Produces images a fixed overhead camera can never capture |
| **Classical CV (background subtraction, edge density)** | Brittle under shadow, rain and lighting change — precisely the conditions that matter |
| **Tuning the decision threshold on the test split** | Would fit the test set and then report the fit as a result; the threshold is chosen on validation only |

---

**Back to:** [README](../README.md)
