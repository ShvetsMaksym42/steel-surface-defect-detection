![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue)
![Computer Vision](https://img.shields.io/badge/Area-Computer%20Vision-success)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Result at a Glance

| Metric | Performance / Outcome |
|--------|----------------------|
| mAP50 (test) | 0.647 — Baseline YOLOv8m |
| Inference speed | 357ms per image / 119ms per tile |
| Dataset scale | 12,568 images → 37,704 tiles |
| Pipeline | RLE masks → YOLO bboxes, end-to-end |
| Training | 85 epochs on Kaggle T4 GPU |

---

# Severstal Steel Defect Detection

**Object detection** pipeline for industrial steel surface defects using **YOLOv8m**, trained on the [Severstal Steel Defect Detection](https://www.kaggle.com/competitions/severstal-steel-defect-detection) dataset.

---

## Overview

The dataset contains:
- **12,568** grayscale steel surface images (256×1600)
- **4** defect classes annotated with RLE masks 
Since the task is object detection, the pipeline converts RLE masks to bounding boxes, slices images into overlapping tiles, and trains YOLOv8m on the resulting dataset.

**Key challenges:**
- RLE masks use 1-indexed, column-major order encoding
- Unusual image dimensions (256×1600) require tiling
- Severe class imbalance: Class 2 has only 247 samples vs 5,150 for Class 3
- 47% of images are defect-free
- Coarse annotations — masks were drawn roughly, reducing bbox precision

---

## Pipeline

1. Decode RLE masks → binary masks per class
2. Slice images into overlapping tiles (256×608, 3 per image)
3. Extract bboxes via connected components analysis
4. Stratified train/val/test split (75/15/10)
5. YOLOv8m training

**Tiling:** Each image is split into 3 overlapping tiles with width 608px (~113px overlap). Overlap ensures defects near tile boundaries appear fully on at least one tile.

**Adaptive thresholds:** Minimum blob area per class was derived from median blob area analysis (`eda/analyze_blobs.py`), using lower thresholds for rare classes to avoid discarding valid annotations.

**Stratified split:** Images are split by original filename (not tile), using class priority order (rarest class first) to ensure rare classes are proportionally distributed across splits.

---

## Experiments

Two training runs were conducted to investigate the effect of class imbalance on model performance.

### Baseline
Standard training on the full dataset with default sampling.

- 85 epochs, batch=32, YOLOv8m pretrained on COCO
- 28,278 tiles (31% defect, 69% defect-free)

### Balanced
An attempt to reduce the impact of class imbalance:
- **Negative undersampling:** A custom `BalancedDetectionTrainer` samples a random subset of negative tiles each epoch (30% of total), without discarding any data — all negatives remain available across epochs.
- **Synthetic inclusion tiles:** 50 augmented samples generated from the 185 real inclusion tiles in the train set (flip + HSV jitter + brightness) to partially compensate for Class 2 scarcity.
- 50 epochs, batch=8, starting from pretrained COCO weights.

---

## Results (Test Set)

| Model    | mAP50 | mAP50-95 | Precision | Recall |
|----------|-------|----------|-----------|--------|
| Baseline | 0.647 | 0.385    | 0.612     | 0.625  |
| Balanced | 0.598 | 0.337    | 0.631     | 0.574  |

**Per-class mAP50 (Baseline):**

| Class         | mAP50 | Recall |
|---------------|-------|--------|
| crazing       | 0.556 | 0.579  |
| inclusion     | 0.552 | 0.532  |
| patches       | 0.759 | 0.741  |
| pitted_surface| 0.689 | 0.647  |

The balanced experiment did not improve overall metrics, likely due to fewer training epochs (50 vs 85) and the OOM-forced batch size reduction from 32 to 8, which introduced noisy gradients.

<p align="center">
  <img src="media/yolo_evaluation/baseline/val_batch2_pred.jpg" width="800"/>
  <br><em>Baseline predictions on test set</em>
</p>

<p align="center">
  <img src="media/yolo_evaluation/baseline/confusion_matrix_normalized.png" width="500"/>
  <br><em>Normalized confusion matrix — Baseline (test set)</em>
</p>

The baseline model processes a full 256×1600 image in ~357ms on CPU (3 tiles sequentially, ~119ms each).

> [!NOTE]
> Inference benchmarked on CPU (AMD Ryzen 5 3600, 6-core).

---

## Future Work

- **Segmentation model:** U-Net with EfficientNet-B0 backbone can use the original RLE masks directly, avoiding the lossy RLE→bbox conversion. Segmentation is better suited for diffuse defects like crazing.
- **Cascade architecture:** YOLO detects defect regions, a lightweight classifier refines the class prediction — trading latency for recall.
- **API & Deployment:** REST API with FastAPI, containerized with Docker.

---

> Full source code, preprocessing pipeline, and Kaggle training notebooks are in this repository.  
> Baseline weights (85 epochs): [Kaggle Models](https://www.kaggle.com/models/shvetsmaksym42/yolo-baseline)
