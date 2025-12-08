# YOLOv12 Small-Data Investigation  
### Full Analysis of Training YOLOv12 on Extremely Small COCO Subsets

---

## 📌 Project Overview

This repository presents a complete study of **how YOLOv12 behaves when trained on very small subsets of the COCO dataset**—from **0.3125% up to 100%** of COCO train2017.

The primary research goals were:

- to understand **how dataset size affects accuracy**,  
- to evaluate **stability of training** on extremely small datasets,  
- to compare **five YOLOv12 model sizes (n, s, m, l, x)**,  
- to measure **training time and convergence behavior**,  
- and to analyze **class distribution retention** under proportional subsampling.

All experiments were run on the **Leonardo Supercomputer**:

- 4 compute nodes  
- 4× GPUs per node  
- **16 GPUs total**  
- Distributed training via SLURM  

The full experimental pipeline — dataset creation, training scripts, subset analysis, and visualization — is included and fully reproducible.

---

# 📁 Dataset Construction

## Step 1 — Removing Empty Images

Before building any reduced dataset, we scanned the full COCO train2017 list and **removed all images with zero annotated objects**.

This ensures that proportional sampling does not select empty samples and maintains valid class statistics in every subset.

The cleaned image list was used as the base source for proportional subset generation.

---

## Step 2 — Subsampling Strategies

Each dataset percentage was generated using **three strategies**:

### 🔷 1. **Proportional (P)** — *Class-balanced subsampling (used for training)*  
This method preserves class proportions by sampling images so that every COCO class is represented according to the original class frequency.

```
train2017_<percent>_p.txt
```

This is the **main dataset type used for all YOLOv12 training** in this study.

---

### 🔶 2. Random  
Uniform random selection of images.

```
train2017_<percent>_random.txt
```

---

### 🔷 3. Top-N  
Takes the first N samples from the filtered COCO list.  
Not class-balanced, included only for comparison.

```
train2017_<percent>_topn.txt
```

---

## Step 3 — Available Dataset Sizes

The following dataset sizes were generated:

```
0.3125%, 0.625%, 1.25%, 2.5%, 5%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%
```

All subset files are located in:

```
datasets/
```

Additional aggregated class statistics are located here:

```
datasets/class_distribution.csv
```

---

# 📉 Dataset Analysis & Distribution Plots

Below are the main visualizations showing how the dataset shrinks at different scales.

### **Overall images & objects (Proportional / Random / Top-N)**

![Overall P](plots/datasets/overall_p_0_5_and_10_100.png)
![Overall Random](plots/datasets/overall_random_0_5_and_10_100.png)
![Overall TopN](plots/datasets/overall_topn_0_5_and_10_100.png)

---

### **Per-Class Trends for Proportional Sampling**

These plots show how each of the 80 COCO classes is reduced proportionally.

#### 0–5% Range
![Per-Class P 0–5%](plots/datasets/per_class_grid_p_0_5_range.png)

#### 10–100% Range
![Per-Class P 10–100%](plots/datasets/per_class_grid_p_10_100_range.png)

---

# 🧪 YOLOv12 Training Experiments

We trained the following model sizes:

```
YOLOv12-n
YOLOv12-s
YOLOv12-m
YOLOv12-l
YOLOv12-x
```

Each model was trained on **all 15 dataset scales**, using the official YOLOv12 distributed training procedure.

Training logs are stored in:

```
small_data_logs/
```

---

# 📈 Training Behavior & Performance

## mAP5095 vs Epoch  
(Overview grid for all models across dataset sizes)

![mAP5095 Models](plots/train/map5095_models_grid.png)

---

## Best mAP vs Dataset Size

![Best mAP](plots/train/best_map5095_vs_size_lines.png)

---

# 📊 Final Results Summary

### ❗ DO NOT MERGE INTO MAIN UNTIL THIS SECTION IS FILLED 
**Insert the final mAP values for each model and each dataset percentage.**

```
| Dataset % | YOLOv12-n | YOLOv12-s | YOLOv12-m | YOLOv12-l | YOLOv12-x |
|-----------|-----------|-----------|-----------|-----------|-----------|
| 0.3125%   | <INSERT>  | <INSERT>  | <INSERT>  | <INSERT>  | <INSERT>  |
| 0.625%    | <INSERT>  | <INSERT>  | <INSERT>  | <INSERT>  | <INSERT>  |
| ...       | ...       | ...       | ...       | ...       | ...       |
| 100%      | <INSERT>  | <INSERT>  | <INSERT>  | <INSERT>  | <INSERT>  |
```

---

# 📦 Repository Structure (Simplified)

```
yolov12_SmallData/
│
├── datasets/  
│   ├── train2017_<percent>_p.txt  
│   ├── train2017_<percent>_random.txt  
│   ├── train2017_<percent>_topn.txt  
│   └── class_distribution.csv
│
├── plots/
│   ├── datasets/    # Dataset distribution plots
│   └── train/       # Training curves and evaluation charts
│
├── small_data_logs/  # YOLOv12 training logs
└── README.md
```

---

# 📝 TEMPLATE BLOCK — TO BE COMPLETED LATER  
### ❗ DO NOT MERGE INTO MAIN UNTIL THIS SECTION IS FILLED

**THE FOLLOWING SECTION IS A TEMPLATE.**

---

# ⬜ TEMPLATE: Full Reproduction Guide

## 1️⃣ Downloading and Preparing COCO  
**DESCRIBE WHICH SCRIPT DOWNLOADS COCO AND WHERE IT STORES FILES.**

## 2️⃣ Generating Proportional / Random / Top-N Subsets  
**DESCRIBE HOW TO RUN THE SUBSET GENERATOR.**

## 3️⃣ Creating YAML Training Configs  
**DESCRIBE HOW YAML FILES ARE GENERATED OR WHERE THEY ARE STORED.**

## 4️⃣ Generating SLURM Training Scripts  
**EXPLAIN HOW SLURM FILES ARE GENERATED FROM A TEMPLATE.**

## 5️⃣ Launching Training on Leonardo  
**EXPLAIN HOW TO RUN SBATCH AND WHAT OUTPUT TO EXPECT.**

---

# 🙌 Acknowledgements

- **Leonardo Supercomputer** — for HPC compute resources  
- **YOLOv12 Research Team** — for model architecture & training code  
- **COCO Consortium** — for dataset availability  
- Everyone who contributed to distributed training and analysis  
