# Small-Data Plots Report (YOLOv12 on COCO Subsets)

This report is auto-generated from `analysis_out/*.csv`.

> Dataset size on the x-axis is plotted with **categorical spacing** (equal distance between sizes).

---

## 1) Best mAP50–95 vs Dataset Size (categorical)

Absolute best mAP50–95 for each model, with equal spacing between dataset sizes.

![Best mAP50–95 vs dataset size](plots/analysis_out/best_map5095_vs_dataset_size_categorical.png)

---

## 2) Retention vs Dataset Size (categorical)

Retention is computed per model as `best@X / best@100%`.

![Retention vs dataset size](plots/analysis_out/retention_vs_dataset_size_categorical.png)

---

## 3) Cliff Effect (5% → 2.5%)

Absolute drop in best mAP50–95 when moving from 5% to 2.5%.

![Cliff effect 5 to 2.5](plots/analysis_out/cliff_effect_5_to_2_5_delta.png)

---

## 4) Deltas Across Key Transitions

Change in best mAP50–95 across the standard transition ladder.

![Deltas across transitions](plots/analysis_out/deltas_across_transitions.png)

---

## 5) Late-Epoch Peaking Indicator (categorical)

Plot of `best_epoch / last_epoch`, with equal spacing between dataset sizes.

![Best epoch ratio](plots/analysis_out/best_epoch_ratio_vs_dataset_size_categorical.png)

---
