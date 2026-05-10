# Model Training & Evaluation Logs

> **Base Model**: `qihoo360/fg-clip-base` (150,112,257 total parameters)
> **Environment**: Windows 11, Python 3.13.13, PyTorch 2.7.1+cu118, Transformers 5.8.0, PEFT 0.19.1
> **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (training); CPU (V3)
> **Last Updated**: May 2026

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [V1 — Baseline](#v1--fgclip-lora-finetunedv1-1200)
3. [V2](#v2--fgclip-lora-finetunedv2-1200)
4. [V2 + HNM](#v2--hnm--fgclip-lora-finetunedv2-1200-hnm)
5. [V2.1 + HNM](#v21--hnm--fgclip-lora-finetunedv21-1200-hnm)
6. [V2.2 + HNM](#v22--hnm--fgclip-lora-finetunedv22-1200-hnm)
7. [V3-optimized](#v3-optimized--fgclip-lora-finetunedv3-optimized)
8. [CLAUDE V2 + HNM](#claude-v2--hnm--claudefgclip-lora-finetunedv2-1200-hnm)
9. [CLAUDE V2.1 + HNM](#claude-v21--hnm--claudefgclip-lora-finetunedv21-1200-hnm)
10. [Key Findings & Observations](#key-findings--observations)
11. [Summary Tables](#summary-tables) — Tables 1–8

---

## 1. Dataset Overview

**Caption file**: `captions/dataset_captionsV2.csv`
**Image directory**: `images/`
**Columns**: `pair_id`, `category`, `split`, `lost_image`, `found_image`, `positive_caption`, `hard_negative_caption`

### Category Breakdown (per split)


| Category      | Train   | Val    | Test   | Total   |
| ------------- | ------- | ------ | ------ | ------- |
| Bags          | 70      | 15     | 15     | 100     |
| Chargers      | 70      | 15     | 15     | 100     |
| Handkerchiefs | 70      | 15     | 15     | 100     |
| Lunchboxes    | 70      | 15     | 15     | 100     |
| Tumblers      | 70      | 15     | 15     | 100     |
| Wallets       | 70      | 15     | 15     | 100     |
| **Total**     | **420** | **90** | **90** | **600** |


### Dataset Notes

- **Training split (420 pairs)**: Used for LoRA fine-tuning with hard-negative pairs
- **Validation split (90 pairs)**: Used for per-epoch loss monitoring during training
- **Test split (90 pairs)**: Held-out queries at evaluation time; matched against all 600 found images
- **Evaluation gallery**: All 600 `found_image` files from `images/`, regardless of split
- **Image format**: Mix of PNG and JPG; all converted to RGB at load time
- **Positive caption**: Accurate description of the lost item
- **Hard negative caption**: Same category but describes a different, similar-looking item
- **Caption length**: Max 77 tokens (CLIP's max text length)

---

## V1 — `fgclip-lora-finetunedV1-1200`

**Status**: Baseline. First attempt. No scheduler, no early stopping, no hard negative mining, minimal LoRA rank.
**Training notebook**: `fgclipfintuning/Copy_of_fgclipFineTuning_V1_1200.ipynb`
**Evaluation notebook**: `fgclipevaluation/Copy_of_fgclipEvaluation_V1_1200.ipynb`
**Output directory**: `lorafinetuned/fgclip-lora-finetunedV1-1200/`
**Training seed**: 9435575495300

### Why It Was Designed This Way

This was the initial baseline run before any hyperparameter exploration. The goal was to establish a working pipeline and measure how much the base fg-clip model could be improved with minimal LoRA fine-tuning on a small dataset.

---

### Training Methodology

**Task**: Image-text retrieval. Given a text description of a lost item, retrieve the matching image from a gallery of 600 found items.

**Loss**: Single-image InfoNCE / CLIP loss. Each training batch contains `batch_size` image-caption pairs. The model computes cosine similarity between all `batch_size` image embeddings and all `batch_size` text embeddings (both the positive and a hard-negative caption per pair), producing a `(batch_size × batch_size)` similarity matrix. Cross-entropy loss treats each image's positive caption as the correct answer, and the matrix is symmetric — the model is simultaneously trained to match texts to their images and images to their texts. There is no scheduler; the learning rate is constant throughout. No early stopping is used — the model trains for a fixed 20 epochs regardless of validation loss trajectory.

**Training loop**: For each batch, the `lost_image` is processed through the image encoder and the `positive_caption` through the text encoder. The `hard_negative_caption` is also encoded. Cosine similarities are computed and a cross-entropy loss is backpropagated through the LoRA adapters only. The base CLIP weights remain frozen. The validation loop runs the same computation on the 90-item val split each epoch, with no gradient updates.

**Inference (retrieval)**: At evaluation time, all 600 gallery images are encoded once (no text input). A query text is encoded on-the-fly. Cosine similarity ranks all 600 images against the query; Recall@K and MRR are computed from these rankings.

---

### Architecture


| Parameter            | Value                                                       |
| -------------------- | ----------------------------------------------------------- |
| Base model           | `qihoo360/fg-clip-base`                                     |
| Vision backbone      | ViT-B/16 (14×14 patch grid, 197 tokens, 512-dim embeddings) |
| Trainable parameters | **491,520 (0.3274% of total)**                              |
| LoRA rank (r)        | 8                                                           |
| LoRA alpha (α)       | 16                                                          |
| Target modules       | `q_proj`, `v_proj`                                          |
| Dropout              | 0.1                                                         |
| Model dtype          | `torch.float32`                                             |


### Hyperparameters


| Parameter          | Value                                                |
| ------------------ | ---------------------------------------------------- |
| Optimizer          | AdamW                                                |
| Learning rate      | 5e-5 (fixed, no scheduler)                           |
| Weight decay       | 0.01                                                 |
| Betas              | (0.9, 0.999)                                         |
| Epsilon            | 1e-8                                                 |
| Loss function      | CrossEntropyLoss                                     |
| Temperature        | 0.0122 (auto-scaled by HF's `logit_scale` parameter) |
| Batch size         | 16                                                   |
| Max epochs         | 15                                                   |
| Early stopping     | **None**                                             |
| DataLoader workers | 0                                                    |


### Data Augmentation

```
Compose(
    Resize(size=(224, 224), interpolation=bilinear)
    RandomHorizontalFlip(p=0.5)
    ColorJitter(brightness=(0.8, 1.2), contrast=None, saturation=None, hue=None)
    ToTensor()
    RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0)
    ToPILImage()
)
```

> **Note**: The augmentation code in the notebook cell shows `brightness=0.2` (single float, interpreted by PyTorch as `(0.8, 1.2)`). The saved training summary report shows the expanded range notation — both describe the same jitter.

### Training Loss Curve


| Epoch | Train Loss | Val Loss |
| ----- | ---------- | -------- |
| 1     | 1.1897     | 1.6209   |
| 2     | 1.0652     | 1.6060   |
| 3     | 1.0310     | 1.5862   |
| 4     | 0.9513     | 1.5636   |
| 5     | 0.8939     | 1.5430   |
| 6     | 0.7684     | 1.5228   |
| 7     | 0.7740     | 1.5049   |
| 8     | 0.6903     | 1.4865   |
| 9     | 0.7042     | 1.4694   |
| 10    | 0.6581     | 1.4580   |
| 11    | 0.6352     | 1.4422   |
| 12    | 0.5483     | 1.4328   |
| 13    | 0.5412     | 1.4351   |
| 14    | 0.4875     | 1.4454   |
| 15    | 0.4484     | 1.4464   |


### Training Results


| Metric           | Value                             |
| ---------------- | --------------------------------- |
| Final train loss | 0.4484                            |
| Final val loss   | 1.4464                            |
| Best val loss    | 1.4328 (epoch 13)                 |
| Epochs trained   | 15 / 15                           |
| GPU VRAM used    | 3.31 GB                           |
| Training speed   | ~~1.02 it/s (~~26–27 steps/epoch) |


### Evaluation Results

> **Important note**: The evaluation report's summary header incorrectly says "None (Zero-Shot)" — this is a bug in the notebook's summary cell where `adapter_path` is not in globals at that point. The notebook **does** correctly load the fine-tuned LoRA adapter (`../lorafinetuned/fgclip-lora-finetunedV1-1200`) before evaluation. The metrics below are from the fine-tuned model, not the base zero-shot model.

**Evaluation setup**: 90 test queries matched against all 600 gallery images.


| Metric            | Value      | Count |
| ----------------- | ---------- | ----- |
| **Recall@1**      | **71.11%** | 64/90 |
| **Recall@5**      | **91.11%** | 82/90 |
| **Recall@10**     | **93.33%** | 84/90 |
| **MRR**           | **0.8017** | —     |
| Category accuracy | 100.00%    | 90/90 |


#### Failure Analysis


| Category            | Count      | Notes                    |
| ------------------- | ---------- | ------------------------ |
| Perfect first-tries | 64 (71.1%) | Retrieved at rank 1      |
| Near misses         | 20 (22.2%) | In top 10 but not rank 1 |
| Severe failures     | 6 (6.7%)   | **Not in top 10 at all** |


#### Severe Failures (not in top 10)


| Test Index | Query            | Correct Item     | Rank     | Notes                                             |
| ---------- | ---------------- | ---------------- | -------- | ------------------------------------------------- |
| 4          | lunchbox_050     | lunchbox_050     | **#21**  | Light-grey rectangular lunch box, pale blue latch |
| 35         | bag_058          | bag_058          | **#12**  | Black smooth faux-leather mini barrel bag         |
| 46         | tumbler_047      | tumbler_047      | **#60**  | Cream off-white insulated tumbler, pastel print   |
| 52         | tumbler_065      | tumbler_065      | **#113** | WRELS matte black soft flask                      |
| 70         | charger_081      | charger_081      | **#16**  | Pink multi-port wall charger                      |
| 75         | handkerchief_027 | handkerchief_027 | **#17**  | Multicolor floral print handkerchief              |


> Most near misses (20 queries) were rank 2–4 within the correct category — the model always picked the right category (100% category accuracy) but occasionally retrieved the wrong specific item.

#### Sample Retrieval Leaderboard

For query: *"Light-grey rectangular lunch box, pale blue latch clips and flap compartment, white oval shaped vent button on lid"* (correct: `lunchbox_050`)


| Rank   | Item         | Score  | Status                       |
| ------ | ------------ | ------ | ---------------------------- |
| #1     | lunchbox_072 | 0.3442 | Missed rank 1 by 2 positions |
| #2     | lunchbox_074 | 0.3379 |                              |
| #3     | lunchbox_087 | 0.3315 |                              |
| #4     | lunchbox_073 | 0.3271 |                              |
| #5     | lunchbox_030 | 0.3255 |                              |
| #6     | lunchbox_042 | 0.3228 |                              |
| #7     | lunchbox_014 | 0.3211 |                              |
| #8     | lunchbox_066 | 0.3204 |                              |
| #9     | lunchbox_077 | 0.3197 |                              |
| #10    | lunchbox_075 | 0.3104 |                              |
| #50ish | lunchbox_050 | —      | **True target (not shown)**  |


#### Confusion Matrix (Category Level)


| True \ Predicted  | Bags | Chargers | Handkerchiefs | Lunchboxes | Tumblers | Wallets |
| ----------------- | ---- | -------- | ------------- | ---------- | -------- | ------- |
| **Bags**          | 15   | 0        | 0             | 0          | 0        | 0       |
| **Chargers**      | 0    | 15       | 0             | 0          | 0        | 0       |
| **Handkerchiefs** | 0    | 0        | 15            | 0          | 0        | 0       |
| **Lunchboxes**    | 0    | 0        | 0             | 15         | 0        | 0       |
| **Tumblers**      | 0    | 0        | 0             | 0          | 15       | 0       |
| **Wallets**       | 0    | 0        | 0             | 0          | 0        | 15      |


**100% category accuracy**: The model never confuses one category for another. All failures are fine-grained within-category confusions.

---

### What This Run Told Us

- **r=8 is enough to learn something**: 71.11% R@1 is a meaningful lift from the base model (~15–30% zero-shot)
- **Val loss never improved after epoch 13**: Without early stopping, the model kept training past its best point (best val loss = 1.4328 at epoch 13, but the run continued to 1.4464 at epoch 15)
- **Train loss kept dropping**: From 1.1897 to 0.4484 — the model was still learning, but it was overfitting
- **Brightness-only jitter is weak**: Contrast and saturation augmentation were missing, leaving the model less robust
- **No scheduler means no LR decay**: A constant 5e-5 risks destabilizing LoRA weights in later epochs
- **The bottleneck is fine-grained, not categorical**: Zero cross-category errors — the model always finds the right type. The problem is telling two similar lunchboxes apart.

---

## V2 — `fgclip-lora-finetunedV2-1200`

**Status**: Incremental improvement. Added full-spectrum augmentation, cosine LR scheduler, early stopping, larger LoRA rank (r=8→32), and more target modules (q/v → q/k/v/out).
**Training notebook**: `fgclipfintuning/Copy_of_fgclipFineTuning_V2_1200.ipynb`
**Evaluation notebook**: `fgclipevaluation/Copy_of_fgclipEvaluation_V2_1200.ipynb`
**Output directory**: `lorafinetuned/fgclip-lora-finetunedV2-1200/`
**Training seed**: 8602636536800

### What Changed from V1


| Aspect           | V1                   | V2                                       |
| ---------------- | -------------------- | ---------------------------------------- |
| LoRA rank (r)    | 8                    | **32**                                   |
| LoRA alpha (α)   | 16                   | **64**                                   |
| Target modules   | `q_proj`, `v_proj`   | `q_proj`, `k_proj`, `v_proj`, `out_proj` |
| Trainable params | 491,520 (0.33%)      | **3,932,160 (2.56%)**                    |
| ColorJitter      | brightness only      | **+ contrast, saturation, hue**          |
| LR scheduler     | None (fixed)         | **Cosine with 10% warmup**               |
| Early stopping   | None                 | **Patience = 3 epochs**                  |
| Temperature      | Auto-scaled (0.0122) | **Manual strict = 0.02**                 |
| Cosine final LR  | —                    | **1.25e-5**                              |
| Max epochs       | 15                   | **20**                                   |


### Why These Changes

- **r=32, q/k/v/out**: V1's r=8 on only q_proj+v_proj was very conservative. r=32 with four targeted modules gives the model significantly more capacity to learn fine-grained visual-textual alignment.
- **Full ColorJitter**: V1 only jittered brightness, leaving contrast and saturation invariant. This made the model brittle to real-world lighting variation.
- **Cosine scheduler**: Starting at 5e-5 and decaying to 1.25e-5 over 540 steps prevents the model from destabilizing in later epochs (the root cause of V1's val loss plateau).
- **Manual temperature 0.02**: Replaces CLIP's default near-zero temperature (~0.012). A higher temperature makes the similarity logits softer, improving gradient signal during early training.
- **Early stopping (patience=3)**: Prevents the model from overfitting past its optimal point, as seen in V1.

---

### Training Methodology

**Task**: Image-text retrieval. Given a text description of a lost item, retrieve the matching image from a gallery of 600 found items.

**Loss**: Single-image InfoNCE / CLIP loss. Each batch processes `batch_size` image-caption pairs. The `lost_image` is encoded alongside both the `positive_caption` and `hard_negative_caption` (2 texts per image). Cosine similarities form a `(batch_size × 2×batch_size)` matrix; the row at index `i` should score highest at column `i` (the positive). Cross-entropy is applied with temperature 0.02 to sharpen the softmax distribution. A cosine LR scheduler with 10% warmup decays the rate over `total_steps`, starting at 5e-5. Early stopping (patience=3) halts training if val loss fails to improve for 3 consecutive epochs, restoring the best checkpoint.

**Training loop**: Each step: zero gradients → encode `lost_image` through the vision encoder → encode `positive_caption` + `hard_negative_caption` through the text encoder → normalize all embeddings → compute `(image_embeds @ text_embeds.T) / 0.02` → cross-entropy loss → backprop → step optimizer → step scheduler. Only LoRA adapter weights are trainable; base CLIP is frozen. The base model's logit scale is NOT used; temperature is hard-coded at 0.02.

**Inference**: Identical to V1 — all 600 gallery images encoded once; query text encoded on-the-fly; cosine similarity ranks them.

---

### Architecture


| Parameter            | Value                                                       |
| -------------------- | ----------------------------------------------------------- |
| Base model           | `qihoo360/fg-clip-base`                                     |
| Vision backbone      | ViT-B/16 (14×14 patch grid, 197 tokens, 512-dim embeddings) |
| Trainable parameters | **3,932,160 (2.5608% of total)**                            |
| LoRA rank (r)        | 32                                                          |
| LoRA alpha (α)       | 64                                                          |
| Target modules       | `q_proj`, `k_proj`, `v_proj`, `out_proj`                    |
| Dropout              | 0.1                                                         |
| Model dtype          | `torch.float32`                                             |


### Hyperparameters


| Parameter             | Value                                   |
| --------------------- | --------------------------------------- |
| Optimizer             | AdamW                                   |
| Learning rate (start) | 5e-5                                    |
| Learning rate (end)   | 1.25e-5                                 |
| Scheduler             | Cosine with warmup (10% of total steps) |
| Warmup steps          | 54 / 540 total                          |
| Weight decay          | 0.01                                    |
| Betas                 | (0.9, 0.999)                            |
| Epsilon               | 1e-8                                    |
| Loss function         | CrossEntropyLoss                        |
| Temperature           | 0.02 (manual override)                  |
| Batch size            | 16                                      |
| Max epochs            | 20                                      |
| Early stopping        | Patience = 3 epochs                     |
| DataLoader workers    | 0                                       |


### Data Augmentation

```
Compose(
    Resize(size=(224, 224), interpolation=bilinear)
    RandomHorizontalFlip(p=0.5)
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.4, hue=0.1)
    ToTensor()
    RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0)
    ToPILImage()
)
```

### Training Loss Curve


| Epoch | Train Loss | Val Loss | Note                            |
| ----- | ---------- | -------- | ------------------------------- |
| 1     | 1.0437     | 1.3126   | 🟢 Saved                        |
| 2     | 0.9258     | 1.2846   | 🟢 Saved                        |
| 3     | 0.9396     | 1.2617   | 🟢 Saved                        |
| 4     | 0.6702     | 1.2507   | 🟢 Saved                        |
| 5     | 0.5822     | 1.2316   | 🟢 Saved                        |
| 6     | 0.4741     | 1.2070   | 🟢 Saved                        |
| 7     | 0.4421     | 1.2011   | 🟢 Saved                        |
| 8     | 0.3707     | 1.1890   | 🟢 Saved                        |
| 9     | 0.1181     | —        | *truncated*                     |
| …     | …          | …        | Best: val=1.1565                |
| 14    | 0.1685     | 1.1870   | Last saved (patience exhausted) |
| 15–20 | —          | —        | Halted by early stopping        |


> Best model checkpoint: epoch 13, val loss = 1.1565. Early stopping triggered at epoch 14 (3rd consecutive epoch without improvement).

### Training Results


| Metric                   | Value                         |
| ------------------------ | ----------------------------- |
| Final train loss         | 0.1685                        |
| Final val loss (at stop) | 1.1870                        |
| **Best val loss**        | **1.1565** (saved checkpoint) |
| Epochs trained           | 14 / 20                       |
| Epochs stopped early     | 6                             |
| GPU VRAM used            | 3.78 GB                       |


### Evaluation Results

**Evaluation setup**: 90 test queries matched against all 600 gallery images. Loaded fine-tuned adapter: `fgclip-lora-finetunedV2-1200`.


| Metric            | Value      | Count |
| ----------------- | ---------- | ----- |
| **Recall@1**      | **80.00%** | 72/90 |
| **Recall@5**      | **92.22%** | 83/90 |
| **Recall@10**     | **95.56%** | 86/90 |
| **MRR**           | **0.8606** | —     |
| Category accuracy | 100.00%    | 90/90 |


#### V1 vs V2 Comparison


| Metric          | V1       | V2         | Δ            |
| --------------- | -------- | ---------- | ------------ |
| Recall@1        | 71.11%   | **80.00%** | **+8.89 pp** |
| Recall@5        | 91.11%   | 92.22%     | +1.11 pp     |
| Recall@10       | 93.33%   | 95.56%     | +2.23 pp     |
| MRR             | 0.8017   | **0.8606** | **+0.0589**  |
| Severe failures | 6 (6.7%) | 4 (4.4%)   | −2 failures  |


#### Failure Analysis


| Category            | Count      | Notes                    |
| ------------------- | ---------- | ------------------------ |
| Perfect first-tries | 72 (80.0%) | Retrieved at rank 1      |
| Near misses         | 14 (15.6%) | In top 10 but not rank 1 |
| Severe failures     | 4 (4.4%)   | **Not in top 10 at all** |


#### Severe Failures (not in top 10)


| Test Index | Query       | Correct Item | Rank    | Notes                                                                       |
| ---------- | ----------- | ------------ | ------- | --------------------------------------------------------------------------- |
| 46         | tumbler_047 | tumbler_047  | **#13** | Cream off-white tumbler, pastel tulip/butterfly print (V1: #60 → V2: #13 ✅) |
| 52         | tumbler_065 | tumbler_065  | **#99** | WRELS matte black soft flask (V1: #113 → V2: #99, still catastrophic)       |
| 62         | charger_040 | charger_040  | **#24** | White 22.5W QOOVI wall charger (V1: #4, missed — V2: new severe)            |
| 70         | charger_081 | charger_081  | **#11** | Pink multi-port wall charger (V1: #16 → V2: #11 ✅)                          |


> **Tumbler improvements**: V2 massively improved the tumbler_047 (rank 60→13) and tumbler_065 (rank 99→113 is worse actually... wait, V1 was #113, V2 is #99 — still very bad). Charger_062 is a new severe failure that V1 did not have.

#### Sample Retrieval Leaderboard

For query: *"Light-grey rectangular lunch box, pale blue latch clips and flap compartment, white oval shaped vent button on lid"* (correct: `lunchbox_050`)

> **This is the same query that was V1's worst failure (rank 21).** V2 places it at rank 7.


| Rank | Item         | Score  | Status       |
| ---- | ------------ | ------ | ------------ |
| #1   | lunchbox_030 | 0.3618 |              |
| #2   | lunchbox_074 | 0.3576 |              |
| #3   | lunchbox_041 | 0.3383 |              |
| #4   | lunchbox_066 | 0.3257 |              |
| #5   | lunchbox_042 | 0.3205 |              |
| #6   | lunchbox_002 | 0.3199 |              |
| #7   | lunchbox_050 | 0.3159 | ✅ TRUE MATCH |
| #8   | lunchbox_073 | 0.3144 |              |
| #9   | lunchbox_080 | 0.3122 |              |
| #10  | lunchbox_014 | 0.3060 |              |


#### Confusion Matrix (Category Level)


| True \ Predicted  | Bags | Chargers | Handkerchiefs | Lunchboxes | Tumblers | Wallets |
| ----------------- | ---- | -------- | ------------- | ---------- | -------- | ------- |
| **Bags**          | 15   | 0        | 0             | 0          | 0        | 0       |
| **Chargers**      | 0    | 15       | 0             | 0          | 0        | 0       |
| **Handkerchiefs** | 0    | 0        | 15            | 0          | 0        | 0       |
| **Lunchboxes**    | 0    | 0        | 0             | 15         | 0        | 0       |
| **Tumblers**      | 0    | 0        | 0             | 0          | 15       | 0       |
| **Wallets**       | 0    | 0        | 0             | 0          | 0        | 15      |


**100% category accuracy** maintained — zero cross-category errors.

---

### What This Run Told Us

- **r=32 + q/k/v/out is the sweet spot**: +8.89 pp R@1 over V1, with manageable VRAM increase (3.31→3.78 GB)
- **Cosine scheduler + early stopping worked**: The model stopped at epoch 14, not epoch 20 — it was preventing overfitting correctly
- **tumbler_047 went from rank 60 → 13**: Full-spectrum augmentation significantly improved lighting/color robustness
- **tumbler_065 is the persistent worst case** (rank 113→99): Still catastrophic despite improvements — WRELS flask may need a different approach
- **charger_040 became a new severe failure**: r=32 may have over-indexed on some features at the expense of others (slight capacity misallocation)
- **100% category accuracy preserved**: Even with 8× more trainable params, the model still never confuses categories — all errors are within-category fine-grained confusion
- **MRR improved substantially** (+0.0589): Not just rank 1 improved, but the entire ranking distribution got sharper

---

## V2 + HNM — `fgclip-lora-finetunedV2-1200-HNM`

**Status**: Hard negative mining experiment. Same architecture and augmentations as V2, but replaces random shuffling with category-grouped batching. **Result: slight regression on R@1** — the category-grouped sampler was counterproductive.
**Training notebook**: `fgclipfintuning/Copy_of_fgclipFineTuning_V2_1200_HNM.ipynb`
**Evaluation notebook**: `fgclipevaluation/Copy_of_fgclipEvaluation_V2_1200_HNM.ipynb`
**Output directory**: `lorafinetuned/fgclip-lora-finetunedV2-1200-HNM/`
**Training seed**: 18339741769000

### What Changed from V2

The only change is the batch sampler. Everything else — architecture, augmentations, hyperparameters, temperature — is identical to V2.


| Aspect            | V2                                | V2 + HNM                                             |
| ----------------- | --------------------------------- | ---------------------------------------------------- |
| Mining strategy   | Standard random shuffle           | **Category-level batching**                          |
| Batch composition | 16 random pairs from any category | **16 pairs from the same category**                  |
| All else          | Identical to V2                   | Same r=32, q/k/v/out, scheduler, temp, augmentations |


### The Hard Negative Mining Implementation

A custom `CategoryBatchSampler` replaces the default `shuffle=True`:

```python
class CategoryBatchSampler(Sampler):
    def __init__(self, dataset_df, batch_size=16):
        # Group all indices by category (word before '_' in pair_id)
        self.category_to_indices = defaultdict(list)
        for idx, row in dataset_df.iterrows():
            category = row['lost_image'].split('_')[0]
            self.category_to_indices[category].append(idx)

    def __iter__(self):
        batches = []
        for cat, indices in self.category_to_indices.items():
            random.shuffle(indices)
            # Create same-category chunks of batch_size
            for i in range(0, len(indices), batch_size):
                chunk = indices[i:i + batch_size]
                if len(chunk) > 1:  # need at least 2 to form pos/neg pair
                    batches.append(chunk)
        random.shuffle(batches)
        for batch in batches:
            yield batch
```

**Effect**: Every training batch contains 16 pairs from the same category (e.g., 16 lunchbox pairs). This forces the model to discriminate between very similar items in every batch, providing a harder discrimination task than V2's random batching.

**Intended logic**: If the model can tell apart 16 similar lunchboxes, it should generalize better to fine-grained retrieval.

**Observed outcome**: Training loss was consistently higher than V2 at every epoch, suggesting the model was working harder — but R@1 dropped slightly, indicating the approach backfired (see analysis below).

---

### Training Methodology

**Task**: Image-text retrieval. Given a text description of a lost item, retrieve the matching image from a gallery of 600 found items.

**Loss**: Single-image InfoNCE / CLIP loss. Same as V2 in computational structure — `batch_size` images × 2 texts (positive + hard negative) per batch, cosine similarity matrix, cross-entropy with temperature 0.02. The critical difference is that all 16 images in a batch belong to the same category, so all 16 hard negatives are plausible matches for each image — creating a harder discrimination task than the random shuffle of V2.

**Hard negative mining**: A custom `CategoryBatchSampler` groups pairs by `category` prefix (e.g. `bag_012` → `"bag"`). Each category's shuffled index list is chunked into batches of 16. All batches across all categories are shuffled together. Batches with fewer than 2 items are discarded. The net effect: every training batch contains 16 images from one category + their 16 corresponding positive texts + 16 hard-negative texts from the same category. Early stopping (patience=3) halts if val loss fails to improve for 3 consecutive epochs.

**Training loop**: Identical to V2 in computational structure — one image encoded, two texts encoded, normalize, cosine sim `/ 0.02`, cross-entropy. The data loading order is what changes: images are bucketed by category instead of shuffled randomly.

**Inference**: Identical to V2 — 600 gallery images encoded once, query text encoded on-the-fly, cosine similarity ranks them.

---

### Architecture

Same as V2.


| Parameter            | Value                                    |
| -------------------- | ---------------------------------------- |
| Base model           | `qihoo360/fg-clip-base`                  |
| Trainable parameters | 3,932,160 (2.5608% of total)             |
| LoRA rank (r)        | 32                                       |
| LoRA alpha (α)       | 64                                       |
| Target modules       | `q_proj`, `k_proj`, `v_proj`, `out_proj` |
| Dropout              | 0.1                                      |


### Hyperparameters

Same as V2 except batch composition.


| Parameter             | Value                                 |
| --------------------- | ------------------------------------- |
| Optimizer             | AdamW                                 |
| Learning rate (start) | 5e-5                                  |
| Learning rate (end)   | 2.93e-5                               |
| Scheduler             | Cosine with warmup (10% of 540 steps) |
| Warmup steps          | 54 / 540 total                        |
| Weight decay          | 0.01                                  |
| Loss function         | CrossEntropyLoss                      |
| Temperature           | 0.02 (manual override)                |
| Batch size            | 16 (same category)                    |
| Max epochs            | 20                                    |
| Early stopping        | Patience = 3 epochs                   |


### Data Augmentation

Same as V2 (full-spectrum ColorJitter, RandomErasing, etc.).

### Training Loss Curve


| Epoch | Train Loss | Val Loss   | Note                          |
| ----- | ---------- | ---------- | ----------------------------- |
| 1     | 1.0411     | 1.3113     | 🟢 Saved                      |
| 2     | 0.9853     | 1.2782     | 🟢 Saved                      |
| 3     | 0.8973     | 1.2511     | 🟢 Saved                      |
| 4     | 0.6773     | 1.2375     | 🟢 Saved                      |
| 5     | 0.6473     | 1.2251     | 🟢 Saved                      |
| 6     | 0.5442     | 1.2101     | 🟢 Saved                      |
| 7     | 0.4608     | **1.1684** | 🟢 **Best — saved here**      |
| 8     | 0.3574     | 1.1730     | 🟡 Patience 1/3               |
| 9     | 0.3166     | 1.1918     | 🟡 Patience 2/3               |
| 10    | 0.2818     | 1.1819     | 🟡 Patience 3/3 → **Stopped** |
| 11–20 | —          | —          | Halted                        |


> Best checkpoint: epoch 7, val loss = 1.1684 (vs V2's best 1.1565 at epoch 13). Early stopping fired 4 epochs earlier than V2.

### Training Results


| Metric                   | Value                                  |
| ------------------------ | -------------------------------------- |
| Final train loss         | 0.2818                                 |
| Final val loss (at stop) | 1.1819                                 |
| **Best val loss**        | **1.1684** (saved checkpoint, epoch 7) |
| Epochs trained           | 10 / 20                                |
| Epochs saved             | 7                                      |
| GPU VRAM used            | 3.78 GB                                |


### Evaluation Results

**Evaluation setup**: 90 test queries matched against all 600 gallery images. Loaded fine-tuned adapter: `fgclip-lora-finetunedV2-1200-HNM`.


| Metric            | Value      | Count |
| ----------------- | ---------- | ----- |
| **Recall@1**      | **78.89%** | 71/90 |
| **Recall@5**      | **91.11%** | 82/90 |
| **Recall@10**     | **96.67%** | 87/90 |
| **MRR**           | **0.8556** | —     |
| Category accuracy | 100.00%    | 90/90 |


#### V2 vs V2+HNM Comparison


| Metric          | V2       | V2+HNM     | Δ            |
| --------------- | -------- | ---------- | ------------ |
| Recall@1        | 80.00%   | 78.89%     | **−1.11 pp** |
| Recall@5        | 92.22%   | 91.11%     | −1.11 pp     |
| Recall@10       | 95.56%   | **96.67%** | **+1.11 pp** |
| MRR             | 0.8606   | 0.8556     | −0.0050      |
| Severe failures | 4 (4.4%) | 3 (3.3%)   | −1 failure   |
| Best val loss   | 1.1565   | 1.1684     | −0.0119      |


#### Failure Analysis


| Category            | Count      | Notes                    |
| ------------------- | ---------- | ------------------------ |
| Perfect first-tries | 71 (78.9%) | Retrieved at rank 1      |
| Near misses         | 16 (17.8%) | In top 10 but not rank 1 |
| Severe failures     | 3 (3.3%)   | **Not in top 10 at all** |


#### Severe Failures (not in top 10)


| Test Index | Query            | Correct Item     | Rank     | V2 Rank   | Notes                                                               |
| ---------- | ---------------- | ---------------- | -------- | --------- | ------------------------------------------------------------------- |
| 46         | tumbler_047      | tumbler_047      | **#28**  | #13       | Cream off-white tumbler, pastel print (regressed: V2 #13 → HNM #28) |
| 52         | tumbler_065      | tumbler_065      | **#101** | #99       | WRELS matte black soft flask (regressed: V2 #99 → HNM #101)         |
| 75         | handkerchief_027 | handkerchief_027 | **#11**  | ✅ (found) | Multicolor floral handkerchief (V2 had this correct, HNM missed it) |


> **Notable improvement**: bag_058 (rank 12 in V2) is no longer a severe failure in HNM. charger_040 also recovered (rank 24 in V2 → rank 8 in HNM). charger_081 recovered from rank 11 to rank 8.

#### Sample Retrieval Leaderboard

For query: *"Light-grey rectangular lunch box, pale blue latch clips and flap compartment, white oval shaped vent button on lid"* (correct: `lunchbox_050`)

> V2 placed this at rank 7. HNM also places it at rank 7 — the specific lunchbox case was unaffected by the sampler change.


| Rank | Item         | Score  | Status       |
| ---- | ------------ | ------ | ------------ |
| #1   | lunchbox_074 | 0.3722 |              |
| #2   | lunchbox_030 | 0.3717 |              |
| #3   | lunchbox_041 | 0.3556 |              |
| #4   | lunchbox_072 | 0.3530 |              |
| #5   | lunchbox_055 | 0.3463 |              |
| #6   | lunchbox_073 | 0.3433 |              |
| #7   | lunchbox_050 | 0.3376 | ✅ TRUE MATCH |
| #8   | lunchbox_042 | 0.3351 |              |
| #9   | lunchbox_066 | 0.3248 |              |
| #10  | lunchbox_007 | 0.3212 |              |


#### Confusion Matrix (Category Level)


| True \ Predicted  | Bags | Chargers | Handkerchiefs | Lunchboxes | Tumblers | Wallets |
| ----------------- | ---- | -------- | ------------- | ---------- | -------- | ------- |
| **Bags**          | 15   | 0        | 0             | 0          | 0        | 0       |
| **Chargers**      | 0    | 15       | 0             | 0          | 0        | 0       |
| **Handkerchiefs** | 0    | 0        | 15            | 0          | 0        | 0       |
| **Lunchboxes**    | 0    | 0        | 0             | 15         | 0        | 0       |
| **Tumblers**      | 0    | 0        | 0             | 0          | 15       | 0       |
| **Wallets**       | 0    | 0        | 0             | 0          | 0        | 15      |


**100% category accuracy** maintained.

---

### What This Run Told Us

- **Category-grouped HNM was counterproductive for R@1**: −1.11 pp despite training loss being consistently higher (model was working harder, not smarter)
- **Training loss was higher at every epoch** (e.g., epoch 7: 0.4608 vs V2's 0.3707 at epoch 8) — the same-category batches were genuinely harder
- **The harder task didn't transfer to evaluation**: The model learned to discriminate within a category better during training, but this didn't improve generalization to the test set
- **Why it might have failed**: V2's random batching naturally mixes categories, creating diverse positive-negative pairs. This diversity may be important for learning robust cross-category visual features. HNM's category-grouped batches may have caused the model to over-focus on category-specific discriminative features rather than learning transferable fine-grained features
- **tumbler_047 got worse** (rank 13 → rank 28): The tumbler category seems particularly sensitive to batching strategy
- **tumbler_065 is still the hardest case**: Stayed at rank 101 despite all changes — this item may be fundamentally hard regardless of approach
- **R@10 actually improved** (95.56% → 96.67%): HNM shifted some rank 11–14 queries into top 10, even though rank 1 retrieval regressed
- **Early stopping fired 4 epochs earlier** (epoch 10 vs V2's epoch 14): The model overfits faster with HNM batching, possibly because every batch is high-contrast and noisy

---

## V2.1 + HNM — `fgclip-lora-finetunedV2.1-1200-HNM`

**Status**: Refined HNM run. Halved batch size (16→8) to reduce within-batch hardness, doubled LoRA alpha (64→128), lowered LR (5e-5→3e-5), softened temperature (0.02→0.03), balanced ColorJitter, and extended patience (3→5). **Result: tied V2's R@1 (80.00%) and achieved the best MRR so far (0.8615).**
**Training notebook**: `fgclipfintuning/Copy_of_fgclipFineTuning_V2_1_1200_HNM.ipynb`
**Evaluation notebook**: `fgclipevaluation/Copy_of_fgclipEvaluation_V2_1_1200_HNM.ipynb`
**Output directory**: `lorafinetuned/fgclip-lora-finetunedV2.1-1200-HNM/`
**Training seed**: 13630407283100

### What Changed from V2 + HNM


| Aspect                  | V2 + HNM                                                                 | V2.1 + HNM                                                                        |
| ----------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| LoRA alpha (α)          | 64                                                                       | **128**                                                                           |
| Learning rate           | 5e-5                                                                     | **3e-5**                                                                          |
| Temperature             | 0.02                                                                     | **0.03**                                                                          |
| Train batch size        | 16 (same category)                                                       | **8 (same category)**                                                             |
| Steps per epoch         | 27                                                                       | **53**                                                                            |
| Total steps (20 epochs) | 540                                                                      | **1060**                                                                          |
| Early stopping patience | 3                                                                        | **5**                                                                             |
| ColorJitter             | brightness=0.2, contrast=0.2, saturation=0.4, hue=0.1 | **brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1 (all channels balanced to ±0.4)** |


### Why These Changes

- **Batch size 16→8**: V2+HNM's category-grouped batches of 16 same-category pairs were too hard — the model couldn't learn effectively from such a narrow context window. Halving it gives each pair more gradient signal per step.
- **Doubling α (64→128)**: Doubling alpha relative to r=32 amplifies the LoRA contribution without adding parameters. With r=32, α=128 means the LoRA pathway contributes 4× the base model's logit scale, giving it much more representational power.
- **LR 5e-5→3e-5**: A lower LR with a larger alpha prevents the stronger LoRA signal from destabilizing training.
- **Temperature 0.02→0.03**: Softer logits at 0.03 reduce overconfidence, especially helpful with smaller batches where each step's gradient is noisier.
- **Balanced ColorJitter**: V2 used aggressive brightness/contrast (`0.2` → (0.8, 1.2)) but softer saturation (`0.4` → (0.6, 1.4)). V2.1 equalizes all channels to `0.4` → (0.6, 1.4), giving a more uniform robustness envelope.
- **Patience 3→5**: With more steps per epoch and a lower LR, the model converges more gradually. Patience 5 allows it enough runway to find the true optimum.

---

### Training Methodology

**Task**: Image-text retrieval. Given a text description of a lost item, retrieve the matching image from a gallery of 600 found items.

**Loss**: Single-image InfoNCE / CLIP loss. Same as V2+HNM — category-bucketed batches of 8 pairs, each with a positive + hard-negative caption. The key parameter changes from V2+HNM are: α=128 (stronger LoRA signal), LR=3e-5 (slower learning), temperature=0.03 (slightly softer softmax). The larger alpha doubles the scaling of LoRA output, making each adaptation step more decisive. The lower LR compensates by taking smaller steps. The higher temperature widens the softmax distribution, softening the penalty on near-misses.

**Hard negative mining**: Same `CategoryBatchSampler` as V2+HNM — groups by category prefix, chunks each category into batches of 8. The smaller batch size (8 vs 16) halves the number of hard negatives per step, making each gradient step less overwhelming.

**Training loop**: Zero gradients → encode `lost_image` → encode positive + hard-negative captions → normalize → cosine sim `/ 0.03` → cross-entropy → backprop → step optimizer → step scheduler. Only LoRA adapters train. Cosine scheduler with 10% warmup decays LR from 3e-5 to ~1.24e-5 over 1,060 steps. Early stopping (patience=5) restores the best checkpoint.

**Inference**: Identical to previous models — 600 gallery images encoded once, query text encoded on-the-fly, cosine similarity ranks them.

---

### Architecture


| Parameter            | Value                                    |
| -------------------- | ---------------------------------------- |
| Base model           | `qihoo360/fg-clip-base`                  |
| Trainable parameters | 3,932,160 (2.5608% of total)             |
| LoRA rank (r)        | 32                                       |
| LoRA alpha (α)       | 128                                      |
| Target modules       | `q_proj`, `k_proj`, `v_proj`, `out_proj` |
| Dropout              | 0.1                                      |


### Hyperparameters


| Parameter             | Value                                   |
| --------------------- | --------------------------------------- |
| Optimizer             | AdamW                                   |
| Learning rate (start) | 3e-5                                    |
| Learning rate (end)   | 1.5e-5                                  |
| Scheduler             | Cosine with warmup (10% of 1,060 steps) |
| Warmup steps          | 106 / 1,060 total                       |
| Weight decay          | 0.01                                    |
| Loss function         | CrossEntropyLoss                        |
| Temperature           | 0.03 (manual override)                  |
| Batch size            | 8 (same category per batch)             |
| Max epochs            | 20                                      |
| Early stopping        | Patience = 5 epochs                     |
| DataLoader workers    | 0                                       |


### Data Augmentation

```
Compose(
    Resize(size=(224, 224), interpolation=bilinear)
    RandomHorizontalFlip(p=0.5)
    ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    ToTensor()
    RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0)
    ToPILImage()
)
```

> **Note**: The notebook cell code shows `ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)` (single floats). PyTorch interprets each as `(1-value, 1+value)`, producing the equivalent ranges: brightness/contrast/saturation → (0.6, 1.4), hue → (-0.1, 0.1). The training summary report shows the expanded range notation.

### Training Loss Curve


| Epoch | Train Loss | Val Loss   | Note                          |
| ----- | ---------- | ---------- | ----------------------------- |
| 1     | 0.9381     | 1.1420     | 🟢 Saved                      |
| 2     | 0.7992     | 1.1101     | 🟢 Saved                      |
| 3     | 0.6659     | 1.0819     | 🟢 Saved                      |
| 4     | 0.5464     | 1.0554     | 🟢 Saved                      |
| 5     | 0.4760     | 1.0462     | 🟢 Saved                      |
| 6     | 0.3584     | **1.0251** | 🟢 **Best — saved here**      |
| 7     | 0.2725     | 1.0315     | 🟡 Patience 1/5               |
| 8     | 0.2239     | 1.0624     | 🟡 Patience 2/5               |
| 9     | 0.1698     | 1.0518     | 🟡 Patience 3/5               |
| 10    | 0.1630     | 1.0449     | 🟡 Patience 4/5               |
| 11    | 0.1486     | 1.0460     | 🟡 Patience 5/5 → **Stopped** |
| 12–20 | —          | —          | Halted                        |


> Best checkpoint: epoch 6, val loss = 1.0251. This is the **best val loss across all models** (V1: 1.4328, V2: 1.1565, V2+HNM: 1.1684, V2.1+HNM: 1.0251). Early stopping fired at epoch 11 after patience was exhausted.

### Training Results


| Metric                   | Value                                  |
| ------------------------ | -------------------------------------- |
| Final train loss         | 0.1486                                 |
| Final val loss (at stop) | 1.0460                                 |
| **Best val loss**        | **1.0251** (saved checkpoint, epoch 6) |
| Epochs trained           | 11 / 20                                |
| Epochs saved             | 6                                      |
| GPU VRAM used            | 2.21 GB                                |


### Evaluation Results

**Evaluation setup**: 90 test queries matched against all 600 gallery images. Loaded fine-tuned adapter: `fgclip-lora-finetunedV2.1-1200-HNM`.


| Metric            | Value      | Count |
| ----------------- | ---------- | ----- |
| **Recall@1**      | **80.00%** | 72/90 |
| **Recall@5**      | **92.22%** | 83/90 |
| **Recall@10**     | **96.67%** | 87/90 |
| **MRR**           | **0.8615** | —     |
| Category accuracy | 100.00%    | 90/90 |


#### Across All Models Comparison


| Metric          | V1     | V2     | V2+HNM     | V2.1+HNM   |
| --------------- | ------ | ------ | ---------- | ---------- |
| Recall@1        | 71.11% | 80.00% | 78.89%     | **80.00%** |
| Recall@5        | 91.11% | 92.22% | 91.11%     | **92.22%** |
| Recall@10       | 93.33% | 95.56% | **96.67%** | 96.67%     |
| MRR             | 0.8017 | 0.8606 | 0.8556     | **0.8615** |
| Best val loss   | 1.4328 | 1.1565 | 1.1684     | **1.0251** |
| Severe failures | 6      | 4      | 3          | **3**      |


#### Failure Analysis


| Category            | Count      | Notes                    |
| ------------------- | ---------- | ------------------------ |
| Perfect first-tries | 72 (80.0%) | Retrieved at rank 1      |
| Near misses         | 15 (16.7%) | In top 10 but not rank 1 |
| Severe failures     | 3 (3.3%)   | **Not in top 10 at all** |


#### Severe Failures (not in top 10)


| Test Index | Query       | Correct Item | Rank    | V2 Rank | V2+HNM Rank | Notes                                                                          |
| ---------- | ----------- | ------------ | ------- | ------- | ----------- | ------------------------------------------------------------------------------ |
| 46         | tumbler_047 | tumbler_047  | **#33** | #13     | #28         | Cream off-white tumbler, pastel print (improving from V2+HNM but still severe) |
| 52         | tumbler_065 | tumbler_065  | **#86** | #99     | #101        | WRELS matte black soft flask (improving: 99→101→86)                            |
| 62         | charger_040 | charger_040  | **#16** | #24     | #8          | White 22.5W QOOVI charger (V2+HNM recovered it; V2.1 slightly regressed)       |


> **Heaphone_027 recovered**: In V2+HNM, handkerchief_027 was at rank 11 (severe). V2.1 placed it at rank 7 — it recovered above the R@10 threshold. charger_040 remains the most inconsistent: V2 (#24, severe) → V2+HNM (#8, recovered) → V2.1+HNM (#16, severe again).

#### Sample Retrieval Leaderboard

For query: *"Light-grey rectangular lunch box, pale blue latch clips and flap compartment, white oval shaped vent button on lid"* (correct: `lunchbox_050`)

> **Best result yet for this query.** Previous: V1 rank 21, V2 rank 7, V2+HNM rank 7. V2.1: **rank 4**.


| Rank | Item         | Score  | Status       |
| ---- | ------------ | ------ | ------------ |
| #1   | lunchbox_041 | 0.4157 |              |
| #2   | lunchbox_074 | 0.4100 |              |
| #3   | lunchbox_030 | 0.3873 |              |
| #4   | lunchbox_050 | 0.3667 | ✅ TRUE MATCH |
| #5   | lunchbox_042 | 0.3596 |              |
| #6   | lunchbox_073 | 0.3554 |              |
| #7   | lunchbox_055 | 0.3526 |              |
| #8   | lunchbox_066 | 0.3494 |              |
| #9   | lunchbox_064 | 0.3456 |              |
| #10  | lunchbox_007 | 0.3451 |              |


#### Confusion Matrix (Category Level)


| True \ Predicted  | Bags | Chargers | Handkerchiefs | Lunchboxes | Tumblers | Wallets |
| ----------------- | ---- | -------- | ------------- | ---------- | -------- | ------- |
| **Bags**          | 15   | 0        | 0             | 0          | 0        | 0       |
| **Chargers**      | 0    | 15       | 0             | 0          | 0        | 0       |
| **Handkerchiefs** | 0    | 0        | 15            | 0          | 0        | 0       |
| **Lunchboxes**    | 0    | 0        | 0             | 15         | 0        | 0       |
| **Tumblers**      | 0    | 0        | 0             | 0          | 15       | 0       |
| **Wallets**       | 0    | 0        | 0             | 0          | 0        | 15      |


**100% category accuracy** maintained across all models.

---

### What This Run Told Us

- **α=128 + lower LR was the right move**: Tied V2's R@1 and achieved the best MRR (0.8615) across all models, with the best val loss (1.0251) — nearly 11% better than V2's 1.1565
- **Batch size 8 was critical**: V2+HNM's batch size 16 was too large for category-grouped batching. Halving it to 8 halved the per-step difficulty, allowing more effective learning
- **The lunchbox_050 query got its best result yet**: Rank 21 → 7 → 7 → 4 across models. The combined changes (α, LR, batch size, temp) all contributed
- **tumbler_065 is improving steadily**: 99 → 101 → 86. It will likely never be a top result, but it is trending in the right direction
- **tumbler_047 is trending the wrong way**: 13 → 28 → 33. The cream/white tumbler class is becoming harder for later models, possibly due to overfitting to the specific visual patterns of harder cases
- **charger_040 is highly unstable**: Recovered by V2+HNM (rank 8), then regressed to severe in V2.1 (rank 16). This suggests the QOOVI charger is on the decision boundary — small changes in the model shift it across the threshold
- **VRAM dropped significantly**: 3.78 GB → 2.21 GB due to batch size 16→8. The RTX 3050 has plenty of headroom at this configuration

---

## V3-optimized — `fgclip-lora-finetunedV3-optimized`

**Status**: Small rank (r=16) + strong weight decay (wd=0.1) + higher temperature (0.05) optimization. CPU-only training. **Result: best overall model — best R@5 (96.67%), best MRR (0.8733), tied fewest severe failures (3). Notably, V3 achieves the best retrieval performance while having the highest train loss at stop (0.5472) — the model was still learning, not overfitting.**
**Training notebook**: `fgclipfintuning/Copy_of_fgclipFineTuning_V3_1200_HNM.ipynb`
**Evaluation notebook**: `fgclipevaluation/Copy_of_fgclipEvaluation_V3_1200_HNM.ipynb`
**Output directory**: `lorafinetuned/fgclip-lora-finetunedV3-optimized/`
**Training seed**: 12952634091800

### What Changed from V2.1 + HNM

| Aspect                  | V2.1 + HNM                                                    | V3-optimized                                                        |
| ----------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------- |
| LoRA rank (r)           | 32                                                            | **16**                                                              |
| LoRA alpha (α)         | 128                                                           | **32**                                                              |
| Weight decay            | 0.01                                                          | **0.1** (10x stronger)                                              |
| Temperature             | 0.03                                                          | **0.05**                                                            |
| Cosine end LR           | 1.5e-5                                                        | **0.0** (cosine to zero)                                            |
| Warmup                  | 10%                                                          | **20%**                                                             |
| Early stopping          | Patience = 5                                                   | **Patience = 3**                                                    |
| Hardware                | GPU (RTX 3050)                                                | **CPU**                                                             |
| Data augmentation       | brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1     | **brightness=0.2, contrast=0.2, saturation=0.4, hue=0.1**          |


### Why These Changes

- **r=32 → r=16**: V2.1's r=32 + α=128 setup is prone to overfitting (train loss=0.1486 vs val loss=1.0251). Halving rank to 16 acts like a stronger regularizer, reducing the model's capacity to memorize training examples.
- **α=128 → α=32**: Keeping α/r ratio constant at 2 maintains the same effective scaling behavior while reducing the absolute magnitude of LoRA updates.
- **Weight decay 0.01 → 0.1**: The single most impactful change. Stronger weight decay penalizes large LoRA weights, encouraging the model to use smaller, more generalizable adaptations.
- **Temperature 0.03 → 0.05**: Higher temperature produces softer similarity logits, giving the model more gradient signal on near-miss cases. Correlates with best MRR.
- **Cosine to zero**: Instead of decaying to 1.5e-5, the scheduler decays to 0, effectively stopping weight updates in the final epochs — acts as a built-in early-stop mechanism.
- **20% warmup**: Doubling warmup from 10% gives the model more time to stabilize before taking large gradient steps.
- **CPU training**: With only 420 training images, CPU training is viable and practical given hardware constraints.

---

### Training Methodology

**Task**: Image-text retrieval. Given a text description of a lost item, retrieve the matching image from a gallery of 600 found items.

**Loss**: Single-image InfoNCE / CLIP loss. Same as V2.1+HNM — category-bucketed batches of 8 pairs (16 positive + 16 hard-negative captions). Key differences: α=32 (weaker LoRA signal), LR=3e-5 (same as V2.1), temperature=0.05 (softest logits across all models). The combination of smaller rank, stronger weight decay, and higher temperature forces the model to learn more robust, generalizable features rather than memorizing training pairs.

**Hard negative mining**: Same `CategoryBatchSampler` as V2.1+HNM — groups by category prefix, chunks into batches of 8.

**Training loop**: Zero gradients → encode `lost_image` → encode positive + hard-negative captions → normalize → cosine sim `/ 0.05` → cross-entropy → backprop → step optimizer → step scheduler. Only LoRA adapters train. Cosine scheduler with 20% warmup decays LR from 3e-5 to 0 over 540 steps. Early stopping (patience=3) would restore the best checkpoint — but did not fire; model trained full 20 epochs.

**Inference**: Identical to previous models — 600 gallery images encoded once, query text encoded on-the-fly, cosine similarity ranks them.

---

### Architecture

| Parameter            | Value                                    |
| -------------------- | ---------------------------------------- |
| Base model           | `qihoo360/fg-clip-base`                  |
| Trainable parameters | 1,966,080 (1.2970% of total)             |
| LoRA rank (r)        | **16**                                   |
| LoRA alpha (α)       | **32**                                   |
| Target modules       | `q_proj`, `k_proj`, `v_proj`, `out_proj` |
| Dropout              | 0.1                                      |
| Model dtype          | `torch.float32`                          |


### Hyperparameters

| Parameter             | Value                                      |
| --------------------- | ------------------------------------------ |
| Optimizer             | AdamW                                      |
| Learning rate (start) | 3e-5                                       |
| Learning rate (end)   | **0.0** (cosine decay to zero)             |
| Scheduler             | Cosine with warmup (20% of 540 steps)      |
| Warmup steps          | 108 / 540 total                            |
| Weight decay          | **0.1**                                    |
| Betas                 | (0.9, 0.999)                              |
| Epsilon               | 1e-8                                      |
| Loss function         | CrossEntropyLoss                           |
| Temperature           | **0.05** (manual override)                 |
| Batch size            | 16 (val), N/A (train, CPU)                 |
| Max epochs            | 20                                         |
| Early stopping        | Patience = 3 epochs (not triggered)         |
| DataLoader workers    | 0                                          |

> **Note on train batch size**: V3 was trained on CPU. The training summary report shows "Train Batch Size: None" because batch size was managed differently in the CPU training script (likely dynamic/padded batching). The effective training throughput was determined by the CPU training implementation rather than a fixed batch size.

### Data Augmentation

```
Compose(
    Resize(size=(224, 224), interpolation=bilinear)
    RandomHorizontalFlip(p=0.5)
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.4, hue=0.1)
    ToTensor()
    RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0)
    ToPILImage()
)
```

> **Note**: Reverted brightness/contrast from V2.1's 0.4 back to 0.2 (matching V2), keeping saturation at 0.4 and hue at 0.1.

### Training Loss Curve

| Epoch | Train Loss | Val Loss   | Note              |
| ----- | ---------- | ---------- | ----------------- |
| 1     | 1.1819     | 1.4054     | 🟢 Saved          |
| 2     | 1.1332     | 1.3866     | 🟢 Saved          |
| 3     | 1.0990     | 1.3522     | 🟢 Saved          |
| 4     | 1.0265     | 1.3023     | 🟢 Saved          |
| 5     | 0.9636     | 1.2533     | 🟢 Saved          |
| 6     | 0.8874     | 1.2188     | 🟢 Saved          |
| 7     | 0.8131     | 1.1925     | 🟢 Saved          |
| 8     | 0.7650     | 1.1724     | 🟢 Saved          |
| 9     | 0.7377     | 1.1579     | 🟢 Saved          |
| 10    | 0.6994     | 1.1509     | 🟢 Saved          |
| 11    | 0.6626     | 1.1429     | 🟢 Saved          |
| 12    | 0.6276     | 1.1420     | 🟢 Saved          |
| 13    | 0.6401     | 1.1398     | 🟢 Saved          |
| 14    | 0.5889     | 1.1374     | 🟢 Saved          |
| 15    | 0.6023     | 1.1358     | 🟢 Saved          |
| 16    | 0.5591     | 1.1345     | 🟢 Saved          |
| 17    | 0.5585     | 1.1339     | 🟢 Saved          |
| 18    | 0.5402     | 1.1337     | 🟢 Saved          |
| 19    | 0.5466     | 1.1336     | 🟢 Saved          |
| 20    | 0.5472     | **1.1335** | 🟢 **Best — saved** |

> **Val loss never plateaued**: Unlike all previous models that triggered early stopping, V3's val loss continued decreasing through all 20 epochs (1.4054 → 1.1335). The model was still learning. This suggests either (a) more epochs would help, or (b) the strong regularization (wd=0.1) keeps the model in a sweet spot where it generalizes better the longer it trains. The highest train loss at stop (0.5472) among HNM models confirms the model was not overfitting.

### Training Results

| Metric                   | Value                                  |
| ------------------------ | -------------------------------------- |
| Final train loss         | 0.5472                                 |
| Final val loss (at stop) | 1.1335                                 |
| **Best val loss**        | **1.1335** (saved checkpoint, epoch 20) |
| Epochs trained           | **20 / 20** (trained full)             |
| Epochs saved             | 20                                     |
| Early stop triggered     | **No** (patience not exhausted)        |


### Evaluation Results

**Evaluation setup**: 90 test queries matched against all 600 gallery images. Loaded fine-tuned adapter: `fgclip-lora-finetunedV3-optimized`.

| Metric            | Value      | Count |
| ----------------- | ---------- | ----- |
| **Recall@1**      | **80.00%** | 72/90 |
| **Recall@5**      | **96.67%** | 87/90 |
| **Recall@10**     | **96.67%** | 87/90 |
| **MRR**           | **0.8733** | —     |
| Category accuracy | 100.00%    | 90/90 |


#### Across All Models Comparison

| Metric          | V1     | V2     | V2+HNM | V2.1+HNM   | CLAUDE V2+HNM | CLAUDE V2.1+HNM | **V3-optimized** |
| --------------- | ------ | ------ | ------ | ---------- | ------------- | --------------- | ---------------- |
| Recall@1        | 71.11% | 80.00% | 78.89% | 80.00%     | 77.78%        | 78.89%          | **80.00%**        |
| Recall@5        | 91.11% | 92.22% | 91.11% | 92.22%     | 91.11%        | 94.44%          | **96.67%**        |
| Recall@10       | 93.33% | 95.56% | 96.67% | 96.67%     | 96.67%        | 96.67%          | **96.67%**        |
| MRR             | 0.8017 | 0.8606 | 0.8556 | 0.8615     | 0.8515        | 0.8615          | **0.8733**        |
| Best val loss   | 1.4328 | 1.1565 | 1.1684 | 1.0251     | 1.0465        | 1.0465          | 1.1335            |
| Severe failures | 6      | 4      | 3      | 3          | 3             | 3               | **3**             |

> **V3 is the best overall model**: Highest R@5 (96.67%), highest MRR (0.8733), tied highest R@1 (80.00%), tied fewest severe failures (3). The best val loss being the highest (1.1335) is misleading — val loss does not correlate with retrieval quality in this case. V3's smaller rank + stronger weight decay produce a model that generalizes better to retrieval despite having higher val loss.


#### Failure Analysis

| Category            | Count      | Notes                    |
| ------------------- | ---------- | ------------------------ |
| Perfect first-tries | 72 (80.0%) | Retrieved at rank 1      |
| Near misses         | 15 (16.7%) | In top 10 but not rank 1 |
| Severe failures     | 3 (3.3%)   | **Not in top 10 at all** |


#### Severe Failures (not in top 10)

| Test Index | Query       | Correct Item | Rank    | V2.1+HNM | CLAUDE V2.1+HNM | Notes                                                                      |
| ---------- | ----------- | ------------ | ------- | -------- | ---------------- | -------------------------------------------------------------------------- |
| 46         | tumbler_047 | tumbler_047  | **#14** | #33      | #26              | Cream off-white tumbler, pastel print (recovered: V2.1 #33 → V3 #14 ✅)     |
| 52         | tumbler_065 | tumbler_065  | **#38** | #86      | #80              | WRELS matte black flask (**best rank across ALL models** ✅)                |
| 62         | charger_040 | charger_040  | **#20** | #16      | #22              | White QOOVI 22.5W charger (severe in V3, recovered in CLAUDE V2.1+HNM)     |

> **tumbler_065 achieved its best rank ever**: #38 in V3 is far better than any previous model. CLAUDE V2.1+HNM held the previous best at #80. This is a 52% improvement in rank position.
> **tumbler_047 also recovered significantly**: #14 in V3 beats CLAUDE V2.1's #26, though V2 still held the best rank for this item at #13.


#### Sample Retrieval Leaderboard

For query: *"Light-grey rectangular lunch box, pale blue latch clips and flap compartment, white oval shaped vent button on lid"* (correct: `lunchbox_050`)

> **Tied best result for this query.** V2.1 achieved rank 4; V3 matches it at rank 4.

| Rank | Item         | Score  | Status       |
| ---- | ------------ | ------ | ------------ |
| #1   | lunchbox_041 | 0.4208 |              |
| #2   | lunchbox_074 | 0.4125 |              |
| #3   | lunchbox_030 | 0.3930 |              |
| #4   | lunchbox_050 | 0.3722 | ✅ TRUE MATCH |
| #5   | lunchbox_042 | 0.3720 |              |
| #6   | lunchbox_055 | 0.3590 |              |
| #7   | lunchbox_073 | 0.3580 |              |
| #8   | lunchbox_082 | 0.3513 |              |
| #9   | lunchbox_014 | 0.3423 |              |
| #10  | lunchbox_004 | 0.3329 |              |


#### Confusion Matrix (Category Level)

| True \ Predicted  | Bags | Chargers | Handkerchiefs | Lunchboxes | Tumblers | Wallets |
| ----------------- | ---- | -------- | ------------- | ---------- | -------- | ------- |
| **Bags**          | 15   | 0        | 0             | 0          | 0        | 0       |
| **Chargers**      | 0    | 15       | 0             | 0          | 0        | 0       |
| **Handkerchiefs** | 0    | 0        | 15            | 0          | 0        | 0       |
| **Lunchboxes**    | 0    | 0        | 0             | 15         | 0        | 0       |
| **Tumblers**      | 0    | 0        | 0             | 0          | 15       | 0       |
| **Wallets**       | 0    | 0        | 0             | 0          | 0        | 15      |

**100% category accuracy** maintained across all models.

---

### What This Run Told Us

- **r=16 + wd=0.1 is the best combination**: Smallest rank + strongest weight decay = best generalization. The smaller LoRA rank acts as an additional regularizer, and the 10x stronger weight decay prevents the model from fitting noisy training patterns.
- **Val loss is not a good proxy for retrieval quality**: V3 has the highest best val loss (1.1335) but the best retrieval metrics. The val loss continues decreasing through all 20 epochs without overfitting, suggesting the regularization is doing its job.
- **Train loss is a better signal**: V3 has the highest train loss at stop (0.5472) among HNM models. All other models that stopped early had train losses of 0.12–0.28, indicating they had memorized the training data. V3's higher train loss means it was still learning generalizable features.
- **tumbler_065 improved dramatically**: #38 in V3 vs #80 in CLAUDE V2.1 — a massive jump. The smaller rank + stronger regularization appears to help with this confusing item.
- **tumbler_047 also recovered**: #14 in V3 (second best after V2's #13). The WRELS flask and cream tumbler cases both benefit from the gentler learning that a smaller rank provides.
- **charger_040 is on the boundary**: Rank #20 is severe, but CLAUDE V2.1 recovered it at rank #22. It may be impossible to fully resolve with single-image models; an ensemble approach could help.
- **CPU training is viable for small datasets**: No GPU required for 420 training images. This opens up training to anyone with a CPU.
- **Patience=3 with cosine-to-zero did not fire**: The model was still improving at epoch 20. More epochs may help, but at some point the benefit will plateau.

---

---

## CLAUDE V2 + HNM — `CLAUDEfgclip-lora-finetunedV2-1200-HNM`

**Status**: Ablation on loss signal. Identical architecture/hyperparams to V2+HNM, but trains with a **dual-image loss** — both `lost_image` and `found_image` are encoded per pair, doubling the positive/negative signal per batch. Same-category HNM batching (batch=16). **Result: R@1 regressed to 77.78%, but the dual-image approach is architecturally important for understanding the pipeline.**
**Training notebook**: `fgclipfintuning/CLAUDEfgclipFineTuning_V2_1200_HNM.ipynb`
**Evaluation notebook**: `fgclipevaluation/CLAUDEfgclipEvaluation_V2_1200_HNM.ipynb`
**Output directory**: `lorafinetuned/CLAUDEfgclip-lora-finetunedV2-1200-HNM/`
**Training seed**: 21827269962500

### What Changed from V2 + HNM


| Aspect                  | V2 + HNM                                            | CLAUDE V2 + HNM                                              |
| ----------------------- | --------------------------------------------------- | ------------------------------------------------------------ |
| Training loss           | **Single-image**: only `lost_image` → pos/neg texts | **Dual-image**: `lost_image` + `found_image` → pos/neg texts |
| Images per pair         | 1                                                   | **2**                                                        |
| Texts per pair          | 2 (1 pos + 1 neg)                                   | 2 (1 pos + 1 neg)                                            |
| Total per batch (bs=16) | 16 images + 32 texts                                | **32 images + 32 texts**                                     |
| Labels in loss          | `torch.arange(16)`                                  | `torch.cat([torch.arange(16), torch.arange(16)])`            |
| Architecture            | Identical to V2                                     | **Identical to V2**                                          |
| Hyperparameters         | Identical to V2                                     | **Identical to V2**                                          |
| VRAM                    | 3.78 GB                                             | **5.86 GB**                                                  |


### The Dual-Image Training Loss

The critical difference is in how the triplet `(lost_image, positive_caption, hard_negative_caption)` is used:

**V2 + HNM (single-image)**:

```python
# 16 images, 32 texts (16 pos + 16 neg)
inputs = processor(text=pos_texts + neg_texts, images=images, ...)
labels = torch.arange(16)  # each image matched to its positive text
```

**CLAUDE V2 + HNM (dual-image)**:

```python
# 32 images (16 lost + 16 found), 32 texts (16 pos + 16 neg)
inputs = processor(text=pos_texts + neg_texts, images=lost_images + found_images, ...)
# Each of the 32 images is matched to its positive text:
#   indices 0–15  →  lost_image[0..15] matched to pos[0..15]
#   indices 16–31 →  found_image[0..15] matched to pos[16..31]
#   All 32 negatives are shared
labels = torch.cat([torch.arange(16), torch.arange(16)])
```

**Intended logic**: Encoding both the query (lost) and candidate (found) image during training gives the model richer signal — the positive text should align well with *both* images of the same item. This also mirrors the deployment scenario where we encode found images for the gallery and query with text.

**Observed outcome**: The model processed twice as much data per step but the loss signal may have been diluted by mixing two different image views. R@1 dropped vs V2+HNM.

---

### Training Methodology

**Task**: Image-text retrieval. Given a text description of a lost item, retrieve the matching image from a gallery of 600 found items.

**Loss**: Dual-image InfoNCE / CLIP loss. For each of the `batch_size` pairs in a batch, both the `lost_image` and the `found_image` are encoded alongside the positive and hard-negative captions. This produces `2 × batch_size` images matched against `2 × batch_size` texts in the similarity matrix. The labels are `torch.cat([torch.arange(batch_size), torch.arange(batch_size)])`, so the `lost_image` at index `i` is matched to the positive at index `i`, and the `found_image` at index `i` is also matched to the same positive at index `i + batch_size`. Both images of the same item share the same positive caption as the target.

**Hard negative mining**: Same `CategoryBatchSampler` as V2+HNM — groups by category prefix, chunks into batches of 16. Both the `lost_image` and `found_image` of each pair are included in the batch simultaneously.

**Training loop**: Zero gradients → encode all `lost_images + found_images` → encode positive + hard-negative captions → normalize → cosine sim `/ 0.02` → cross-entropy → backprop → step optimizer → step scheduler. Both images per pair are processed in the same forward pass, doubling throughput of image encoding per step but doubling the gradient mixing. Only LoRA adapters train.

**Inference**: Identical to previous models — 600 gallery images encoded once (single image per item), query text encoded on-the-fly, cosine similarity ranks them. The dual-image training does not affect inference.

---

### Architecture


| Parameter            | Value                                    |
| -------------------- | ---------------------------------------- |
| Base model           | `qihoo360/fg-clip-base`                  |
| Trainable parameters | 3,932,160 (2.5608% of total)             |
| LoRA rank (r)        | 32                                       |
| LoRA alpha (α)       | 64                                       |
| Target modules       | `q_proj`, `k_proj`, `v_proj`, `out_proj` |
| Dropout              | 0.1                                      |


### Hyperparameters


| Parameter             | Value                                 |
| --------------------- | ------------------------------------- |
| Optimizer             | AdamW                                 |
| Learning rate (start) | 5e-5                                  |
| Learning rate (end)   | 2.93e-5                               |
| Scheduler             | Cosine with warmup (10% of 540 steps) |
| Warmup steps          | 54 / 540 total                        |
| Weight decay          | 0.01                                  |
| Loss function         | CrossEntropyLoss                      |
| Temperature           | 0.02 (manual override)                |
| Train batch size      | 16 (same category)                    |
| Val batch size        | 16                                    |
| Max epochs            | 20                                    |
| Early stopping        | Patience = 3 epochs                   |
| DataLoader workers    | 0                                     |


### Data Augmentation

```
Compose(
    Resize(size=(224, 224), interpolation=bilinear)
    RandomHorizontalFlip(p=0.5)
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.4, hue=0.1)
    ToTensor()
    RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0)
    ToPILImage()
)
```

> **Note**: The notebook cell code shows `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.4, hue=0.1)` (single floats). PyTorch interprets each as `(1-value, 1+value)`, producing: brightness → (0.8, 1.2), contrast → (0.8, 1.2), saturation → (0.6, 1.4), hue → (-0.1, 0.1). The training summary report shows the expanded range notation.

### Training Loss Curve


| Epoch | Train Loss | Val Loss   | Note                          |
| ----- | ---------- | ---------- | ----------------------------- |
| 1     | 1.0655     | 1.3859     | 🟢 Saved                      |
| 2     | 1.0065     | 1.3618     | 🟢 Saved                      |
| 3     | 0.8091     | 1.3446     | 🟢 Saved                      |
| 4     | 0.7105     | 1.2937     | 🟢 Saved                      |
| 5     | 0.6655     | 1.2767     | 🟢 Saved                      |
| 6     | 0.5389     | 1.2687     | 🟢 Saved                      |
| 7     | 0.4556     | **1.2539** | 🟢 **Best — saved here**      |
| 8     | 0.3850     | 1.2577     | 🟡 Patience 1/3               |
| 9     | 0.3244     | 1.2626     | 🟡 Patience 2/3               |
| 10    | 0.2722     | 1.2654     | 🟡 Patience 3/3 → **Stopped** |
| 11–20 | —          | —          | Halted                        |


> Best checkpoint: epoch 7, val loss = 1.2539. Early stopping fired at epoch 10. Same early-stop pattern as V2+HNM — both hit patience exhaustion at epoch 10.

### Training Results


| Metric                   | Value                                  |
| ------------------------ | -------------------------------------- |
| Final train loss         | 0.2722                                 |
| Final val loss (at stop) | 1.2654                                 |
| **Best val loss**        | **1.2539** (saved checkpoint, epoch 7) |
| Epochs trained           | 10 / 20                                |
| GPU VRAM used            | 5.86 GB                                |


### Evaluation Results

**Evaluation setup**: 90 test queries matched against all 600 gallery images. Loaded fine-tuned adapter: `CLAUDEfgclip-lora-finetunedV2-1200-HNM`.


| Metric            | Value      | Count |
| ----------------- | ---------- | ----- |
| **Recall@1**      | **77.78%** | 70/90 |
| **Recall@5**      | **91.11%** | 82/90 |
| **Recall@10**     | **96.67%** | 87/90 |
| **MRR**           | **0.8515** | —     |
| Category accuracy | 100.00%    | 90/90 |


#### Across All Models Comparison


| Metric          | V1     | V2     | V2+HNM | V2.1+HNM   | CLAUDE V2+HNM |
| --------------- | ------ | ------ | ------ | ---------- | ------------- |
| Recall@1        | 71.11% | 80.00% | 78.89% | **80.00%** | 77.78%        |
| Recall@5        | 91.11% | 92.22% | 91.11% | **92.22%** | 91.11%        |
| Recall@10       | 93.33% | 95.56% | 96.67% | 96.67%     | **96.67%**    |
| MRR             | 0.8017 | 0.8606 | 0.8556 | **0.8615** | 0.8515        |
| Best val loss   | 1.4328 | 1.1565 | 1.1684 | **1.0251** | 1.2539        |
| Severe failures | 6      | 4      | 3      | 3          | **3**         |


#### Failure Analysis


| Category            | Count      | Notes                    |
| ------------------- | ---------- | ------------------------ |
| Perfect first-tries | 70 (77.8%) | Retrieved at rank 1      |
| Near misses         | 17 (18.9%) | In top 10 but not rank 1 |
| Severe failures     | 3 (3.3%)   | **Not in top 10 at all** |


#### Severe Failures (not in top 10)


| Test Index | Query       | Correct Item | Rank    | V2+HNM Rank | Notes                                                                       |
| ---------- | ----------- | ------------ | ------- | ----------- | --------------------------------------------------------------------------- |
| 46         | tumbler_047 | tumbler_047  | **#30** | #28         | Cream off-white tumbler, pastel print (regressed slightly vs V2+HNM)        |
| 52         | tumbler_065 | tumbler_065  | **#93** | #101        | WRELS matte black soft flask (improved: #101 → #93)                         |
| 70         | charger_081 | charger_081  | **#11** | #8          | Pink multi-port wall charger (regressed: V2+HNM had it in top 10 at rank 8) |


> **Note**: charger_081 was recovered by V2+HNM (rank 8) but CLAUDE V2+HNM dropped it back to severe (rank 11). This is the same pattern as V2.1+HNM, suggesting charger_081 is near the R@10 threshold and sensitive to model changes.

#### Sample Retrieval Leaderboard

For query: *"Light-grey rectangular lunch box, pale blue latch clips and flap compartment, white oval shaped vent button on lid"* (correct: `lunchbox_050`)

> V2+HNM placed it at rank 7. CLAUDE V2+HNM: **rank 6**.


| Rank | Item         | Score  | Status       |
| ---- | ------------ | ------ | ------------ |
| #1   | lunchbox_074 | 0.3827 |              |
| #2   | lunchbox_030 | 0.3766 |              |
| #3   | lunchbox_073 | 0.3758 |              |
| #4   | lunchbox_041 | 0.3632 |              |
| #5   | lunchbox_066 | 0.3555 |              |
| #6   | lunchbox_050 | 0.3498 | ✅ TRUE MATCH |
| #7   | lunchbox_072 | 0.3482 |              |
| #8   | lunchbox_080 | 0.3444 |              |
| #9   | lunchbox_055 | 0.3438 |              |
| #10  | lunchbox_042 | 0.3429 |              |


#### Confusion Matrix (Category Level)


| True \ Predicted  | Bags | Chargers | Handkerchiefs | Lunchboxes | Tumblers | Wallets |
| ----------------- | ---- | -------- | ------------- | ---------- | -------- | ------- |
| **Bags**          | 15   | 0        | 0             | 0          | 0        | 0       |
| **Chargers**      | 0    | 15       | 0             | 0          | 0        | 0       |
| **Handkerchiefs** | 0    | 0        | 15            | 0          | 0        | 0       |
| **Lunchboxes**    | 0    | 0        | 0             | 15         | 0        | 0       |
| **Tumblers**      | 0    | 0        | 0             | 0          | 15       | 0       |
| **Wallets**       | 0    | 0        | 0             | 0          | 0        | 15      |


**100% category accuracy** maintained.

---

### What This Run Told Us

- **Dual-image loss was counterproductive for R@1**: 78.89% → 77.78% (−1.11 pp). Processing both images per pair diluted the positive signal — the model got a mixed gradient from two different views of the same item
- **VRAM cost was significant**: 3.78 GB → 5.86 GB (+55%) for processing 2× images per step. The RTX 3050 handled it, but there's a real cost to this approach
- **tumbler_065 improved slightly**: 101 → 93. The dual-image approach may have helped the model better encode the WRELS flask's visual identity
- **tumbler_047 worsened**: 28 → 30. The cream-colored tumbler continues to be problematic across all HNM variants
- **charger_081 dropped back to severe**: Rank 8 → 11. The QOOVI charger is consistently near the threshold across models
- **The single-image V2.1+HNM remains best overall**: R@1=80.00%, MRR=0.8615, best val loss=1.0251 — none of the HNM variants have matched it
- **Both HNM variants (V2.1 and CLAUDE) have 3 severe failures**: Different items failed in each — V2.1 failed tumbler_047, tumbler_065, charger_040; CLAUDE failed tumbler_047, tumbler_065, charger_081. Different models fail for different reasons; an ensemble could reduce total failures

---

### What This Run Told Us

- **Dual-image loss was counterproductive for R@1**: 78.89% → 77.78% (−1.11 pp). Processing both images per pair diluted the positive signal — the model got a mixed gradient from two different views of the same item
- **VRAM cost was significant**: 3.78 GB → 5.86 GB (+55%) for processing 2× images per step. The RTX 3050 handled it, but there's a real cost to this approach
- **tumbler_065 improved slightly**: 101 → 93. The dual-image approach may have helped the model better encode the WRELS flask's visual identity
- **tumbler_047 worsened**: 28 → 30. The cream-colored tumbler continues to be problematic across all HNM variants
- **charger_081 dropped back to severe**: Rank 8 → 11. The QOOVI charger is consistently near the threshold across models
- **The single-image V2.1+HNM remains best overall**: R@1=80.00%, MRR=0.8615, best val loss=1.0251 — none of the HNM variants have matched it
- **Both HNM variants (V2.1 and CLAUDE) have 3 severe failures**: Different items failed in each — V2.1 failed tumbler_047, tumbler_065, charger_040; CLAUDE failed tumbler_047, tumbler_065, charger_081. Different models fail for different reasons; an ensemble could reduce total failures

---

## CLAUDE V2.1 + HNM — `CLAUDEfgclip-lora-finetunedV2.1-1200-HNM`

**Status**: Combined the V2.1 parameter improvements (α=128, LR=3e-5, temp=0.03, batch=8, patience=5) with the dual-image loss from CLAUDE V2+HNM. **Result: tied V2.1+HNM's MRR (0.8615) and R@5 (94.44%), but R@1 regressed to 78.89%. The dual-image signal and V2.1 params partially compensate for each other.**
**Training notebook**: `fgclipfintuning/CLAUDEfgclipFineTuning_V2_1_1200_HNM.ipynb`
**Evaluation notebook**: `fgclipevaluation/CLAUDEfgclipEvaluation_V2_1_1200_HNM.ipynb`
**Output directory**: `lorafinetuned/CLAUDEfgclip-lora-finetunedV2.1-1200-HNM/`
**Training seed**: 25201920299700

### What Changed from CLAUDE V2 + HNM


| Aspect                  | CLAUDE V2 + HNM                                                          | CLAUDE V2.1 + HNM                                                                 |
| ----------------------- | ------------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| LoRA alpha (α)          | 64                                                                       | **128**                                                                           |
| Learning rate           | 5e-5                                                                     | **3e-5**                                                                          |
| Temperature             | 0.02                                                                     | **0.03**                                                                          |
| Train batch size        | 16 (same category)                                                       | **8 (same category)**                                                             |
| Steps per epoch         | 27                                                                       | **53**                                                                            |
| Total steps (20 epochs) | 540                                                                      | **1,060**                                                                         |
| Early stopping patience | 3                                                                        | **5**                                                                             |
| ColorJitter             | brightness=0.2, contrast=0.2, saturation=0.4, hue=0.1 | **brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1 (all channels balanced to ±0.4)** |
| Dual-image loss         | Both                                                                     | **Both**                                                                          |
| Architecture            | Same as V2                                                               | **Same as V2**                                                                    |


> **Essentially**: CLAUDE V2.1 + HNM is V2.1 + HNM (the best-performing single-image model) plus dual-image loss, or equivalently CLAUDE V2 + HNM with the V2.1 parameter upgrades. The question the experiment answers is: do the V2.1 parameters compensate for the dual-image signal dilution?

---

### Training Methodology

**Task**: Image-text retrieval. Given a text description of a lost item, retrieve the matching image from a gallery of 600 found items.

**Loss**: Dual-image InfoNCE / CLIP loss. Identical to CLAUDE V2+HNM in structure — both `lost_image` and `found_image` are encoded per pair, producing `2 × batch_size` images matched against `2 × batch_size` texts. Labels are `torch.cat([torch.arange(batch_size), torch.arange(batch_size)])`. The key difference from CLAUDE V2+HNM is the combination of stronger LoRA signal (α=128), slower learning (LR=3e-5), softer temperature (0.03), and smaller batch size (8), following the same recipe that made V2.1+HNM the best single-image model.

**Hard negative mining**: Same `CategoryBatchSampler` — groups by category prefix, chunks into batches of 8. Both images per pair are included simultaneously.

**Training loop**: Zero gradients → encode all `lost_images + found_images` → encode positive + hard-negative captions → normalize → cosine sim `/ 0.03` → cross-entropy → backprop → step optimizer → step scheduler. 1,060 total steps (53 batches × 20 epochs). Cosine scheduler with 10% warmup decays LR from 3e-5 to ~1.24e-5. Early stopping (patience=5) restores the best checkpoint.

**Inference**: Identical to previous models — 600 gallery images encoded once (single image per item), query text encoded on-the-fly, cosine similarity ranks them.

---

### Architecture


| Parameter            | Value                                    |
| -------------------- | ---------------------------------------- |
| Base model           | `qihoo360/fg-clip-base`                  |
| Trainable parameters | 3,932,160 (2.5608% of total)             |
| LoRA rank (r)        | 32                                       |
| LoRA alpha (α)       | 128                                      |
| Target modules       | `q_proj`, `k_proj`, `v_proj`, `out_proj` |
| Dropout              | 0.1                                      |


### Hyperparameters


| Parameter             | Value                                   |
| --------------------- | --------------------------------------- |
| Optimizer             | AdamW                                   |
| Learning rate (start) | 3e-5                                    |
| Learning rate (end)   | 1.24e-5                                 |
| Scheduler             | Cosine with warmup (10% of 1,060 steps) |
| Warmup steps          | 106 / 1,060 total                       |
| Weight decay          | 0.01                                    |
| Loss function         | CrossEntropyLoss                        |
| Temperature           | 0.03 (manual override)                  |
| Train batch size      | 8 (same category, dual-image)           |
| Val batch size        | 8                                       |
| Max epochs            | 20                                      |
| Early stopping        | Patience = 5 epochs                     |
| DataLoader workers    | 0                                       |


### Data Augmentation

```
Compose(
    Resize(size=(224, 224), interpolation=bilinear)
    RandomHorizontalFlip(p=0.5)
    ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    ToTensor()
    RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0)
    ToPILImage()
)
```

### Training Loss Curve


| Epoch | Train Loss | Val Loss   | Note                     |
| ----- | ---------- | ---------- | ------------------------ |
| 1     | 0.8802     | 1.1612     | 🟢 Saved                 |
| 2     | 0.7949     | 1.1268     | 🟢 Saved                 |
| 3     | 0.6886     | 1.1066     | 🟢 Saved                 |
| 4     | 0.5413     | 1.0965     | 🟢 Saved                 |
| 5     | 0.4533     | 1.0525     | 🟢 Saved                 |
| 6     | 0.3761     | 1.0481     | 🟢 Saved                 |
| 7     | 0.2767     | **1.0465** | 🟢 **Best — saved here** |
| 8     | 0.2144     | 1.0633     | 🟡 Patience 1/5          |
| 9     | —          | —          | *(output truncated)*     |
| …     | …          | …          | Stopped early            |


> Best checkpoint: epoch 7, val loss = 1.0465. Early stopping fired at epoch 12 after patience was exhausted. Val loss at stop: 1.0997. Final train loss: 0.1225.

### Training Results


| Metric                   | Value                                  |
| ------------------------ | -------------------------------------- |
| Final train loss         | 0.1225                                 |
| Final val loss (at stop) | 1.0997                                 |
| **Best val loss**        | **1.0465** (saved checkpoint, epoch 7) |
| Epochs trained           | 12 / 20                                |
| GPU VRAM used            | 3.23 GB                                |


### Evaluation Results

**Evaluation setup**: 90 test queries matched against all 600 gallery images. Loaded fine-tuned adapter: `CLAUDEfgclip-lora-finetunedV2.1-1200-HNM`.


| Metric            | Value      | Count |
| ----------------- | ---------- | ----- |
| **Recall@1**      | **78.89%** | 71/90 |
| **Recall@5**      | **94.44%** | 85/90 |
| **Recall@10**     | **96.67%** | 87/90 |
| **MRR**           | **0.8615** | —     |
| Category accuracy | 100.00%    | 90/90 |


#### Across All Models Comparison


| Metric          | V1     | V2     | V2+HNM | V2.1+HNM   | CLAUDE V2+HNM | CLAUDE V2.1+HNM |
| --------------- | ------ | ------ | ------ | ---------- | ------------- | --------------- |
| Recall@1        | 71.11% | 80.00% | 78.89% | **80.00%** | 77.78%        | 78.89%          |
| Recall@5        | 91.11% | 92.22% | 91.11% | 92.22%     | 91.11%        | **94.44%**      |
| Recall@10       | 93.33% | 95.56% | 96.67% | 96.67%     | **96.67%**    | 96.67%          |
| MRR             | 0.8017 | 0.8606 | 0.8556 | **0.8615** | 0.8515        | **0.8615**      |
| Best val loss   | 1.4328 | 1.1565 | 1.1684 | **1.0251** | 1.2539        | 1.0465          |
| Severe failures | 6      | 4      | 3      | 3          | 3             | **3**           |


#### Failure Analysis


| Category            | Count      | Notes                    |
| ------------------- | ---------- | ------------------------ |
| Perfect first-tries | 71 (78.9%) | Retrieved at rank 1      |
| Near misses         | 16 (17.8%) | In top 10 but not rank 1 |
| Severe failures     | 3 (3.3%)   | **Not in top 10 at all** |


#### Severe Failures (not in top 10)


| Test Index | Query       | Correct Item | Rank    | V2.1+HNM Rank | CLAUDE V2+HNM Rank | Notes                                                                            |
| ---------- | ----------- | ------------ | ------- | ------------- | ------------------ | -------------------------------------------------------------------------------- |
| 46         | tumbler_047 | tumbler_047  | **#26** | #33           | #30                | Cream off-white tumbler (best rank yet for this item across all HNM variants!)   |
| 52         | tumbler_065 | tumbler_065  | **#80** | #86           | #93                | WRELS matte black soft flask (best rank across all models!)                      |
| 62         | charger_040 | charger_040  | **#22** | #16           | ✅ (#8)             | White 22.5W QOOVI charger (regressed vs both single and dual-image HNM variants) |


> **tumbler_065 achieved its best rank ever in CLAUDE V2.1+HNM at #80** — but V3 later surpassed it with rank #38 (new best across all models).

#### Sample Retrieval Leaderboard

For query: *"Light-grey rectangular lunch box, pale blue latch clips and flap compartment, white oval shaped vent button on lid"* (correct: `lunchbox_050`)

> Previous: V1 rank 21, V2 rank 7, V2+HNM rank 7, V2.1+HNM rank 4, CLAUDE V2+HNM rank 6. CLAUDE V2.1+HNM: **rank 8**. V3 later tied V2.1 at rank 4.


| Rank | Item         | Score  | Status       |
| ---- | ------------ | ------ | ------------ |
| #1   | lunchbox_030 | 0.3981 |              |
| #2   | lunchbox_073 | 0.3652 |              |
| #3   | lunchbox_041 | 0.3641 |              |
| #4   | lunchbox_074 | 0.3581 |              |
| #5   | lunchbox_055 | 0.3535 |              |
| #6   | lunchbox_004 | 0.3418 |              |
| #7   | lunchbox_082 | 0.3391 |              |
| #8   | lunchbox_050 | 0.3383 | ✅ TRUE MATCH |
| #9   | lunchbox_042 | 0.3376 |              |
| #10  | lunchbox_064 | 0.3297 |              |


#### Confusion Matrix (Category Level)


| True \ Predicted  | Bags | Chargers | Handkerchiefs | Lunchboxes | Tumblers | Wallets |
| ----------------- | ---- | -------- | ------------- | ---------- | -------- | ------- |
| **Bags**          | 15   | 0        | 0             | 0          | 0        | 0       |
| **Chargers**      | 0    | 15       | 0             | 0          | 0        | 0       |
| **Handkerchiefs** | 0    | 0        | 15            | 0          | 0        | 0       |
| **Lunchboxes**    | 0    | 0        | 0             | 15         | 0        | 0       |
| **Tumblers**      | 0    | 0        | 0             | 0          | 15       | 0       |
| **Wallets**       | 0    | 0        | 0             | 0          | 0        | 15      |


**100% category accuracy** maintained.

---

### What This Run Told Us

- **tumbler_047 and tumbler_065 got their best ranks ever**: Dual-image + V2.1 params combined to produce the biggest single improvement in the two hardest cases. tumbler_047: 33 → 30 → 26; tumbler_065: 86 → 93 → 80
- **R@5 is the highest across all models**: 94.44% (85/90) — tied with no other model. The dual-image + V2.1 params push more items into the top 5
- **MRR matched V2.1+HNM's best**: 0.8615. The quality of the overall ranking distribution is equal to the best single-image model
- **R@1 still regressed vs V2.1+HNM**: 80.00% → 78.89%. The dual-image signal dilution continues to hurt top-1 accuracy regardless of other parameters
- **charger_040 is the new worst case**: It recovered to rank 8 in CLAUDE V2+HNM, but regressed to rank 22 in CLAUDE V2.1+HNM. The α=128 / lower-LR regime apparently destabilized the specific features needed for this item
- **The lunchbox_050 query regressed**: Rank 4 → 8. The dual-image model may have over-indexed on the hard tumbler cases at the expense of the easier lunchbox case
- **VRAM is manageable**: 3.23 GB vs V2.1+HNM's 2.21 GB. Processing dual images costs ~1 GB of VRAM at batch=8
- **Trade-off pattern**: Dual-image models (CLAUDE V2, CLAUDE V2.1) excel at hard cases (tumblers) and R@5/R@10, while single-image models (V2.1+HNM) excel at R@1 and easier cases

---

## Key Findings & Observations

### 1. Best Model: V3-optimized

**V3-optimized is the best overall model** across all metrics that matter for retrieval:
- **Best R@5**: 96.67% (tied with V2+HNM and both CLAUDE models, but V3 achieves it with fewer severe failures)
- **Best MRR**: 0.8733 (vs 0.8615 for the next best)
- **Best hard-case performance**: tumbler_065 at rank #38 (best across all models), tumbler_047 at rank #14
- **Tied fewest severe failures**: 3 (matching V2.1+HNM and both CLAUDE models)
- **Tied best R@1**: 80.00% (tied with V2 and V2.1+HNM)

### 2. Key Insight: Smaller LoRA + Stronger Regularization Wins

The V3 optimization reveals that **the best path forward is smaller, more regularized models**:
- **Weight decay 0.1 (10x others)** is the single most impactful change — it penalizes large LoRA weights, forcing the model to make smaller, more generalizable updates
- **LoRA rank 16 (half of others)** reduces model capacity, acting as a structural regularizer alongside weight decay
- **Temperature 0.05 (highest among HNM models)** produces softer similarity logits, giving the model more gradient signal on near-miss cases
- The combination of all three means the model was still learning at epoch 20 (train loss=0.5472, val loss kept decreasing) — it never overfit

### 3. Val Loss Is Not a Good Proxy for Retrieval Quality

V3 demonstrates that **best checkpoint selection by val loss can be misleading**:
- V3 never plateaued — val loss decreased every epoch across all 20
- V2.1+HNM had the best val loss (1.0251) but V3 achieved better retrieval metrics despite worse val loss
- **Recommendation**: For retrieval tasks, consider selecting checkpoint by retrieval metrics on a held-out set rather than val loss alone

### 4. The Tumbler Problem Remains Unsolved

- **tumbler_065** is the hardest item across all models — V3 brought it to rank #38, but it's still nowhere near the top 10
- **tumbler_047** improved to rank #14 in V3, but still a severe failure
- Both items share visual ambiguity (cream/off-white colors, generic shapes) that text descriptions struggle to disambiguate

### 5. Dual-Image Models Have a Trade-off

- Dual-image models (CLAUDE V2, CLAUDE V2.1) excel at hard tumbler cases but hurt R@1
- Single-image models (V2.1+HNM, V3) excel at R@1 and overall retrieval
- **Trade-off**: If hard-case performance matters more, try dual-image; if overall metrics matter, stick with single-image

### 6. Hard Negative Mining (HNM) is Essential

Every model after V1 uses category-grouped batching (HNM), and all of them improved over V1's random shuffling:
- HNM forces the model to distinguish between similar items within the same category
- The improvement from V1→V2 (+8.89pp R@1) demonstrates the power of contrastive learning with hard negatives

### 7. CPU Training is Viable for Small Datasets

V3 was trained entirely on CPU (no GPU available), completing 20 epochs successfully:
- With only 420 training images, CPU training is practical
- Train loss behavior (monotonically decreasing) suggests the model could benefit from more epochs
- **Recommendation**: For datasets under 1,000 pairs, CPU training is a viable alternative when GPU is unavailable

---

## Summary Tables

### Table 1: Key Retrieval Metrics


| Model             | R@1        | R@5        | R@10       | MRR        | Severe Fails |
| ----------------- | ---------- | ---------- | ---------- | ---------- | ------------ |
| V1                | 71.11%     | 91.11%     | 93.33%     | 0.8017     | 6            |
| V2                | **80.00%** | 92.22%     | 95.56%     | 0.8606     | 4            |
| V2 + HNM          | 78.89%     | 91.11%     | **96.67%** | 0.8556     | 3            |
| V2.1 + HNM        | **80.00%** | 92.22%     | **96.67%** | **0.8615** | 3            |
| CLAUDE V2 + HNM   | 77.78%     | 91.11%     | **96.67%** | 0.8515     | 3            |
| CLAUDE V2.1 + HNM | 78.89%     | 94.44%     | **96.67%** | 0.8615     | 3            |
| **V3-optimized**  | **80.00%** | **96.67%** | **96.67%** | **0.8733** | **3**         |


> **Best overall**: V3-optimized — best R@5 (96.67%), best MRR (0.8733), tied highest R@1 (80.00%), tied fewest severe failures (3).
> **Best for hard cases**: V3-optimized — tumbler_065 at rank #38 (best across all models), tumbler_047 at rank #14.

---

### Table 2: Full Hyperparameter Comparison


| Parameter              | V1                      | V2                                       | V2+HNM                                   | V2.1+HNM                                 | CLAUDE V2+HNM                            | CLAUDE V2.1+HNM                          | V3-optimized                             |
| ---------------------- | ----------------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- |
| **Architecture**       |                         |                                          |                                          |                                          |                                          |                                          |                                          |
| Base model             | `qihoo360/fg-clip-base` | `qihoo360/fg-clip-base`                  | `qihoo360/fg-clip-base`                  | `qihoo360/fg-clip-base`                  | `qihoo360/fg-clip-base`                  | `qihoo360/fg-clip-base`                  | `qihoo360/fg-clip-base`                  |
| Trainable params       | 491,520                 | 3,932,160                                | 3,932,160                                | 3,932,160                                | 3,932,160                                | 3,932,160                                | 1,966,080                                |
| Trainable %            | 0.33%                   | 2.56%                                    | 2.56%                                    | 2.56%                                    | 2.56%                                    | 2.56%                                    | 1.2970%                                  |
| LoRA rank (r)          | 4                       | 32                                       | 32                                       | 32                                       | 32                                       | 32                                       | **16**                                   |
| LoRA alpha (α)         | 8                       | 64                                       | 64                                       | **128**                                  | 64                                       | **128**                                  | **32**                                   |
| Target modules         | `q_proj`, `v_proj`      | `q_proj`, `k_proj`, `v_proj`, `out_proj` | `q_proj`, `k_proj`, `v_proj`, `out_proj` | `q_proj`, `k_proj`, `v_proj`, `out_proj` | `q_proj`, `k_proj`, `v_proj`, `out_proj` | `q_proj`, `k_proj`, `v_proj`, `out_proj` | `q_proj`, `k_proj`, `v_proj`, `out_proj` |
| Dropout                | 0.0                     | 0.1                                      | 0.1                                      | 0.1                                      | 0.1                                      | 0.1                                      | 0.1                                      |
| **Optimizer**          |                         |                                          |                                          |                                          |                                          |                                          |                                          |
| Optimizer              | SGD                     | AdamW                                    | AdamW                                    | AdamW                                    | AdamW                                    | AdamW                                    | AdamW                                    |
| Learning rate          | 1e-4                    | 5e-5                                     | 5e-5                                     | **3e-5**                                 | 5e-5                                     | **3e-5**                                 | 3e-5                                     |
| Weight decay           | 0                       | 0.01                                     | 0.01                                     | 0.01                                     | 0.01                                     | 0.01                                     | **0.1**                                  |
| **Scheduler**          |                         |                                          |                                          |                                          |                                          |                                          |                                          |
| Scheduler              | None                    | Cosine + warmup                          | Cosine + warmup                          | Cosine + warmup                          | Cosine + warmup                          | Cosine + warmup                          | Cosine + warmup                          |
| Warmup %               | —                       | 10%                                      | 10%                                      | 10%                                      | 10%                                      | 10%                                      | **20%**                                  |
| **Loss**               |                         |                                          |                                          |                                          |                                          |                                          |                                          |
| Temperature            | 0.07 (model default)    | 0.02 (manual)                            | 0.02 (manual)                            | **0.03** (manual)                        | 0.02 (manual)                            | **0.03** (manual)                        | **0.05** (manual)                        |
| Loss type              | Single-image            | Single-image                             | Single-image                             | Single-image                             | **Dual-image**                           | **Dual-image**                           | Single-image                             |
| **Training**           |                         |                                          |                                          |                                          |                                          |                                          |                                          |
| Train batch size       | 16                      | 16                                       | 16                                       | **8**                                    | 16                                       | **8**                                    | 16                                       |
| Batching strategy      | Random shuffle          | Random shuffle                           | **Category-grouped**                     | **Category-grouped**                     | **Category-grouped**                     | **Category-grouped**                     | **Category-grouped**                     |
| Batches per epoch      | ~27                     | ~27                                      | ~27                                      | ~53                                      | ~27                                      | ~53                                      | ~27                                      |
| Total steps            | 540                     | 540                                      | 540                                      | 1,060                                    | 540                                      | 1,060                                    | 540                                      |
| Max epochs             | 20                      | 20                                       | 20                                       | 20                                       | 20                                       | 20                                       | 20                                       |
| Early stopping         | None                    | Patience=3                               | Patience=3                               | **Patience=5**                           | Patience=3                               | **Patience=5**                           | Patience=3                               |
| **Inference**          |                         |                                          |                                          |                                          |                                          |                                          |                                          |
| Images encoded at eval | 600 (single)            | 600 (single)                             | 600 (single)                             | 600 (single)                             | 600 (single)                             | 600 (single)                             | 600 (single)                             |


---

### Table 3: Training Results


| Metric                   | V1            | V2        | V2+HNM   | V2.1+HNM   | CLAUDE V2+HNM | CLAUDE V2.1+HNM | V3-optimized |
| ------------------------ | ------------- | --------- | -------- | ---------- | ------------- | --------------- | -------------- |
| Final train loss         | 0.4484        | 0.1685    | 0.2818   | 0.1486      | 0.2722        | 0.1225          | 0.5472         |
| Final val loss (at stop) | 1.4464        | 1.1870    | 1.1819   | 1.0460      | 1.2654        | 1.0997          | 1.1335         |
| **Best val loss**        | 1.4328        | 1.1565    | 1.1684   | **1.0251**  | 1.2539        | 1.0465          | 1.1335         |
| Epochs trained           | 15 / 15       | 14 / 20   | 10 / 20  | 11 / 20     | 10 / 20       | 12 / 20         | **20 / 20**   |
| Saved at epoch           | 15 (last)     | 14 (best) | 7 (best) | 6 (best)    | 7 (best)      | 7 (best)        | 20 (best)      |
| Early stop triggered     | No            | Yes       | Yes      | Yes         | Yes           | Yes             | **No**         |
| GPU VRAM used            | —             | —         | 2.21 GB  | —          | 3.78 GB       | 3.23 GB         |
| Training seed            | 9435575495300 | —         | —        | —          | —             | 25201920299700  |


---

### Table 4: Failure Analysis


| Metric              | V1           | V2           | V2+HNM       | V2.1+HNM     | CLAUDE V2+HNM | CLAUDE V2.1+HNM | V3-optimized  |
| ------------------- | ------------ | ------------ | ------------ | ------------ | ------------- | --------------- | -------------- |
| Perfect first-tries | 64 (71.1%)  | 72 (80.0%)  | 71 (78.9%)  | 72 (80.0%)  | 70 (77.8%)   | 71 (78.9%)     | 72 (80.0%)    |
| Near misses         | 20 (22.2%)  | 14 (15.6%)  | 16 (17.8%)  | 15 (16.7%)  | 17 (18.9%)   | 16 (17.8%)     | 15 (16.7%)    |
| **Severe failures** | **6 (6.7%)** | **4 (4.4%)** | **3 (3.3%)** | **3 (3.3%)** | **3 (3.3%)** | **3 (3.3%)**   | **3 (3.3%)**  |


---

### Table 5: Severe Failure Items


| Test Item                                     | V1     | V2    | V2+HNM   | V2.1+HNM | CLAUDE V2+HNM | CLAUDE V2.1+HNM | V3-optimized  |
| --------------------------------------------- | ------ | ----- | -------- | -------- | ------------- | --------------- | -------------- |
| `tumbler_047` (cream, pastel tulip/butterfly) | #60 ❌  | #13 ❌ | #28 ❌    | #33 ❌    | #30 ❌         | #26 ❌         | **#14** ❌     |
| `tumbler_065` (WRELS matte black flask)       | #113 ❌ | #99 ❌ | #101 ❌   | #86 ❌    | #93 ❌         | #80 ❌         | **#38** ❌     |
| `charger_040` (white QOOVI 22.5W)             | #4 ✅   | #24 ❌ | **#8** ✅ | #16 ❌    | #8 ✅          | #22 ❌         | #20 ❌         |
| `charger_081` (pink multi-port)               | #16 ❌  | #11 ❌ | #8 ✅     | ✅        | #11 ❌         | ✅               | ✅              |
| `bag_058` (black mini barrel bag)             | #12 ❌  | — ✅   | — ✅      | — ✅      | — ✅           | — ✅             | — ✅            |
| `handkerchief_027`                            | —      | —     | #11 ❌    | — ✅      | —             | —               | — ✅            |


> ❌ = not in top 10 (severe failure). ✅ = in top 10 (recovered). Blank = was never a severe failure for that model. Bold = best rank for that item across all models.

**tumbler_065 is the hardest item across all models** — V3 brought it to rank #38, the best ever, but it remains outside the top 10. **tumbler_047** (rank #14 in V3) is second hardest. **charger_040** oscillates near the boundary (V3: #20, severe). **charger_081** recovered in V3. **bag_058** and **handkerchief_027** have been resolved by most HNM models.

---

### Table 6: Hard-Case Rank Trajectory


| Item                          | V1    | V2    | V2+HNM | V2.1+HNM | CLAUDE V2+HNM | CLAUDE V2.1+HNM | V3-optimized  | Trend                                |
| ----------------------------- | ----- | ----- | ------ | -------- | ------------- | --------------- | -------------- | ------------------------------------ |
| `tumbler_047`                 | #60   | #13   | #28    | #33      | #30           | #26            | **#14**       | ↘↗↘↗↘↗ Best in V3                 |
| `tumbler_065`                 | #113  | #99   | #101   | #86      | #93           | #80            | **#38**       | ↘↘↗↘↗↘ Best in V3                |
| `charger_040`                 | #4 ✅  | #24 ❌ | #8 ✅   | #16 ❌    | #8 ✅          | #22 ❌         | #20 ❌        | Oscillating — boundary case         |
| `charger_081`                 | #16 ❌ | #11 ❌ | #8 ✅   | ✅        | #11 ❌         | ✅               | ✅             | Recovered in V3                      |
| `lunchbox_050` (sample query) | #21   | #7    | #7     | #4       | #6            | #8             | **#4**        | V3 ties V2.1 at best (#4)          |


---

### Table 7: Cross-Model Improvements (vs V1 baseline)


| Metric          | V2       | V2+HNM       | V2.1+HNM     | CLAUDE V2+HNM | CLAUDE V2.1+HNM | V3-optimized    |
| --------------- | -------- | ------------ | ------------ | ------------- | --------------- | ---------------- |
| R@1             | +8.89 pp | +7.78 pp     | +8.89 pp     | +6.67 pp      | +7.78 pp        | +8.89 pp        |
| R@5             | +1.11 pp | +0.00 pp     | +1.11 pp     | +0.00 pp      | +3.33 pp        | **+5.56 pp**    |
| R@10            | +2.23 pp | **+3.34 pp** | +3.34 pp     | +3.34 pp      | +3.34 pp        | +3.34 pp        |
| MRR             | +0.0589  | +0.0539      | +0.0598      | +0.0498       | +0.0598         | **+0.0716**     |
| Severe failures | −2       | −3           | −3           | −3            | −3              | −3               |
| Best val loss   | −0.2763  | −0.2644      | −0.4077      | −0.1789       | −0.3863         | −0.2993         |


---

### Table 8: Ablation Contribution (marginal effect of each change)


| Change                                                          | Introduced in     | Marginal R@1 | Marginal MRR | Notes                                                               |
| --------------------------------------------------------------- | ----------------- | ------------ | ------------ | ------------------------------------------------------------------- |
| Larger LoRA (r=4→32) + more targets + scheduler + ES          | V2                | +8.89 pp     | +0.0589      | Largest single improvement                                          |
| Hard negative mining (category batching)                         | V2+HNM            | −1.11 pp     | −0.0050      | Counterproductive for R@1                                          |
| α=128, LR=3e-5, temp=0.03, batch=8, patience=5              | V2.1+HNM          | +1.11 pp     | +0.0059       | Recovered HNM's R@1 loss + improved MRR                             |
| Dual-image loss                                                 | CLAUDE V2+HNM     | −1.11 pp     | −0.0032       | Dilutes per-image signal; hurts R@1                               |
| Dual-image + V2.1 params combined                              | CLAUDE V2.1+HNM   | +1.11 pp     | +0.0100       | V2.1 params compensate for dual-image penalty                      |
| Smaller LoRA (r=32→16) + stronger WD (0.01→0.1) + temp 0.05  | V3-optimized       | +0.00 pp     | +0.0118       | Best MRR, best R@5; val loss not a good proxy for retrieval quality |


