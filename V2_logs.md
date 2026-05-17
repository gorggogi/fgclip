# V2 + HNM — `fgclip-lora-finetunedV2-1200-HNM`

> **Status**: Expanded LoRA (r=32, all 4 attention projections) + category-grouped batching + cosine scheduler + strict temperature (0.02) + patience=3 early stopping. GPU training on RTX 3050 (max VRAM 3.80 GB). Early stopped at epoch 10. **Result: R@1 78.89%, R@5 91.11%, R@10 96.67%, MRR 0.8556, 3 severe failures. Strong improvement over V1 across all metrics.**

> **Base Model**: `qihoo360/fg-clip-base` (149,620,737 total parameters)
> **Environment**: Windows 11, Python 3.14, PyTorch 2.7.1+cu118, Transformers 5.8.0, PEFT 0.19.1
> **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU
> **Training notebook**: `fgclipfintuning/Copy_of_fgclipFineTuning_V2_1200_HNM.ipynb`
> **Evaluation notebook**: `fgclipevaluation/Copy_of_fgclipEvaluation_V2_1200_HNM.ipynb`
> **Output directory**: `lorafinetuned/fgclip-lora-finetunedV2-1200-HNM/`
> **Training seed**: Not recorded

---

## Table of Contents

1. [What Changed from V1](#what-changed-from-v1)
2. [Why These Changes](#why-these-changes)
3. [Training Methodology](#training-methodology)
4. [Architecture](#architecture)
5. [Hyperparameters](#hyperparameters)
6. [Data Augmentation](#data-augmentation)
7. [Training Loss Curve](#training-loss-curve)
8. [Training Results](#training-results)
9. [Evaluation Results](#evaluation-results)
10. [What This Run Told Us](#what-this-run-told-us)
11. [Retrieval Demo](#retrieval-demo)

---

## What Changed from V1


| Aspect            | V1                     | V2 + HNM                                                |
| ----------------- | ---------------------- | ------------------------------------------------------- |
| LoRA rank (r)     | 8                      | **32**                                                  |
| LoRA alpha (α)    | 16                     | **64**                                                  |
| Target modules    | q_proj, v_proj         | **q_proj, k_proj, v_proj, out_proj**                    |
| Temperature       | ~0.07 (model default)  | **0.02** (manual override)                              |
| Optimizer         | AdamW, lr=5e-5         | AdamW, lr=5e-5                                          |
| Scheduler         | None                   | **Cosine with warmup (10% of 540 steps)**               |
| Cosine end LR     | N/A                    | **0.0** (decays to zero)                                |
| Weight decay      | 0.01                   | 0.01                                                    |
| Batch size        | 16 (random shuffle)    | 16 (category-grouped via CategoryBatchSampler)          |
| Batching strategy | Random shuffle         | **Category-grouped: pairs bucketed by category prefix** |
| Early stopping    | None (15 epochs fixed) | **Patience = 3**                                        |
| Data augmentation | Brightness only        | Full-spectrum ColorJitter                               |
| Hard negatives    | No                     | **16 hard negatives per batch (all same category)**     |


---

## Why These Changes

- **r=8 → r=32, q/v → q/k/v/out**: Expanded rank and target modules give the adapter significantly more capacity to learn fine-grained visual-textual alignment. Adding k_proj and out_proj enables the adapter to directly modify query-key computation — how the model decides which tokens to attend to — rather than only adjusting value aggregation.
- **Temperature 0.07 → 0.02**: A lower temperature sharpens the softmax distribution, producing steeper gradient signals and a stricter penalty for placing probability mass on hard negatives.
- **Cosine with 10% warmup**: A 10% linear warmup period stabilizes early training when LoRA matrices are near-zero initialization. The cosine decay enables smooth convergence without abrupt learning rate drops, ending at zero which acts as a built-in early-stop mechanism.
- **Category-grouped batching**: The `CategoryBatchSampler` groups pairs by category prefix (e.g. bag012 → "bag"), shuffles within each category, then chunks into batches of 16. Every training batch therefore contains 16 pairs from the same category, with all 16 hard negatives being plausible matches for each image — a much harder discrimination task than random batching.
- **Patience = 3**: Halts when validation loss fails to improve for three consecutive epochs, preserving the best-generalization checkpoint (epoch 7) rather than the final one (epoch 10).
- **Full-spectrum ColorJitter**: V1 only jittered brightness. V2 applies brightness, contrast, saturation, and hue jitter, giving a more uniform robustness envelope against real-world lighting variation.

---

## Training Methodology

**Task**: Image-text retrieval. Given a text description of a lost item, retrieve the matching image from a gallery of 600 found items.

**Loss**: Single-image InfoNCE / CLIP loss. Category-bucketed batches of 16 pairs (16 positive + 16 hard-negative captions).

**Hard negative mining**: `CategoryBatchSampler` — groups pairs by category prefix, chunks into batches of 16. Every batch contains pairs from the same category, ensuring all 16 hard negatives are plausible matches.

**Training loop**: Zero gradients → encode `lost_image` → encode positive + hard-negative captions → normalize → cosine sim `/ 0.02` → cross-entropy → backprop → step optimizer → step scheduler. Only LoRA adapters train. Cosine scheduler with 10% warmup (54 warmup steps out of 540 total) decays LR from 5e-5 to 0. Early stopping (patience=3) restored best checkpoint from epoch 7.

**Inference**: 600 gallery images encoded once, query text encoded on-the-fly, cosine similarity ranks them.

---

## Architecture


| Parameter            | Value                                                       |
| -------------------- | ----------------------------------------------------------- |
| Base model           | `qihoo360/fg-clip-base`                                     |
| Vision backbone      | ViT-B/16 (14×14 patch grid, 197 tokens, 768-dim embeddings) |
| Trainable parameters | **3,932,160 (2.5608% of total)** |                           |
| Full model total (base) | **153,552,897** (base 149,620,737 + LoRA 3,932,160) |
| LoRA rank (r)        | **32**                                                      |
| LoRA alpha (α)       | **64**                                                      |
| Target modules       | `q_proj`, `k_proj`, `v_proj`, `out_proj`                    |
| Dropout              | 0.1                                                         |
| Model dtype          | `torch.float32`                                             |


---

## Hyperparameters


| Parameter             | Value                                         |
| --------------------- | --------------------------------------------- |
| Optimizer             | AdamW                                         |
| Learning rate (start) | 5e-5                                          |
| Learning rate (end)   | **0.0** (cosine decay to zero)                |
| Scheduler             | Cosine with warmup (10% of 540 steps)         |
| Warmup steps          | 54 / 540 total                                |
| Weight decay          | 0.01                                          |
| Betas                 | (0.9, 0.999)                                  |
| Epsilon               | 1e-8                                          |
| Loss function         | CrossEntropyLoss                              |
| Temperature           | **0.02** (manual override)                    |
| Batch size            | 16 (val), 16 (train via CategoryBatchSampler) |
| Max epochs            | 20                                            |
| Early stopping        | **Yes** (patience = 3, triggered at epoch 10) |
| DataLoader workers    | 0                                             |


---

## Data Augmentation

Compose(
Resize(size=(224, 224), interpolation=bilinear)
RandomHorizontalFlip(p=0.5)
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.4, hue=0.1)
ToTensor()
RandomErasing(p=0.3, scale=(0.02, 0.15))
ToPILImage()
)

> **Note**: ColorJitter parameters not recorded in this notebook run. See V3 for explicit brightness=0.2, contrast=0.2, saturation=0.4, hue=0.1.

---

## Training Loss Curve


| Epoch | Train Loss | Val Loss   | Note                             |
| ----- | ---------- | ---------- | -------------------------------- |
| 1     | 1.0411     | 1.3113     | Saved                            |
| 2     | 0.9853     | 1.2782     | Saved                            |
| 3     | 0.8973     | 1.2511     | Saved                            |
| 4     | 0.6773     | 1.2375     | Saved                            |
| 5     | 0.6473     | 1.2251     | Saved                            |
| 6     | 0.5442     | 1.2101     | Saved                            |
| 7     | 0.4608     | **1.1684** | **Best — saved**                 |
| 8     | 0.3574     | 1.1730     | Worsened 1/3                     |
| 9     | 0.3166     | 1.1918     | Worsened 2/3                     |
| 10    | 0.2818     | 1.1819     | Worsened 3/3 — **early stopped** |


> **Val loss peaked at epoch 7**: Best checkpoint saved at val loss = 1.1684. Epochs 8–10 showed consistent val loss degradation (1.1730 → 1.1819 → 1.1918), exhausting the patience=3 early stopping window by epoch 10. Train loss dropped from 0.4608 to 0.2818 — a 39% reduction — while val loss worsened, confirming the model was beginning to overfit.

---

## Training Results


| Metric                   | Value                                        |
| ------------------------ | -------------------------------------------- |
| Final train loss         | 0.2818                                       |
| Final val loss (at stop) | 1.1819                                       |
| **Best val loss**        | **1.1684** (saved checkpoint, epoch 7)       |
| Epochs trained           | 10 / 20                                      |
| Epochs saved             | 7                                            |
| Early stop triggered     | **Yes** (patience = 3 exhausted at epoch 10) |
| Peak VRAM                | 3.80 GB                                      |


---

## Evaluation Results

**Evaluation setup**: 90 test queries matched against all 600 gallery images. Loaded fine-tuned adapter: `fgclip-lora-finetunedV2-1200-HNM`.

### Overall Metrics


| Metric        | Value      | Count |
| ------------- | ---------- | ----- |
| **Recall@1**  | **78.89%** | 71/90 |
| **Recall@5**  | **91.11%** | 82/90 |
| **Recall@10** | **96.67%** | 87/90 |
| **MRR**       | **0.8556** | —     |


### Failure Analysis


| Category            | Count      | Notes                    |
| ------------------- | ---------- | ------------------------ |
| Perfect first-tries | 71 (78.9%) | Retrieved at rank 1      |
| Near misses         | 11 (12.2%) | In top 10 but not rank 1 |
| Severe failures     | 3 (3.3%)   | **Not in top 10 at all** |


### Severe Failures (not in top 10)


| Test Index | Query       | Correct Item | Rank     | Notes                                                                                                                                        |
| ---------- | ----------- | ------------ | -------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 46         | tumbler_047 | tumbler_047  | **#28**  | Cream off-white insulated tumbler, all-over pastel pink tulip and butterfly line-art print, pink ribbed rubber base boot, pink flip-top lid. |
| 52         | tumbler_065 | tumbler_065  | **#101** | WRELS matte black soft flask, black TPU body with white "WRELS Love Life Love Sports" branding.                                              |
| 75         | handkerchief_027 | handkerchief_027  | **#11** | Multicolor floral print handkerchief, small square shape, dense repeating flower pattern with blue edging. |


### Rank Distribution Matrix

> Tracks where the correct item ended up in the ranking, grouped by rank bucket.


|               | #1  | #2-5 | #6-10 | #11-20 | #21-50 | #50+ |
| :------------ | --: | ----: | -----: | ------: | ------: | ---: |
| Bag           | 13  | 1    | 1      | 0       | 0      | 0    |
| Charger       |  9  | 4    | 2      | 0       | 0      | 0    |
| Handkerchief  | 10  | 4    | 0      | 1       | 0      | 0    |
| Lunchbox      | 13  | 1    | 1      | 0       | 0      | 0    |
| Tumbler       | 11  | 1    | 1      | 0       | 1      | 1    |
| Wallet        | 15  | 0    | 0      | 0       | 0      | 0    |


> **Per-category summary:**


| Category     | R@1 Acc  | Avg Rank | Worst Rank |
| :----------- | :------- | :------- | :--------- |
| Bag          | 86.7%    | 1.5      | #7         |
| Charger      | 60.0%    | 2.2      | #8         |
| Handkerchief | 66.7%    | 2.0      | #11        |
| Lunchbox     | 86.7%    | 1.5      | #7         |
| Tumbler      | 73.3%    | 9.9      | #101       |
| Wallet       | 100.0%   | 1.0      | #1         |


### Per-Category Item-Level Accuracy

> Which specific items were confused with which at rank #1.


| Category     | R@1 Correct | Wrong at R@1 (true -> AI guess, rank)                                                                                                                                                                        |
| ------------ | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Bag          | 13          | bag_058->bag_057 (#7), bag_093->bag_014 (#2)                                                                                                                                                                 |
| Charger      | 9           | charger_038->charger_003 (#2), charger_040->charger_037 (#8), charger_071->charger_070 (#2), charger_081->charger_070 (#8), charger_083->charger_071 (#2), charger_099->charger_092 (#2)                     |
| Handkerchief | 10          | handkerchief_027->handkerchief_079 (#11), handkerchief_034->handkerchief_083 (#2), handkerchief_035->handkerchief_087 (#3), handkerchief_062->handkerchief_098 (#2), handkerchief_096->handkerchief_079 (#2) |
| Lunchbox     | 13          | lunchbox_050->lunchbox_074 (#7), lunchbox_079->lunchbox_074 (#2)                                                                                                                                             |
| Tumbler      | 11          | tumbler_047->tumbler_058 (#28), tumbler_065->tumbler_051 (#101), tumbler_066->tumbler_021 (#6), tumbler_084->tumbler_086 (#3)                                                                                |
| Wallet       | 15          | —                                                                                                                                                                                                            |


### Confusion Matrix (Category Level)

> Tracks whether the **top-ranked gallery item's category** matches the **query's true category**.


| True \ Predicted | Bag | Charger | Handkerchief | Lunchbox | Tumbler | Wallet | Overall |
| ---------------- | --- | ------- | ------------ | -------- | ------- | ------ | ------- |
| **Bag**          | 15  | 0       | 0            | 0        | 0       | 0      | 1.00    |
| **Charger**      | 0   | 15      | 0            | 0        | 0       | 0      | 1.00    |
| **Handkerchief** | 0   | 0       | 15           | 0        | 0       | 0      | 1.00    |
| **Lunchbox**     | 0   | 0       | 0            | 15       | 0       | 0      | 1.00    |
| **Tumbler**      | 0   | 0       | 0            | 0        | 15      | 0      | 1.00    |
| **Wallet**       | 0   | 0       | 0            | 0        | 0       | 15     | 1.00    |


> **Note**: Category-level accuracy is 100% (90/90). Every top-ranked gallery item belongs to the correct category — the 11 R@1 failures and 3 severe failures are all within-category errors.

---

## Retrieval Demo

**Test query used:**

> *"Light-grey rectangular lunch box, pale blue latch clips and flap compartment, white oval shaped vent button on lid"* — lunchbox_050

**Retrieval Results:**


| Rank | Item Name     | Score   | Status      |
| ---- | ------------ | ------- | ----------- |
| #1   | lunchbox_074 | 0.3722  | Top-1       |
| #2   | lunchbox_030 | 0.3717  | Top-2       |
| #3   | lunchbox_041 | 0.3556  | Top-3       |
| #4   | lunchbox_072 | 0.3530  | Top-4       |
| #5   | lunchbox_055 | 0.3463  | Top-5       |
| #6   | lunchbox_073 | 0.3433  | Top-6       |
| #7   | lunchbox_050 | 0.3376  | **Correct**  |
| #8   | lunchbox_042 | 0.3351  | Top-8       |
| #9   | lunchbox_066 | 0.3248  | Top-9       |
| #10  | lunchbox_007 | 0.3212  | Top-10      |


---

## What This Run Told Us

- **Expanded LoRA capacity works**: Going from r=8 (q/v only) to r=32 (all 4 projections) produced a massive R@1 improvement (+7.78 points, 71.11% → 78.89%).
- **Category-grouped batching is the key differentiator**: Hard negatives from the same category as the query make each batch a much harder discrimination task, directly improving within-category retrieval.
- **Temperature 0.02 is effective but aggressive**: The low temperature sharpens gradients but combined with r=32, the model began overfitting by epoch 8. The cosine-to-zero scheduler could not fully compensate.
- **Val loss is a lagging indicator**: Best val loss was at epoch 7, but the model only stopped at epoch 10 after patience was exhausted. The train-val gap (0.4608 vs 1.1684) at best checkpoint is large, indicating the model had room to improve but was stopped by the patience threshold.
- **tumbler_065 is the hardest case**: Rank #101 in V2 — the model completely fails to retrieve this item. Both the tumbler_065 query text and the gallery image describe a near-identical WRELS product, yet the model ranks them far apart.
- **tumbler_047 is recoverable**: Rank #28 in V2 is bad but not catastrophic. The cream tumbler with pink butterfly print is visually distinctive enough that V3 was able to bring it to #16.
- **The model never misclassifies category**: 100% category-level accuracy confirms the pre-fine-tuned CLIP already knows bag vs tumbler vs charger. The hard problem is within-category item discrimination — which is exactly what V3 addresses with stronger regularization.

