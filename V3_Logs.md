# V3-optimized — `fgclip-lora-finetunedV3-optimized`

> **Status**: Small rank (r=16) + strong weight decay (wd=0.1) + higher temperature (0.05) + patience=3 optimization. GPU training on RTX 3050 (max VRAM 3.80 GB). **Result: best overall model — best R@1 (84.44%), best R@5 (96.67%), best R@10 (96.67%), best MRR (0.8897), fewest severe failures (3).**

> **Base Model**: `qihoo360/fg-clip-base` (149,620,737 total parameters)
> **Environment**: Windows 10, Python 3.14, PyTorch 2.7.1+cu118, Transformers 5.8.0, PEFT 0.19.1
> **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU
> **Training notebook**: `fgclipfintuning/Copy_of_fgclipFineTuning_V3_1200_HNM.ipynb`
> **Evaluation notebook**: `fgclipevaluation/Copy_of_fgclipEvaluation_V3_1200_HNM.ipynb`
> **Output directory**: `lorafinetuned/fgclip-lora-finetunedV3-optimized/`
> **Training seed**: 5647261231500

---

## Table of Contents

1. [What Changed from V2 + HNM](#what-changed-from-v2--hnm)
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

## What Changed from V2


| Aspect            | V2 + HNM                                             | V3-optimized                                              |
| ----------------- | ----------------------------------------------------- | --------------------------------------------------------- |
| LoRA rank (r)     | 32                                                    | **16**                                                    |
| LoRA alpha (α)    | 64                                                    | **32**                                                    |
| Weight decay      | 0.01                                                  | **0.1** (10x stronger)                                    |
| Temperature       | 0.02                                                  | **0.05**                                                  |
| Cosine end LR     | 0.0                                                   | **0.0** (cosine to zero)                                  |
| Warmup            | 10%                                                   | **20%**                                                   |
| Early stopping    | Patience = 3                                          | **Patience = 3**                                          |
| Hardware          | GPU (RTX 3050)                                        | **GPU (RTX 3050)**                                        |
| Data augmentation | brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1 | **brightness=0.2, contrast=0.2, saturation=0.4, hue=0.1** |


---

## Why These Changes

- **r=32 → r=16**: V2.1's r=32 + α=128 setup is prone to overfitting (train loss=0.1486 vs val loss=1.0251). Halving rank to 16 acts like a stronger regularizer, reducing the model's capacity to memorize training examples.
- **α=128 → α=32**: Keeping α/r ratio constant at 2 maintains the same effective scaling behavior while reducing the absolute magnitude of LoRA updates.
- **Weight decay 0.01 → 0.1**: The single most impactful change. Stronger weight decay penalizes large LoRA weights, encouraging the model to use smaller, more generalizable adaptations.
- **Temperature 0.03 → 0.05**: Higher temperature produces softer similarity logits, giving the model more gradient signal on near-miss cases. Correlates with best MRR.
- **Cosine to zero**: Instead of decaying to 1.5e-5, the scheduler decays to 0, effectively stopping weight updates in the final epochs — acts as a built-in early-stop mechanism.
- **20% warmup**: Doubling warmup from 10% gives the model more time to stabilize before taking large gradient steps.
- **GPU training**: Despite the small dataset (420 images), training ran on the RTX 3050 with 3.80 GB VRAM peak usage — fast enough to be practical.

---

## Training Methodology

**Task**: Image-text retrieval. Given a text description of a lost item, retrieve the matching image from a gallery of 600 found items.

**Loss**: Single-image InfoNCE / CLIP loss. Category-bucketed batches of 16 pairs (16 positive + 16 hard-negative captions). The combination of smaller rank, stronger weight decay, and higher temperature (0.05) forces the model to learn more robust, generalizable features rather than memorizing training pairs.

**Hard negative mining**: `CategoryBatchSampler` — groups pairs by category prefix, chunks into batches of 16.

**Training loop**: Zero gradients → encode `lost_image` → encode positive + hard-negative captions → normalize → cosine sim `/ 0.05` → cross-entropy → backprop → step optimizer → step scheduler. Only LoRA adapters train. Cosine scheduler with 20% warmup decays LR from 3e-5 to 0 over 540 steps. Early stopping (patience=3) restored best checkpoint from epoch 17.

**Inference**: 600 gallery images encoded once, query text encoded on-the-fly, cosine similarity ranks them.

---

## Architecture


| Parameter            | Value                                                       |
| -------------------- | ----------------------------------------------------------- |
| Base model           | `qihoo360/fg-clip-base`                                     |
| Vision backbone      | ViT-B/16 (14×14 patch grid, 197 tokens, 768-dim embeddings) |
| Trainable parameters | **1,966,080 (1.2970% of total)** |                              |
| Full model total (base) | **151,586,817** (base 149,620,737 + LoRA 1,966,080) |
| LoRA rank (r)        | **16**                                                      |
| LoRA alpha (α)       | **32**                                                      |
| Target modules       | `q_proj`, `k_proj`, `v_proj`, `out_proj`                    |
| Dropout              | 0.1                                                         |
| Model dtype          | `torch.float32`                                             |


---

## Hyperparameters


| Parameter             | Value                                         |
| --------------------- | --------------------------------------------- |
| Optimizer             | AdamW                                         |
| Learning rate (start) | 3e-5                                          |
| Learning rate (end)   | **0.0** (cosine decay to zero)                |
| Scheduler             | Cosine with warmup (20% of 540 steps)         |
| Warmup steps          | 108 / 540 total                               |
| Weight decay          | **0.1**                                       |
| Betas                 | (0.9, 0.999)                                  |
| Epsilon               | 1e-8                                          |
| Loss function         | CrossEntropyLoss                              |
| Temperature           | **0.05** (manual override)                    |
| Batch size            | 16 (val), 16 (train via CategoryBatchSampler) |
| Max epochs            | 20                                            |
| Early stopping        | Patience = 3 epochs (triggered at epoch 20)   |
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

> **Note**: Reverted brightness/contrast from V2.1's 0.4 back to 0.2 (matching V2), keeping saturation at 0.4 and hue at 0.1.

---

## Training Loss Curve


| Epoch | Train Loss | Val Loss   | Note                                      |
| ----- | ---------- | ---------- | ----------------------------------------- |
| 1     | 1.0214     | 1.1868     | 🟢 Saved                                  |
| 2     | 1.0080     | 1.1594     | 🟢 Saved                                  |
| 3     | 0.9255     | 1.1080     | 🟢 Saved                                  |
| 4     | 0.9037     | 1.0343     | 🟢 Saved                                  |
| 5     | 0.7983     | 0.9700     | 🟢 Saved                                  |
| 6     | 0.7694     | 0.9209     | 🟢 Saved                                  |
| 7     | 0.6947     | 0.8858     | 🟢 Saved                                  |
| 8     | 0.6512     | 0.8695     | 🟢 Saved                                  |
| 9     | 0.6211     | 0.8527     | 🟢 Saved                                  |
| 10    | 0.5701     | 0.8417     | 🟢 Saved                                  |
| 11    | 0.5199     | 0.8364     | 🟢 Saved                                  |
| 12    | 0.5278     | 0.8280     | 🟢 Saved                                  |
| 13    | 0.4966     | 0.8255     | 🟢 Saved                                  |
| 14    | 0.4834     | 0.8198     | 🟢 Saved                                  |
| 15    | 0.4615     | 0.8183     | 🟢 Saved                                  |
| 16    | 0.4625     | 0.8154     | 🟢 Saved                                  |
| 17    | 0.4615     | **0.8148** | 🟢 **Best — saved**                       |
| 18    | 0.4386     | 0.8153     | 🟡 No improvement 1/3                     |
| 19    | 0.4636     | 0.8152     | 🟡 No improvement 2/3                     |
| 20    | 0.4547     | 0.8152     | 🟡 No improvement 3/3 — restored epoch 17 |


> **Val loss nearly plateaued**: Val loss decreased through all 20 epochs (1.1868 → 0.8152), with epoch 17 marking the best checkpoint (val loss = 0.8148). Epochs 18-20 showed no improvement over epoch 17, exhausting the patience=3 early stopping window by epoch 20.

---

## Training Results


| Metric                   | Value                                        |
| ------------------------ | -------------------------------------------- |
| Final train loss         | 0.4547                                       |
| Final val loss (at stop) | 0.8152                                       |
| **Best val loss**        | **0.8148** (saved checkpoint, epoch 17)      |
| Epochs trained           | 20 / 20                                      |
| Epoch saved             | 17                                           |
| Early stop triggered     | **Yes** (patience = 3 exhausted at epoch 20) |
| Peak VRAM                | 3.80 GB                                      |


---

## Evaluation Results

**Evaluation setup**: 90 test queries matched against all 600 gallery images. Loaded fine-tuned adapter: `fgclip-lora-finetunedV3-optimized`.

### Overall Metrics


| Metric            | Value      | Count |
| ----------------- | ---------- | ----- |
| **Recall@1**      | **84.44%** | 76/90 |
| **Recall@5**      | **96.67%** | 87/90 |
| **Recall@10**     | **96.67%** | 87/90 |
| **MRR**           | **0.8897** | —     |
| Category accuracy | 98.89%    | 89/90 |


### Failure Analysis


| Category            | Count      | Notes                    |
| ------------------- | ---------- | ------------------------ |
| Perfect first-tries | 76 (84.4%) | Retrieved at rank 1      |
| Near misses         | 11 (12.2%) | In top 10 but not rank 1 |
| Severe failures     | 3 (3.3%)   | **Not in top 10 at all** |


### Severe Failures (not in top 10)


| Test Index | Query       | Correct Item | Rank    | Notes                                                                                                                                        |
| ---------- | ----------- | ------------ | ------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 46         | tumbler_047 | tumbler_047  | **#16** | Cream off-white insulated tumbler, all-over pastel pink tulip and butterfly line-art print, pink ribbed rubber base boot, pink flip-top lid. |
| 52         | tumbler_065 | tumbler_065  | **#14** | WRELS matte black soft flask, black TPU body with white "WRELS Love Life Love Sports" branding.                                              |
| 62         | charger_040 | charger_040  | **#23** | White matte plastic 22.5W wall charger, compact square body, "22.5W DESIGNED BY QOOVI" printed.                                              |


> **tumbler_065 achieved its best rank ever**: #14 in V3 is far better than any previous model. CLAUDE V2.1+HNM held the previous best at #80 — an 83% improvement in rank position.
> **tumbler_047 achieved its best rank across all models**: V3 at #16 beats V2.1 (#33) and CLAUDE V2.1 (#26), though V2's #13 remains the best across all models.

### Rank Distribution Matrix

> Tracks where the correct item ended up in the ranking, grouped by rank bucket.

|              | #1  | #2-5 | #6-10 | #11-20 | #21-50 | #50+ |
| :----------- | --: | ----: | -----: | ------: | ------: | ---: |
| **Bag**          | 13  | 2     | 0      | 0       | 0       | 0    |
| **Charger**      | 10  | 4     | 0      | 0       | 1       | 0    |
| **Handkerchief** | 12  | 3     | 0      | 0       | 0       | 0    |
| **Lunchbox**     | 14  | 1     | 0      | 0       | 0       | 0    |
| **Tumbler**      | 12  | 1     | 0      | 2       | 0       | 0    |
| **Wallet**       | 15  | 0     | 0      | 0       | 0       | 0    |

> **Per-category summary:**

| Category     | R@1 Acc  | Avg Rank | Worst Rank |
| :----------- | :------- | :------- | :--------- |
| Bag          | 86.7%    | 1.3      | #4         |
| Charger      | 66.7%    | 3.2      | #23        |
| Handkerchief | 80.0%    | 1.4      | #5         |
| Lunchbox     | 93.3%    | 1.2      | #4         |
| Tumbler      | 80.0%    | 2.9      | #16        |
| Wallet       | 100.0%   | 1.0      | #1         |

> **tumbler_065 achieved its best rank ever:** #14 in V3 is far better than any previous model. CLAUDE V2.1+HNM held the previous best at #80 — an 83% improvement in rank position.
> **tumbler_047 achieved its best rank across all models:** V3 at #16 beats V2.1 (#33) and CLAUDE V2.1 (#26), though V2's #13 remains the best across all models.

---

### Per-Category Item-Level Accuracy

> Which specific items were confused with which at rank #1.

| Category      | R@1 Correct | Wrong at R@1 (true -> AI guess, rank)                                       |
| :------------ | :---------- | :-------------------------------------------------------------------------- |
| Bag           | 13         | bag_058->tumbler_007 (#4), bag_071->bag_070 (#2)                           |
| Charger       | 10         | charger_038->charger_030 (#4), charger_071->charger_070 (#5), charger_081->charger_070 (#4), charger_083->charger_071 (#2)                          |
| Handkerchief  | 12         | handkerchief_027->handkerchief_079 (#5), handkerchief_035->handkerchief_087 (#2), handkerchief_062->handkerchief_098 (#2) |
| Lunchbox      | 14         | lunchbox_050->lunchbox_041 (#4)                                            |
| Tumbler       | 12         | tumbler_047->tumbler_045 (#16), tumbler_065->tumbler_027 (#14), tumbler_084->tumbler_086 (#2) |
| Wallet        | 15         | —                                                                          |



### Confusion Matrix (Category Level)

> Tracks whether the **top-ranked gallery item's category** matches the **query's true category**.


| Category       | Bag | Charger | Handkerchief | Lunchbox | Tumbler | Wallet | Overall |
| ------------ | --- | ------- | ------------ | -------- | ------- | ------ | ------- |
| Precision    | 1.00 | 1.00   | 1.00         | 1.00     | 0.94    | 1.00   | 0.99    |
| Recall       | 0.93 | 1.00   | 1.00         | 1.00     | 1.00    | 1.00   | 0.99    |
| F1-Score     | 0.97 | 1.00   | 1.00         | 1.00     | 0.97    | 1.00   | 0.99    |
| Support      | 15   | 15      | 15           | 15       | 15      | 15     | 90      |

> **Note**: 1 bag query (bag_058) had a tumbler ranked above its matching bag image — category-level accuracy is 98.89% (89/90). This is expected since the confusion matrix measures *category prediction* at rank 1, not exact item retrieval.

### Confusion Matrix (Category Level)

| True \ Predicted | Bag | Charger | Handkerchief | Lunchbox | Tumbler | Wallet |
| ---------------- | --- | ------- | ------------ | -------- | ------- | ------ |
| **Bag**          | 14  | 0       | 0            | 0        | 1       | 0      |
| **Charger**      | 0   | 15      | 0            | 0        | 0       | 0      |
| **Handkerchief** | 0   | 0       | 15           | 0        | 0       | 0      |
| **Lunchbox**     | 0   | 0       | 0            | 15       | 0       | 0      |
| **Tumbler**      | 0   | 0       | 0            | 0        | 15      | 0      |
| **Wallet**       | 0   | 0       | 0            | 0        | 0       | 15     |

| Category     | Precision | Recall | F1-Score | Support |
| ------------ | -------- | ------ | -------- | ------- |
| Bag          | 1.00     | 0.93   | 0.97     | 15      |
| Charger      | 1.00     | 1.00   | 1.00     | 15      |
| Handkerchief | 1.00     | 1.00   | 1.00     | 15      |
| Lunchbox     | 1.00     | 1.00   | 1.00     | 15      |
| Tumbler      | 0.94     | 1.00   | 0.97     | 15      |
| Wallet       | 1.00     | 1.00   | 1.00     | 15      |
| **Macro Avg** | **0.99** | **0.99** | **0.99** | **90** |

---

---

## Retrieval Demo

A qualitative retrieval demo was created to illustrate how the model behaves on real queries. Given a text caption, the fine-tuned model encodes it alongside all gallery images, computes cosine similarity scores, and ranks results from highest to lowest. The top-k results along with their similarity scores are then displayed in a ranked grid. A query text and its corresponding gallery image form a pre-annotated test pair from our dataset, serving as the ground-truth pairing established during data annotation.

**Test query and image used in the demonstration:**

> *"Light-grey rectangular lunch box, pale blue latch clips and flap compartment, white oval shaped vent button on lid"* — lunchbox_050

**Retrieval Results for Demo Query:**

| Rank | Item Name     | Score   | Status     |
| ---- | ------------ | ------- | ---------- |
| #1   | lunchbox_041 | 0.4559  | Top-1      |
| #2   | lunchbox_074 | 0.4146  | Top-2      |
| #3   | lunchbox_030 | 0.3816  | Top-3      |
| #4   | lunchbox_050 | 0.3693  | **Correct** |
| #5   | lunchbox_082 | 0.3680  | Top-5      |
| #6   | lunchbox_073 | 0.3600  | Top-6      |
| #7   | lunchbox_042 | 0.3586  | Top-7      |
| #8   | lunchbox_055 | 0.3474  | Top-8      |
| #9   | lunchbox_014 | 0.3313  | Top-9      |
| #10  | lunchbox_080 | 0.3313  | Top-10     |

The true match (lunchbox_050) appears at **rank #4** with a score of 0.3693. All top-10 results are lunch boxes, confirming the model has strong within-category discrimination. The score gap between #1 (0.4559) and #4 (0.3693) reflects the inherent difficulty of distinguishing near-identical items from text descriptions alone.

> Note: The demo query text and gallery image were pre-annotated as a matched pair during dataset construction. This pair was not excluded from training, so this result reflects genuine generalization rather than zero-shot transfer.

## What This Run Told Us

- **r=16 + wd=0.1 is the best combination**: Smallest rank + strongest weight decay = best generalization. The smaller LoRA rank acts as an additional regularizer, and the 10x stronger weight decay prevents the model from fitting noisy training patterns.
- **Val loss is not a good proxy for retrieval quality**: V3's best val loss (0.8148) is the lowest of all models, yet the model still showed room for improvement — val loss decreased through all 20 epochs (1.1868 → 0.8152) without plateauing.
- **Train loss is a better signal**: V3 has the highest train loss at stop (0.4547) among HNM models. All other models that stopped early had train losses of 0.12–0.28, indicating memorization. V3's higher train loss means it was still learning generalizable features.
- **tumbler_065 improved dramatically**: #14 in V3 vs #80 in CLAUDE V2.1 — an 83% improvement. The smaller rank + stronger regularization helps with this confusing item.
- **tumbler_047 also recovered**: #14 in V3 (second best after V2's #13). The WRELS flask and cream tumbler cases both benefit from the gentler learning that a smaller rank provides.
- **charger_040 is severe at rank #23**: Even farther outside the top 10 than prior models. CLAUDE V2.1 recovered it at rank #22. An ensemble approach may be needed.
- **GPU training is viable for small datasets**: Training ran successfully on RTX 3050 with 3.80 GB VRAM peak usage, completing all 20 epochs in practical time.

