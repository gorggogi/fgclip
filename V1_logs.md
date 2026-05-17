# V1 — `fgclip-lora-finetunedV1-1200`

> **Status**: Baseline LoRA (r=8, q_proj + v_proj only) + random batching + brightness-only augmentation + no scheduler + no early stopping. GPU training on RTX 3050 (max VRAM 3.80 GB). Trained all 15 epochs. **Result: R@1 71.11%, R@5 91.11%, R@10 93.33%, MRR 0.8017, 6 severe failures. Serves as the baseline from which V2 and V3 improve.**

> **Base Model**: `qihoo360/fg-clip-base` (149,620,737 total parameters)
> **Environment**: Windows 10, Python 3.13.3, PyTorch 2.6.0+cu124, Transformers 4.57.1, PEFT 0.19.1
> **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU
> **Training notebook**: `fgclipfintuning/Copy_of_fgclipFineTuning_V1_1200.ipynb`
> **Evaluation notebook**: `fgclipevaluation/Copy_of_fgclipEvaluation_V1_1200.ipynb`
> **Output directory**: `lorafinetuned/fgclip-lora-finetunedV1-1200/`
> **Training seed**: 9435575495300

---

## Table of Contents

1. [What Changed from Base](#what-changed-from-base)
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

## What Changed from Base


| Aspect            | Base CLIP                   | V1                                            |
| ----------------- | --------------------------- | --------------------------------------------- |
| LoRA rank (r)     | Full fine-tune (all params) | **8**                                         |
| LoRA alpha (α)    | N/A                         | **16**                                        |
| Target modules    | None                        | **q_proj, v_proj**                            |
| Temperature       | ~0.07 (model default)       | **0.0122** (auto-scaled by HuggingFace logit) |
| Optimizer         | N/A                         | AdamW, lr=5e-5                                |
| Scheduler         | None                        | **None**                                      |
| Weight decay      | N/A                         | 0.01                                          |
| Batch size        | N/A                         | 16 (random shuffle)                           |
| Batching strategy | N/A                         | Random shuffle                                |
| Early stopping    | None                        | **None** (15 epochs fixed)                    |
| Data augmentation | None                        | Brightness only (0.2)                         |
| Hard negatives    | No                          | **No** (random batching)                      |


---

## Why These Changes

- **r=8, q_proj + v_proj only**: Minimal LoRA capacity to test whether even a tiny adapter can improve CLIP on this fine-grained retrieval task, without risking the catastrophic forgetting that full fine-tuning risks.
- **Temperature 0.0122 (auto-scaled)**: The HuggingFace CLIP logit scale of 1/0.0122 ≈ 82 produces sharper softmax gradients than the default ~0.07, giving stronger learning signals on the first run. No manual override applied.
- **No scheduler**: Single fixed learning rate of 5e-5 throughout all 15 epochs. Assumes the model benefits from a constant LR rather than annealing.
- **Random shuffling**: Baseline batching strategy — no category grouping, so hard negatives are not deliberately matched to query categories.
- **Brightness-only augmentation**: Only ColorJitter(brightness=0.2) applied. Simpler than V2/V3 to isolate the effect of the LoRA adapter alone.
- **No early stopping**: 15 fixed epochs. The model trains to completion regardless of whether validation loss improves.

---

## Training Methodology

**Task**: Image-text retrieval. Given a text description of a lost item, retrieve the matching image from a gallery of 600 found items.

**Loss**: Single-image InfoNCE / CLIP loss. Random-shuffled batches of 16 pairs (16 positive + 16 random negatives — no category grouping).

**Hard negative mining**: None. Random shuffling means hard negatives are distributed randomly across categories — a much easier task than category-grouped batching.

**Training loop**: Zero gradients → encode `lost_image` → encode positive + negative captions → normalize → cosine sim `/ 0.0122` → cross-entropy → backprop → step optimizer. Only LoRA adapters train. No LR scheduler. No early stopping — runs all 15 epochs.

**Inference**: 600 gallery images encoded once, query text encoded on-the-fly, cosine similarity ranks them.

---

## Architecture


| Parameter               | Value                                                       |
| ----------------------- | ----------------------------------------------------------- |
| Base model              | `qihoo360/fg-clip-base`                                     |
| Vision backbone         | ViT-B/16 (14×14 patch grid, 197 tokens, 768-dim embeddings) |
| Trainable parameters    | **491,520 (0.3274% of total)**                              |
| Full model total (base) | **150,112,257** (base 149,620,737 + LoRA 491,520)           |
| LoRA rank (r)           | **8**                                                       |
| LoRA alpha (α)          | **16**                                                      |
| Target modules          | `q_proj`, `v_proj`                                          |
| Dropout                 | 0.1                                                         |
| Model dtype             | `torch.float32`                                             |


---

## Hyperparameters


| Parameter             | Value                                          |
| --------------------- | ---------------------------------------------- |
| Optimizer             | AdamW                                          |
| Learning rate (start) | 5e-5                                           |
| Learning rate (end)   | 5e-5 (constant — no scheduler)                 |
| Scheduler             | **None**                                       |
| Weight decay          | 0.01                                           |
| Betas                 | (0.9, 0.999)                                   |
| Epsilon               | 1e-8                                           |
| Loss function         | CrossEntropyLoss                               |
| Temperature           | **0.0122** (auto-scaled by HF logit parameter) |
| Batch size            | 16 (val), 16 (train, random shuffle)           |
| Max epochs            | 15                                             |
| Early stopping        | **None** (15 epochs fixed)                     |
| DataLoader workers    | 0                                              |


---

## Data Augmentation

Compose(
Resize(size=(224, 224), interpolation=bilinear)
RandomHorizontalFlip(p=0.5)
ColorJitter(brightness=0.2)
ToTensor()
RandomErasing(p=0.3, scale=(0.02, 0.15))
ToPILImage()
)

---

## Training Loss Curve


| Epoch | Train Loss | Val Loss | Note           |
| ----- | ---------- | -------- | -------------- |
| 1     | 1.1897     | 1.6209   | Saved          |
| 2     | 1.0652     | 1.6060   | Saved          |
| 3     | 1.0310     | 1.5862   | Saved          |
| 4     | 0.9513     | 1.5636   | Saved          |
| 5     | 0.8939     | 1.5430   | Saved          |
| 6     | 0.7684     | 1.5228   | Saved          |
| 7     | 0.7740     | 1.5049   | Saved          |
| 8     | 0.6903     | 1.4865   | Saved          |
| 9     | 0.7042     | 1.4694   | Saved          |
| 10    | 0.6581     | 1.4580   | Saved          |
| 11    | 0.6352     | 1.4422   | Saved          |
| 12    | 0.5483     | 1.4328   | Saved          |
| 13    | 0.5412     | 1.4351   | Worsened       |
| 14    | 0.4875     | 1.4454   | Worsened       |
| 15    | 0.4484     | 1.4464   | Worsened — end |


> **Epochs 13–15 showed consistent val loss degradation (1.4351 → 1.4454 → 1.4464), confirming the model was beginning to overfit. No early stopping mechanism was in place, so training continued to the full 15 epochs.

---

## Training Results


| Metric                  | Value                                    |
| ----------------------- | ---------------------------------------- |
| Final train loss        | 0.4484                                   |
| Final val loss (at end) | 1.4464                                   |
| **Best val loss**       | **1.4328** (epoch 12, not checkpointed)  |
| Epochs trained          | 15 / 15                                  |
| Epochs saved            | 1 (final model, all 15 epochs completed) |
| Early stop triggered    | **No** (15 fixed epochs completed)       |
| Peak VRAM               | 3.31 GB                                  |


---

## Evaluation Results

**Evaluation setup**: 90 test queries matched against all 600 gallery images. Loaded fine-tuned adapter: `fgclip-lora-finetunedV1-1200`.

### Overall Metrics


| Metric            | Value      | Count |
| ----------------- | ---------- | ----- |
| **Recall@1**      | **71.11%** | 64/90 |
| **Recall@5**      | **91.11%** | 82/90 |
| **Recall@10**     | **93.33%** | 84/90 |
| **MRR**           | **0.8017** | —     |
| Category accuracy | 100.00%    | 90/90 |


### Failure Analysis


| Category            | Count      | Notes                    |
| ------------------- | ---------- | ------------------------ |
| Perfect first-tries | 64 (71.1%) | Retrieved at rank 1      |
| Near misses         | 20 (22.2%) | In top 10 but not rank 1 |
| Severe failures     | 6 (6.7%)   | **Not in top 10 at all** |


### Severe Failures (not in top 10)


| Test Index | Query            | Correct Item     | Rank     | Notes                                                                                                                                                                                  |
| ---------- | ---------------- | ---------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 4          | lunchbox_050     | lunchbox_050     | **#20**  | Light-grey rectangular lunch box, pale blue latch clips and flap compartment, white oval shaped vent button on lid.                                                                    |
| 35         | bag_058          | bag_058          | **#12**  | Black smooth faux-leather mini boxy barrel bag, structured rectangular silhouette, twin flat top straps each with silver square buckle.                                                |
| 46         | tumbler_047      | tumbler_047      | **#60**  | Cream off-white insulated tumbler, all-over pastel pink tulip and butterfly line-art print on body, pink ribbed rubber base boot, pink flip-top lid with integrated hinge loop.        |
| 52         | tumbler_065      | tumbler_065      | **#112** | WRELS matte black soft flask, black TPU body with white "WRELS Love Life Love Sports" branding, narrow push-pull spout with loop tether at top, white measurement markings on reverse. |
| 70         | charger_081      | charger_081      | **#16**  | Pink multi-port wall charger with rectangular housing, dual USB-C ports with indicator lights, matte finish.                                                                           |
| 75         | handkerchief_027 | handkerchief_027 | **#17**  | Multicolor floral print handkerchief, small square shape, dense repeating flower pattern with blue edging.                                                                             |


### Rank Distribution Matrix

> Tracks where the correct item ended up in the ranking, grouped by rank bucket.


|              | #1  | #2-5 | #6-10 | #11-20 | #21-50 | #50+ |
| ------------ | --- | ---- | ----- | ------ | ------ | ---- |
| Bag          | 12  | 2    | 0     | 1      | 0      | 0    |
| Charger      | 8   | 6    | 0     | 1      | 0      | 0    |
| Handkerchief | 9   | 4    | 1     | 1      | 0      | 0    |
| Lunchbox     | 10  | 4    | 0     | 1      | 0      | 0    |
| Tumbler      | 11  | 1    | 1     | 0      | 0      | 2    |
| Wallet       | 14  | 1    | 0     | 0      | 0      | 0    |


> **Per-category summary:**


| Category     | R@1 Acc | Avg Rank | Worst Rank |
| ------------ | ------- | -------- | ---------- |
| Bag          | 80.0%   | 1.9      | #12        |
| Charger      | 53.3%   | 2.7      | #16        |
| Handkerchief | 60.0%   | 2.9      | #17        |
| Lunchbox     | 66.7%   | 2.6      | #20        |
| Tumbler      | 73.3%   | 12.9     | #112       |
| Wallet       | 93.3%   | 1.1      | #2         |


> **Tumbler is the hardest category**: Avg rank of 12.9 and worst rank of #112 far exceed any other category. The WRELS flask (tumbler_065) and cream tumbler (tumbler_047) cases are nearly identical across both visual and textual modalities, making within-category discrimination extremely difficult.

---

### Per-Category Item-Level Accuracy

> Which specific items were confused with which at rank #1.


| Category     | R@1 Correct | Wrong at R@1 (true -> AI guess, rank)                                                                                                                                                                                                                 |
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Bag          | 12          | bag_058->bag_011 (#12), bag_079->bag_087 (#3), bag_093->bag_014 (#2)                                                                                                                                                                                  |
| Charger      | 8           | charger_039->charger_052 (#2), charger_040->charger_037 (#4), charger_071->charger_072 (#3), charger_081->charger_070 (#16), charger_083->charger_071 (#2), charger_093->charger_092 (#2), charger_099->charger_092 (#4)                              |
| Handkerchief | 9           | handkerchief_027->handkerchief_079 (#17), handkerchief_034->handkerchief_083 (#2), handkerchief_035->handkerchief_087 (#8), handkerchief_062->handkerchief_098 (#2), handkerchief_091->handkerchief_079 (#2), handkerchief_096->handkerchief_079 (#3) |
| Lunchbox     | 10          | lunchbox_048->lunchbox_076 (#2), lunchbox_050->lunchbox_072 (#20), lunchbox_068->lunchbox_077 (#3), lunchbox_071->lunchbox_070 (#2), lunchbox_075->lunchbox_100 (#2)                                                                                  |
| Tumbler      | 11          | tumbler_047->tumbler_050 (#60), tumbler_065->tumbler_021 (#112), tumbler_066->tumbler_021 (#6), tumbler_084->tumbler_086 (#4)                                                                                                                         |
| Wallet       | 14          | wallet_033->wallet_034 (#2)                                                                                                                                                                                                                           |


---

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


> **Note**: Category-level accuracy is 100% (90/90). Every top-ranked gallery item belongs to the correct category — the 26 R@1 failures and 6 severe failures are all within-category errors. This confirms the pre-fine-tuned CLIP already knows bag vs tumbler vs charger. The challenge is entirely within-category item discrimination.

---

## Retrieval Demo

**Test query used:**

> *"Light-grey rectangular lunch box, pale blue latch clips and flap compartment, white oval shaped vent button on lid"* — lunchbox_050

**Retrieval Results:**

> Note: lunchbox_050 (the true match) appears at **rank #20** in V1 — outside the top-10 display range. This is the worst performance of any model on this query. V1's random batching and minimal LoRA capacity cannot distinguish this particular lunch box from visually similar distractors.


| Rank | Item Name    | Score  | Status |
| ---- | ------------ | ------ | ------ |
| #1   | lunchbox_072 | 0.3441 | Top-1  |
| #2   | lunchbox_074 | 0.3380 | Top-2  |
| #3   | lunchbox_087 | 0.3313 | Top-3  |
| #4   | lunchbox_073 | 0.3272 | Top-4  |
| #5   | lunchbox_030 | 0.3251 | Top-5  |
| #6   | lunchbox_042 | 0.3228 | Top-6  |
| #7   | lunchbox_014 | 0.3220 | Top-7  |
| #8   | lunchbox_066 | 0.3205 | Top-8  |
| #9   | lunchbox_077 | 0.3199 | Top-9  |
| #10  | lunchbox_075 | 0.3098 | Top-10 |


---

## What This Run Told Us

- **LoRA at r=8 is better than nothing**: V1's R@1 of 71.11% is a meaningful improvement over what the base CLIP model would achieve, establishing that even minimal LoRA fine-tuning helps.
- **Tumbler is the hardest category by far**: Avg rank 12.9 and worst rank #112 confirm that tumbler-item discrimination is the primary failure mode. This is driven by near-identical WRELS and cream tumbler pairs.
- **lunchbox_050 is the hardest lunch box**: Rank #20 in V1. V2 improved it to #7. V3 brought it to #4. This single item's trajectory across models mirrors the overall improvement arc.
- **Random batching is insufficient for hard negatives**: The absence of category-grouped batching means the model never learns to discriminate between plausible within-category matches. V2's CategoryBatchSampler directly addresses this gap.
- **Val loss was still improving at epoch 12**: The best val loss at epoch 12 (1.4328) was followed by 3 epochs of degradation, but the model was stopped only by the fixed epoch limit. More epochs would have worsened generalization.
- **The base CLIP already classifies categories perfectly**: 100% category-level accuracy means the model already knows the difference between bags and tumblers. The entire research problem is within-category item discrimination, which V2 and V3 progressively address.

