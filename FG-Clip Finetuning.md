# FG-CLIP Fine-Tuning

> **Base Model**: `qihoo360/fg-clip-base` (150,112,257 total parameters)
> **Environment**: Windows 10, Python 3.13.3, PyTorch 2.6.0+cu124, Transformers 4.57.1, PEFT 0.19.1
> **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU

---

## 1. Low-Rank Adaptation (LoRA)

Full fine-tuning of large pre-trained models carries a high risk of catastrophic forgetting on small datasets. To address this, we utilize a technique called Parameter Efficient Fine-Tuning (PEFT), specifically, Low-Rank Adaptation (LoRA) (Hu et al., 2022), which freezes the original model weights and injects trainable low-rank matrices into each attention layer:

W' = W + ΔW = W + BA

Where W represents the frozen pre-trained weight matrix, while B and A are the newly introduced trainable decomposition matrices. These are smaller matrices that act as a bottleneck.

Hypothetically, if an attention layer has a dimension of 10,000 × 10,000, LoRA reduces computational load by restricting the parameter updates to a much smaller subspace (rank r). If r = 32:

- Matrix A: Projects the input dimension down to the lower-rank space (10,000 × 32), resulting in 320,000 parameters.
- Matrix B: Projects the lower-rank representation back up to the original dimensionality (32 × 10,000), resulting in 320,000 parameters.

This gives a total of 640,000 trainable parameters instead of updating millions of parameters across the full architecture.

When applied to our experimental configuration, the efficiency of this approach was demonstrated clearly. The FG-CLIP base architecture has a total of 150,112,257 parameters. Through LoRA, our V1 configuration required updating only 491,520 parameters (0.33% of the total), and our later configurations updated 3,932,160 parameters (2.56% of the total). This prevents the neural network from severely overfitting to the small dataset while preserving the model's pre-trained knowledge.

---

## 2. Dataset and Data Preparation

### 2.1 Dataset Construction

Training consisted of 420 image pairs (840 images total) gathered from e-commerce platforms such as Facebook Marketplace, Shopee, and Carousell. This approach helped capture the variable lighting, backgrounds, and camera angles typical of real-world lost and found images. Each pair consists of one image labeled as the "lost" item, and another image of the same item labeled as the "found" item. The dataset included 6 categories: Bags, Chargers, Handkerchiefs, Lunchboxes, Tumblers, and Wallets.

The dataset was partitioned as follows:


| Category      | Total Pairs | Train (70%) | Validation (20%) | Test (10%) |
| ------------- | ----------- | ----------- | ---------------- | ---------- |
| Bags          | 100         | 70          | 15               | 15         |
| Chargers      | 100         | 70          | 15               | 15         |
| Handkerchiefs | 100         | 70          | 15               | 15         |
| Lunchboxes    | 100         | 70          | 15               | 15         |
| Tumblers      | 100         | 70          | 15               | 15         |
| Wallets       | 100         | 70          | 15               | 15         |
| **Total**     | **600**     | **420**     | **90**           | **90**     |


**Table 3.2-31. Data Split per Category (Train–Validation–Test)**

### 2.2 Caption Generation and Textual Hard Negatives

For image captioning, the researchers utilized a vision-language model (VLM), Claude AI, to generate descriptions for all 840 images, which were then verified by a human annotator. Each caption follows a structured attribute taxonomy:

`[Color] + [Brand/Text] + [Specific Category] + [Condition/Distinguishing Feature]`

For example: *"Pink Guess compact zip-around wallet, quilted diamond-stitched faux leather, silver triangle GUESS logo plate centered on front flap."*

For each positive caption (an accurate description of the item), a corresponding negative caption was constructed by modifying one defining attribute or feature such as color, material, hardware, texture, or shape, while keeping the rest of the caption identical. This serves as our Textual Hard Negative, forcing the model's text encoder to differentiate at a microscopic level rather than relying on broad semantic categories.

**Intersection Rule**

An intersection rule was introduced to control which attributes appear in the positive captions. The rule dictates that only features visible in both images of a pair are described. This prevents the text encoder from referencing attributes the vision encoder cannot observe, preventing model hallucination and ensuring that both encoders are grounded in mutually visible evidence.

### 2.3 Data Augmentation

Training images pass through an augmentation pipeline to simulate the variability of typical user-submitted lost and found photographs:

- Resize to 224 × 224
- Random horizontal flip (p = 0.5): simulates different shooting orientations
- Color jitter: changes brightness, contrast, saturation, and hue to account for camera and lighting variation
- Random erasing (p = 0.3, scale = [0.02, 0.15]): randomly masks a rectangular region, preventing over-reliance on a single salient feature (e.g., a brand logo) and encouraging the model to leverage shape, texture, and material cues holistically

Validation and test images receive only the resize operation to ensure unbiased performance evaluation.

---

## 3. Training Methodology and Experimental Configurations

### 3.1 Contrastive Loss

For a batch of N image-caption pairs, the model receives N images alongside 2N text descriptions (N positives + N hard negatives). All embeddings are L2-normalized before computing similarity logits, divided by a temperature parameter τ to sharpen the softmax distribution. The contrastive loss penalizes the model when a hard-negative description receives a higher similarity score to an image than its correct positive caption. This training setup is consistent with the hard-negative component of FG-CLIP's pre-training but applied to the campus lost and found domain.

### 3.1.2 Categorical Batch Mining (Visual Hard Negatives)

While the dataset provides textual hard negatives, standard randomized data loading fails to consistently challenge the vision encoder, as visually dissimilar items (e.g., a bag and a charger) are easily distinguished. To address this, we implemented a custom Categorical Batch Sampler for our advanced training configurations.

Instead of randomly shuffling the entire dataset, the sampler actively "mines" the dataset to construct batches consisting entirely of items from the same category (e.g., a batch of 16 highly similar lunchboxes). By forcing visually similar items into the same batch, the model is penalized for relying on macro-features like general shape or dominant color. This forces the vision encoder to visually map the micro-features detailed in the textual hard negatives (such as a specific latch or zipper) to lower the contrastive loss.

### 3.2 Experimental Configurations

Three fine-tuning configurations were developed to trace the impact of each design decision:


| Hyperparameter       | V1                                                                | V2 + HNM                                                | V2.1 + HNM                                                                        |
| -------------------- | ----------------------------------------------------------------- | ------------------------------------------------------- | --------------------------------------------------------------------------------- |
| **Architecture**     |                                                                   |                                                         |                                                                                   |
| LoRA rank (r)        | 8                                                                 | 32                                                      | 32                                                                                |
| LoRA alpha (α)       | 16                                                                | 64                                                      | **128**                                                                           |
| Target modules       | qproj, vproj                                                      | qproj, kproj, vproj, outproj                            | qproj, kproj, vproj, outproj                                                      |
| LoRA dropout         | 0.1                                                               | 0.1                                                     | 0.1                                                                               |
| Trainable parameters | 491,520 (0.33%)                                                   | 3,932,160 (2.56%)                                       | 3,932,160 (2.56%)                                                                 |
| **Optimizer**        |                                                                   |                                                         |                                                                                   |
| Optimizer            | AdamW                                                             | AdamW                                                   | AdamW                                                                             |
| Learning rate        | 5e-5 (fixed)                                                      | 5e-5                                                    | **3e-5**                                                                          |
| Scheduler            | None                                                              | Cosine + 10% warmup                                     | Cosine + 10% warmup                                                               |
| Weight decay         | 0.01                                                              | 0.01                                                    | 0.01                                                                              |
| **Loss**             |                                                                   |                                                         |                                                                                   |
| Temperature (τ)      | ~0.07 (model default)                                             | 0.02 (fixed)                                            | **0.03** (fixed)                                                                  |
| Loss type            | Single-image                                                      | Single-image                                            | Single-image                                                                      |
| **Training**         |                                                                   |                                                         |                                                                                   |
| Batch size           | 16                                                                | 16                                                      | **8**                                                                             |
| Batching strategy    | Random shuffle                                                    | **Category-grouped**                                    | **Category-grouped**                                                              |
| Early stopping       | None                                                              | Patience = 3                                            | **Patience = 5**                                                                  |
| Max epochs           | 15                                                                | 20                                                      | 20                                                                                |
| Color jitter         | (brightness=(0.8, 1.2), contrast=None, saturation=None, hue=None) | (brightness=0.2, contrast=0.2, saturation=0.4, hue=0.1) | *(brightness***=**0.4*, contrast***=**0.4*, saturation***=**0.4*, hue***=**0.1*)* |


**Table 3.2-32. Three experimental fine-tuning configurations of LoRA**

---

## 4. Key Design Decisions

### V1 → V2 + HNM: Architecture and Training Upgrades

V1 established a baseline using conservative LoRA settings (rank 8, q/v projections only, no scheduler, brightness-only augmentation). V2 + HNM introduced a suite of changes to the architecture, training procedure, and data pipeline:

**LoRA rank 8 → 32, q/v → q/k/v/out.** Expanding both the rank and target modules gives the adapter significantly more capacity to learn fine-grained visual-textual alignment. Adding kproj and outproj enables the adapter to directly modify the query-key computation — how the model decides which tokens to attend to — rather than only adjusting the value aggregation that follows. This increases trainable parameters from ~0.33% to ~2.56% of the full model.

**Random shuffling → Category-grouped batching.** The custom Categorical Batch Sampler groups pairs by category prefix (e.g. `bag_012` → `"bag"`). Each category's shuffled index list is chunked into batches of 16. Every training batch therefore contains 16 pairs from the same category, with all 16 hard negatives being plausible matches for each image — creating a much harder discrimination task than random batching.

**Temperature 0.07 → 0.02.** A lower temperature sharpens the softmax distribution, resulting in steeper gradient signals and a stricter penalty for placing probability mass on hard negatives.

**Fixed LR → Cosine scheduler with warmup.** A 10% linear warmup period stabilizes early training when the LoRA matrices are near-zero initialization. The cosine decay enables smooth convergence without abrupt learning rate drops.

**No early stopping → Patience = 3.** Training halts when validation loss fails to improve for three consecutive epochs, preserving the best-generalization checkpoint rather than the final one.

**Brightness-only → Full-spectrum ColorJitter.** V1 only jittered brightness, leaving contrast and saturation invariant. V2 + HNM applies brightness, contrast, saturation, and hue jitter, giving a more uniform robustness envelope against real-world lighting variation.

The combined effect was a substantial improvement from V1: **+7.78 pp on R@1** (71.11% → 78.89%), **+3.34 pp on R@10** (93.33% → 96.67%), and **−3 severe failures** (6 → 3).

### V2 + HNM → V2.1 + HNM: Optimizing the Training Regime

While V2 + HNM significantly outperformed V1, its category-grouped batching and larger LoRA capacity introduced a harder per-step learning problem. V2.1 + HNM applied targeted parameter adjustments to unlock the full potential of the architecture:

**α = 64 → 128.** Doubling alpha relative to r=32 amplifies the LoRA contribution without adding parameters. At α=128, the LoRA pathway contributes 4× the base model's effective scaling, giving the adapter much more representational power to handle the harder same-category batches.

**LR 5e-5 → 3e-5.** A lower learning rate with a larger alpha prevents the stronger LoRA signal from destabilizing training.

**Temperature 0.02 → 0.03.** Softer logits at 0.03 reduce overconfidence, especially helpful with the smaller batch size (8) and noisier gradient signal per step.

**Batch size 16 → 8.** Halving the batch size halves the number of hard negatives per training step, reducing per-step difficulty and allowing more effective gradient signal per pair.

**Patience 3 → 5.** With more steps per epoch (53 vs ~27) and a lower LR, the model converges more gradually. Patience 5 gives it enough runway to find the true optimum.

**Balanced ColorJitter.** All channels set to (0.6–1.4) for brightness, contrast, and saturation, giving a uniform robustness envelope.

The result: V2.1 + HNM recovered from the training difficulty introduced by V2 + HNM's harder batches, matching V2 + HNM's R@10 (96.67%) while restoring R@1 to 80.00% and producing the best MRR (0.8615) and validation loss (1.0251) of any configuration.

---

## 5. Results

### 5.1 Retrieval Metrics


| Metric          | V1       | V2 + HNM   | V2.1 + HNM   |
| --------------- | -------- | ---------- | ------------ |
| **Recall@1**    | 71.11%   | 78.89%     | **80.00%**   |
| **Recall@5**    | 91.11%   | 91.11%     | **92.22%**   |
| **Recall@10**   | 93.33%   | **96.67%** | **96.67%**   |
| **MRR**         | 0.8017   | 0.8556     | **0.8615**   |
| Severe failures | 6 (6.7%) | 3 (3.3%)   | **3 (3.3%)** |


**Table 5.1-1. Key Retrieval Metrics across Fine-Tuning Configurations**

### 5.2 Improvement over Baseline


| Metric          | V1 → V2 + HNM | V2 + HNM → V2.1 + HNM | V1 → V2.1 + HNM |
| --------------- | ------------- | --------------------- | --------------- |
| R@1             | **+7.78 pp**  | +1.11 pp              | **+8.89 pp**    |
| R@5             | +0.00 pp      | +1.11 pp              | **+1.11 pp**    |
| R@10            | **+3.34 pp**  | +0.00 pp              | **+3.34 pp**    |
| MRR             | **+0.0539**   | +0.0059               | **+0.0598**     |
| Severe failures | −3            | 0                     | −3              |


**Table 5.1-2. Incremental Improvement across Configurations**

### 5.3 Failure Analysis

All three configurations maintained **100% category accuracy** — the model never confused one category for another. All failures were fine-grained within-category confusions.


| Category                   | Count      | Notes                    |
| -------------------------- | ---------- | ------------------------ |
| Perfect first-tries (V2.1) | 72 (80.0%) | Retrieved at rank 1      |
| Near misses (V2.1)         | 15 (16.7%) | In top 10 but not rank 1 |
| Severe failures (V2.1)     | 3 (3.3%)   | Not in top 10 at all     |


#### Severe Failure Items (V2.1 + HNM)


| Test Item     | Description                                                     | V1 Rank | V2 + HNM Rank | V2.1 + HNM Rank |
| ------------- | --------------------------------------------------------------- | ------- | ------------- | --------------- |
| `tumbler_047` | Cream off-white insulated tumbler, pastel tulip/butterfly print | #60     | #28           | #33             |
| `tumbler_065` | WRELS matte black soft flask                                    | #113    | #101          | #86             |
| `charger_040` | White QOOVI 22.5W wall charger                                  | #4 ✅    | #8 ✅          | #16             |


**Table 5.1-3. Severe Failure Item Ranks across Configurations**

Note: `tumbler_065` is the hardest item across all configurations — never in any top 10 — though it improved substantially across runs (V1: #113 → V2.1 + HNM: #86). `tumbler_047` regressed from V2 + HNM (rank 28) to V2.1 + HNM (rank 33), suggesting the cream/white tumbler class is sensitive to hyperparameter choices. `charger_040` oscillates between severe and recovered depending on configuration, sitting near the R@10 decision boundary.

---

## 6. Ablation Summary

The progression from V1 → V2 + HNM → V2.1 + HNM traces three distinct contributions:

1. **Architecture + HNM**: The jump from V1 to V2 + HNM (+7.78 pp R@1) demonstrates that scaling LoRA rank and target modules, combined with category-grouped batching and full-spectrum augmentation, substantially improves fine-grained retrieval. The cosine scheduler and early stopping prevent overfitting.
2. **HNM hyperparameter compensation**: The step from V2 + HNM to V2.1 + HNM (+1.11 pp R@1) demonstrates that category-grouped batching alone is not sufficient — it must be paired with the right learning dynamics (lower LR, higher α, softer temperature, smaller batch). These changes recovered the R@1 regression and produced the best MRR and validation loss of any configuration.
3. **Consistent category-level accuracy**: All three configurations maintained 100% category accuracy, confirming that the bottleneck is fine-grained within-category discrimination, not cross-category confusion.

---

## 7. Conclusion

Fine-tuning FG-CLIP with LoRA on a domain-specific lost-and-found dataset yielded strong improvements over the pre-trained baseline. The final configuration, V2.1 + HNM, achieved 80.00% Recall@1 and 0.8615 MRR on 90 held-out test queries against a 600-item gallery, representing an **+8.89 percentage-point improvement in R@1** over the baseline. Category-grouped hard negative mining, when combined with properly tuned hyperparameters (α=128, LR=3e-5, τ=0.03, batch=8), proved to be the most effective single change — recovering from the regression of naive HNM and producing the best overall retrieval quality.