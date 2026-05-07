**Batch statistics** vary by configuration (see Table 3.4-2).

| Metric | Value |
|---|---|

#### Stage 2 — Dual-Stream Encoding

The batch size depends on the configuration (see Table 3.4-2). Each batch contains N images, N positive captions, and N hard-negative captions (2N total text descriptions). Both streams are encoded simultaneously via the LoRA-adapted FG-CLIP model.

**Vision encoder — ViT-B/16 architecture.** Each 224×224 RGB image is processed as follows:

1. **Patch embedding.** A Conv2d layer with kernel and stride of 16×16 divides the image into a 14×14 grid of 196 non-overlapping patches. Each patch is flattened into a 768-dimensional vector (3×16×16). The resulting sequence is 196×768.

2. **CLS token prepending.** A learnable `[CLS]` token is prepended to the sequence, producing a 197×768 tensor. This token aggregates global image information through self-attention and is used as the image-level embedding.

3. **Positional embedding.** A 197×768 learnable positional embedding is added to the sequence to encode spatial position.

4. **Transformer encoder.** The 197-token sequence passes through 12 identical transformer encoder blocks. Each block contains:
   - Multi-head self-attention (8 heads, each with dimension 768/8 = 96)
   - Layer norm and residual connection
   - Feed-forward network (3072 hidden units, GELU activation)
   - Another layer norm and residual connection

5. **Output projection.** The final layer norm is applied and the `[CLS]` token output is projected through a final linear layer to produce the 768-dimensional image embedding.

**LoRA integration.** LoRA adapters are injected into all four attention projection matrices — `q_proj`, `k_proj`, `v_proj`, and `out_proj` — of every transformer layer in both the vision and text encoders. For each frozen weight matrix W ∈ ℝ^{d×k}, two trainable matrices A ∈ ℝ^{r×k} and B ∈ ℝ^{d×r} are introduced. During the forward pass, the output is:

\[
W'(x) = W \cdot x + \frac{\alpha}{r} \cdot B \cdot A \cdot x
\]

Total trainable parameters depend on configuration (see Table 3.4-2). The frozen base weights preserve all pre-trained visual-language alignment.

**Text encoder.** All 16 captions are tokenized by the FG-CLIP tokenizer (vocabulary size 49,408, max sequence length 77 tokens). Token sequences shorter than 77 are padded; shorter sequences within a batch are padded to the longest sequence in that batch. The text encoder processes the padded token sequence through the same 12-layer transformer architecture (adapted with LoRA), producing 16 text embeddings — one per caption.

**Output of Stage 2.** Two embedding tensors: N normalized image embeddings and 2N normalized text embeddings, each of dimension 768, in the shared FG-CLIP embedding space.

#### Stage 3 — Similarity Computation and Contrastive Loss

The embeddings from Stage 2 are already normalized, but the model additionally applies explicit L2-normalization as a numerical safeguard before computing the image-text similarity matrix:

```
image_embeds = outputs.image_embeds / ||outputs.image_embeds||₂
text_embeds  = outputs.text_embeds  / ||outputs.text_embeds||₂
```

**Cosine similarity matrix.** The N×2N similarity matrix S is computed as the dot product of image and text embeddings:

\[
S_{ij} = \frac{\mathbf{i}_i \cdot \mathbf{t}_j}{\tau} = \frac{\mathbf{i}_i^\top \mathbf{t}_j}{\tau}
\]

where τ is the fixed temperature (see Table 3.4-2). This produces a matrix where row *i* contains image *i*'s similarity to all 2N captions — the first N entries are similarities to the positive captions, and the last N are similarities to the hard negatives.

**Contrastive cross-entropy loss.** The standard CLIP contrastive loss is applied as a symmetric cross-entropy over the similarity matrix:

```
labels = torch.arange(N).to(device)   # diagonal = correct matches
loss = CrossEntropyLoss(logits, labels)   # logits = S, shape (N, 2N)
```

PyTorch's `CrossEntropyLoss` applies softmax over the rows of the logits matrix, then computes the negative log-likelihood of the correct (diagonal) entry. The hard-negative captions are implicit negatives — they do not receive any explicit label; the model simply must assign them lower scores than the positive caption to minimize loss.

**Why the diagonal is the correct match.** In each batch, caption *j* (where j < N) is the positive description of image *j*, and caption *j* + N is its hard negative. The `labels = torch.arange(N)` tensor therefore maps image 0 → caption 0, image 1 → caption 1, and so on. The hard negatives never appear in the label vector — they are penalized only by having their scores pushed below the positive score.

#### Stage 4 — Backpropagation and Weight Update

**Gradient flow.** The contrastive loss scalar is backpropagated through the similarity computation, text encoder (LoRA layers), and vision encoder (LoRA layers). The frozen base model weights receive no gradients — only the LoRA adapter matrices A and B are updated.

**Optimizer: AdamW.** The AdamW optimizer (Adam with decoupled weight decay) maintains per-parameter momentum and adaptive learning rates. It is applied exclusively to the trainable LoRA parameters:

- Learning rate: **3e-5** (see Table 3.4-2; V1 uses a fixed LR with no scheduler)
- Weight decay: 0.01 (applied to LoRA weights only, not the frozen base)
- betas: (0.9, 0.999) — exponential moving average of gradients and squared gradients
- epsilon: 1e-8 — numerical stability constant in denominator

**Learning rate schedule.** A cosine decay schedule with linear warmup adjusts the learning rate over the full training run:

- **Phase 1 — Linear warmup (first 10% of steps):** LR linearly increases from 0 to the peak value. This stabilizes early training when LoRA matrices are near their zero-initialized starting point; a full learning rate applied to near-zero weights would produce arbitrarily large first-step updates.
- **Phase 2 — Cosine decay (remaining 90% of steps):** LR follows a smooth cosine curve from the peak value down to approximately half the peak value. This enables the model to take large, exploratory steps early in training while smoothly reducing the step size as it approaches the optimum.

**Early stopping.** After each training epoch, the model runs a full validation pass (no gradient computation). If the validation loss does not improve for a set number of consecutive epochs (configurable per configuration), training halts. The best checkpoint — the one with the lowest validation loss across all completed epochs — is preserved. This guards against overfitting on the training set.

**Checkpointing.** The LoRA adapter weights are saved using the Hugging Face PEFT `save_pretrained()` method, which writes `adapter_model.safetensors` and `adapter_config.json`. The base model is not duplicated; only the delta (LoRA weights) is stored.

#### Stage Progression Summary

| Stage | Input | Operation | Output |
|---|---|---|---|
| 1 | 420 dataset pairs | Categorical Batch Sampler (category-grouped; batch size varies by config) | N image-caption pairs per step |
| 2 | N images + 2N captions | LoRA-adapted dual-stream encoding (ViT-B/16 + text transformer) | N image embeds (768-d) + 2N text embeds (768-d) |
| 3 | All embeddings | L2-normalize → cosine sim → τ (config-dependent) → CE loss over (N×2N) matrix | Scalar loss |
| 4 | Loss scalar | AdamW + cosine LR (config-dependent) + 10% warmup + early stopping (config-dependent) | LoRA weight updates (ΔW = BA) |

### 3.2 Contrastive Loss

For a batch of N image-caption pairs, the model receives N images alongside 2N text descriptions (N positives + N hard negatives). All embeddings are L2-normalized before computing similarity logits, divided by a temperature parameter τ to sharpen the softmax distribution. The contrastive loss penalizes the model when a hard-negative description receives a higher similarity score to an image than its correct positive caption. This training setup is consistent with the hard-negative component of FG-CLIP's pre-training but applied to the campus lost and found domain.

### 3.3 Categorical Batch Mining (Visual Hard Negatives)

While the dataset provides textual hard negatives, standard randomized data loading fails to consistently challenge the vision encoder, as visually dissimilar items (e.g., a bag and a charger) are easily distinguished. To address this, we implemented a custom Categorical Batch Sampler for our advanced training configurations.

Instead of randomly shuffling the entire dataset, the sampler actively "mines" the dataset to construct batches consisting entirely of items from the same category (e.g., a batch of 16 highly similar lunchboxes). By forcing visually similar items into the same batch, the model is penalized for relying on macro-features like general shape or dominant color. This forces the vision encoder to visually map the micro-features detailed in the textual hard negatives (such as a specific latch or zipper) to lower the contrastive loss.

### 3.4 Experimental Configurations

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


**Table 3.4-2. Three experimental fine-tuning configurations of LoRA**

---

## 5. Key Design Decisions

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

## 6. Results

### 6.1 Retrieval Metrics


| Metric          | V1       | V2 + HNM   | V2.1 + HNM   |
| --------------- | -------- | ---------- | ------------ |
| **Recall@1**    | 71.11%   | 78.89%     | **80.00%**   |
| **Recall@5**    | 91.11%   | 91.11%     | **92.22%**   |
| **Recall@10**   | 93.33%   | **96.67%** | **96.67%**   |
| **MRR**         | 0.8017   | 0.8556     | **0.8615**   |
| Severe failures | 6 (6.7%) | 3 (3.3%)   | **3 (3.3%)** |


**Table 6.1-1. Key Retrieval Metrics across Fine-Tuning Configurations**

### 6.2 Improvement over Baseline


| Metric          | V1 → V2 + HNM | V2 + HNM → V2.1 + HNM | V1 → V2.1 + HNM |
| --------------- | ------------- | --------------------- | --------------- |
| R@1             | **+7.78 pp**  | +1.11 pp              | **+8.89 pp**    |
| R@5             | +0.00 pp      | +1.11 pp              | **+1.11 pp**    |
| R@10            | **+3.34 pp**  | +0.00 pp              | **+3.34 pp**    |
| MRR             | **+0.0539**   | +0.0059               | **+0.0598**     |
| Severe failures | −3            | 0                     | −3              |


**Table 6.1-2. Incremental Improvement across Configurations**

### 6.3 Failure Analysis

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


**Table 6.3-1. Severe Failure Item Ranks across Configurations**

Note: `tumbler_065` is the hardest item across all configurations — never in any top 10 — though it improved substantially across runs (V1: #113 → V2.1 + HNM: #86). `tumbler_047` regressed from V2 + HNM (rank 28) to V2.1 + HNM (rank 33), suggesting the cream/white tumbler class is sensitive to hyperparameter choices. `charger_040` oscillates between severe and recovered depending on configuration, sitting near the R@10 decision boundary.

---

## 7. Ablation Summary

The progression from V1 → V2 + HNM → V2.1 + HNM traces three distinct contributions:

1. **Architecture + HNM**: The jump from V1 to V2 + HNM (+7.78 pp R@1) demonstrates that scaling LoRA rank and target modules, combined with category-grouped batching and full-spectrum augmentation, substantially improves fine-grained retrieval. The cosine scheduler and early stopping prevent overfitting.
2. **HNM hyperparameter compensation**: The step from V2 + HNM to V2.1 + HNM (+1.11 pp R@1) demonstrates that category-grouped batching alone is not sufficient — it must be paired with the right learning dynamics (lower LR, higher α, softer temperature, smaller batch). These changes recovered the R@1 regression and produced the best MRR and validation loss of any configuration.
3. **Consistent category-level accuracy**: All three configurations maintained 100% category accuracy, confirming that the bottleneck is fine-grained within-category discrimination, not cross-category confusion.

---

## 8. Conclusion

Fine-tuning FG-CLIP with LoRA on a domain-specific lost-and-found dataset yielded strong improvements over the pre-trained baseline. The final configuration, V2.1 + HNM, achieved 80.00% Recall@1 and 0.8615 MRR on 90 held-out test queries against a 600-item gallery, representing an **+8.89 percentage-point improvement in R@1** over the baseline. Category-grouped hard negative mining, when combined with properly tuned hyperparameters (α=128, LR=3e-5, τ=0.03, batch=8), proved to be the most effective single change — recovering from the regression of naive HNM and producing the best overall retrieval quality.
