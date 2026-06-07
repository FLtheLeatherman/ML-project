# Poster Outline

Title suggestion:

**Bridging the Gap Between Numerical and Categorical Objectives in Time-Series Foundation Models**

Subtitle suggestion:

Loss-level adaptation for Chronos-T5 and head-level adaptation for TimesFM 2.5.

## 1. Background

Core message:

Modern time-series foundation models transfer pretrained sequence models to forecasting, but they make different choices about how future values are represented and optimized.

Content to include:

- Classical forecasting predicts continuous future values, often optimized with numerical losses.
- Transformer-based forecasting foundation models reuse large-scale sequence modeling ideas.
- Chronos converts scaled values into discrete tokens and trains with cross-entropy.
- TimesFM keeps continuous outputs and predicts point / quantile forecasts.
- This creates a central objective-design question: should forecasting be treated as regression, classification, or a combination of both?

Suggested visual:

- Small two-branch diagram:
  - Chronos: real values -> scaling -> bins/tokens -> CE
  - TimesFM: real values -> patches -> continuous forecast head

## 2. Terminologies

Core message:

Define only the terms needed to read the poster. Keep this section compact.

Terms to include:

- **Time-series foundation model**: pretrained model that transfers to new forecasting datasets.
- **Point forecast**: one predicted value for each future step.
- **Probabilistic forecast**: distribution, samples, or quantiles over future values.
- **WQL**: weighted quantile loss; lower means better probabilistic forecasting.
- **MASE**: mean absolute scaled error; lower means better point forecasting.
- **Cross-entropy (CE)**: classification loss over discrete bins/tokens.
- **LoRA**: parameter-efficient fine-tuning with low-rank adapters.
- **RevIN**: reversible instance normalization used in TimesFM preprocessing.

Suggested visual:

- A small glossary box with 6-8 one-line definitions.

## 3. Task

Core message:

We study how changing the objective or output head affects pretrained time-series models on downstream forecasting.

Task statement:

- Input: historical context sequence.
- Output: 24-step forecast on `tourism_monthly_dataset`.
- Metrics: WQL and MASE.
- Models:
  - Chronos-T5: `amazon/chronos-t5-base`
  - TimesFM 2.5: `google/timesfm-2.5-200m-pytorch`

Research questions:

- For Chronos, can distance-aware losses improve over plain CE fine-tuning?
- For TimesFM, can a CE classification head improve a continuous forecasting backbone?
- Do WQL and MASE prefer the same objective?

Suggested visual:

- One compact input-output forecasting diagram:
  - context window -> model -> 24-step forecast -> WQL / MASE

## 4. Challenges

Core message:

The two model families expose opposite but related difficulties.

Challenges to include:

- **Classification ignores distance**: in plain CE, nearby and far-away wrong bins are both wrong classes.
- **Regression can be conservative**: numerical losses can favor average-looking forecasts.
- **Objective and inference mismatch**: Chronos trains token logits but forecasts through autoregressive sampling.
- **Head replacement is unstable**: TimesFM CE heads are newly initialized and much larger than a small adapter.
- **Learning-rate sensitivity**: CE head LoRA, especially CE256, can overfit with high head learning rates.
- **Reproducibility constraints**: pretrained weights and Monash data are external; evaluation depends on fixed splits and checkpoints.

Suggested visual:

- Two-column challenge box:
  - Chronos side: CE lacks ordinal distance.
  - TimesFM side: CE head needs pretraining and careful downstream LR.

## 5. Our Approach

Core message:

We test two complementary adaptation routes.

### Route A: Loss-Level Adaptation for Chronos

Keep Chronos architecture fixed and change only the LoRA fine-tuning loss.

Losses compared:

- CE baseline
- bin-index MSE
- bin-MSE + CE
- Wasserstein-style W1 / W2
- CRPS-style CDF loss
- Ordinal CE
- Huber loss in bin-index space

Key implementation idea:

- Compute losses in Chronos bin-index space after masking invalid / NaN targets.
- Normalize distance losses so they are comparable to CE scale.

### Route B: Head-Level Adaptation for TimesFM

Keep TimesFM backbone and compare original continuous head with CE classification heads.

Variants compared:

- Base zero-shot
- Base LoRA with original regression objective
- Original forecast head trained from scratch
- 64-bin CE head
- 256-bin CE head
- Staged large-pool 64-bin CE head

Key implementation idea:

- Build bins in RevIN-normalized space.
- Train CE head with frozen backbone first.
- Fine-tune downstream with LoRA and small head learning rate.
- Convert CE probabilities back to real forecasts using bin-center expectation and CDF inversion.

Suggested visual:

- Main method figure with two horizontal pipelines:
  - Chronos pipeline: pretrained Chronos -> LoRA -> alternative losses -> forecast
  - TimesFM pipeline: pretrained TimesFM -> CE head pretrain -> downstream LoRA -> forecast

## 6. Experiments

Core message:

The best objective depends on the model architecture and metric.

### Chronos Setup

- Dataset: `tourism_monthly_dataset`
- Split: 293 train / 73 eval
- Prediction length: 24
- Fine-tuning: LoRA
- Metrics: WQL / MASE

Main Chronos results:

| Model | WQL | MASE | Takeaway |
|---|---:|---:|---|
| Zero-shot | 1.5441 | 1.6617 | Baseline |
| CE+LoRA | **1.2244** | 1.3806 | Best WQL |
| bin-W1+LoRA | 1.2750 | **1.3151** | Best MASE |
| bin-CRPS+LoRA | 1.2325 | 1.3713 | Best balanced new loss |
| bin-OrdinalCE+LoRA | 1.2357 | 1.3765 | Close to CRPS |

### TimesFM Setup

- Dataset: `tourism_monthly_dataset`
- Split: 257 train / 36 val / 73 test
- Context length: 128
- Prediction length: 24
- Metrics: WQL / MASE

Main TimesFM results:

| Model | WQL | MASE | Takeaway |
|---|---:|---:|---|
| Base zero-shot | **1.1899** | 1.1688 | Best WQL |
| Base LoRA | 1.2058 | 1.1410 | Better MASE, worse WQL |
| Original head + LoRA | 1.2049 | 1.1468 | Strong continuous-head control |
| CE64 + low-LR LoRA | 1.2308 | **1.1162** | Best MASE |
| CE256 + low-LR LoRA | 1.2691 | 1.2336 | Sensitive to head LR |
| CE64 large-pool + LoRA | 1.4641 | 1.3968 | Large pool transfers poorly |

Key conclusions for poster:

- Chronos: original CE remains strongest for WQL, but W1 improves MASE.
- TimesFM: zero-shot remains strongest for WQL, but CE64 + low-LR LoRA improves MASE.
- WQL and MASE do not select the same model.
- Classification and numerical-distance objectives are complementary, not interchangeable.

Suggested visuals:

- Chronos bar chart: WQL / MASE by loss.
- TimesFM table or grouped bars.
- Small WQL-vs-MASE scatter showing metric trade-off.

## Final Takeaway Box

Suggested text:

**There is no universally best output objective for time-series foundation models. Chronos benefits from distance-aware losses for point accuracy, while TimesFM can gain MASE through a classification head. The best adaptation depends on the pretrained architecture and the target metric.**

## Poster Layout Suggestion

If the template is a standard three-column poster:

- Left column: Background, Terminologies, Task.
- Middle column: Challenges, Our Approach with the main method figure.
- Right column: Experiments, Results, Final Takeaway.

Keep most text as bullets. Put the largest visual in "Our Approach" or "Experiments", not in the background section.
