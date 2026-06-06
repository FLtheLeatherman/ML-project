# Reference Notes

This note fixes the citation route used by the Introduction.  It is not a full
literature review; it only records the papers that are necessary to motivate the
project and the role each citation should play.

## Core Forecasting Path

- `box1970time`: Box and Jenkins, *Time Series Analysis: Forecasting and
  Control*.  Use this as the classical ARIMA/statistical forecasting reference.
- `winters1960forecasting`: Winters' exponential smoothing paper.  Use this with
  Box-Jenkins to represent the pre-deep-learning forecasting tradition.

## Transformer and Foundation-Model Path

- `vaswani2017attention`: Transformer origin.  Use this when explaining why
  sequence modeling in forecasting shifted toward attention-based architectures.
- `zhou2021informer`: Early efficient Transformer for long-sequence time-series
  forecasting.
- `wu2021autoformer`: Transformer forecasting with decomposition and
  auto-correlation.
- `nie2023patchtst`: Patch-based Transformer forecasting, relevant because both
  modern forecasting foundation models and TimesFM-style models use patching or
  patch-like sequence compression.
- `das2024timesfm`: TimesFM, arXiv:2310.10688.  This is the required citation for
  the decoder-only forecasting foundation model used in our experiments.
- `ansari2024chronos`: Chronos, arXiv:2403.07815.  This is the required citation
  for the tokenized, language-model-style forecasting foundation model used in
  our experiments.

## Loss and Objective Path

- `huber1964robust`: Huber loss.  Use as the standard robust regression-loss
  reference.
- `koenker1978regression`: Quantile regression / pinball loss origin.  Relevant
  because weighted quantile loss is one of our evaluation metrics and many
  probabilistic forecasters optimize quantile-style objectives.
- `matheson1976scoring`: Continuous ranked probability score (CRPS) and
  continuous distribution scoring.  Use as the classical CRPS reference.
- `villani2008optimal`: Optimal transport / Wasserstein distance background.
  This is the mathematical route behind distance-aware distribution losses.
- `chernov2024wasserstein`: Wasserstein loss for fine-tuning a time-series
  foundation model, arXiv:2409.15367.  This is the required direct comparison
  for modifying Chronos-style classification objectives with distance-aware
  information.
- `wang2025ordinalce`: Ordinal cross-entropy for probabilistic time-series
  forecasting, arXiv:2511.10200.  This is the required citation for replacing
  plain continuous regression with an ordered classification objective.

## How the Introduction Should Use These References

The Introduction should follow this chain:

1. Forecasting is a real planning problem with a long statistical tradition:
   ARIMA and exponential smoothing.
2. Transformers changed sequence modeling, and time-series forecasting adopted
   them through specialized architectures such as Informer, Autoformer, and
   PatchTST.
3. TimesFM and Chronos push this further into pretrained forecasting foundation
   models, but they make opposite output-design choices: TimesFM predicts
   continuous values, while Chronos discretizes values and uses cross-entropy.
4. This creates a central objective-design tension.  Regression and proper
   scoring losses preserve numerical distance, while classification losses are
   stable probabilistic objectives but ignore ordinal distance unless modified.
5. Recent Wasserstein and ordinal-cross-entropy work shows that this tension is
   active.  Our project studies two practical combinations: loss-level
   fine-tuning and architecture/head-level replacement.
