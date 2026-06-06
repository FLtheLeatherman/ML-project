# Chronos 实验说明

## 实验思路

Chronos-T5 本质是分类模型：连续时间序列先经过 mean scaling 和均匀分箱，被转成离散 token；T5 decoder 预测下一个 token 的 logits，默认训练目标是交叉熵。

本实验反过来问一个问题：如果不改变 Chronos 的 token 化和 T5 架构，只改变微调阶段的损失函数，能否让模型在 tourism_monthly 上取得更好的泛化结果。

核心做法：

- 保留 `amazon/chronos-t5-base` 作为基座模型。
- 对 `tourism_monthly_dataset.tsf` 做序列级 holdout：前 80% 训练，后 20% 评估，即 `293 train / 73 eval`。
- 所有微调实验使用 LoRA，统一评估 `WQL / MASE`。
- 比较默认 CE、bin-index MSE、MSE+CE、Wasserstein、CRPS、Ordinal CE、Huber 等损失。
- 数据来自 [Monash Time Series Forecasting Archive](https://forecastingdata.org/)。

## 代码实现

入口文件：

- `scripts/chronos_finetune.py`
- `src/chronos_finetune.py`

关键实现点：

- `parse_tsf()` 解析 Monash `.tsf` 文件。
- `tsf_to_arrow()` 把 `.tsf` 转为 Chronos 训练需要的 feather / Arrow 数据，并生成 train/eval split。
- `write_tsf()` 把 eval split 写回 `.tsf`，用于统一评估。
- `ChronosDataset` 在官方数据流基础上额外输出 `target_values`，供非 CE 损失使用。
- `HighPrecisionTrainer` 提高 HF Trainer 日志精度，避免小量级 loss 被显示成 `0.0000`。
- `BinIndexMSETrainer.compute_loss()` 是所有改造损失的核心实现。
- `load_model_for_finetune()` 加载 `AutoModelForSeq2SeqLM`，并按需挂 LoRA。
- `evaluate_model()` 调用 `ChronosPipeline.predict_quantiles()` 计算 WQL / MASE，并保存 `predictions.npz` 供 notebook 画图。

损失函数实现：

- `ce`：Chronos 原始 token CE，作为默认分类基线。
- `mse`：在 bin-index 空间计算 softmax 期望和目标 token 的 MSE。
- `mse_ce`：`mse_norm + ce_lambda * ce`，其中 `mse_norm = mse / (n_bins - 1)^2`。
- `wass1` / `wass2`：在离散 bin 分布上计算一阶 / 二阶 Wasserstein 风格距离。
- `crps`：离散 CDF 上的 squared loss，更接近 ranked probability score。
- `ordinal_ce`：基于累计分布的 ordinal CE，属于 paper-inspired 兼容实现。
- `huber`：bin-index 空间 Huber loss，当前主表使用 `delta=16`。

## 实验细节与注意事项

- Chronos 的预训练数据包含部分 Monash 数据集；本项目默认用 `tourism_monthly_dataset.tsf`，它对 Chronos 属于 zero-shot 评估集。
- Chronos 的评估使用采样式 `predict_quantiles()`，因此评估前必须重置随机数，否则结果会受实验顺序影响。
- 训练数据 shuffle、ChronosDataset 随机窗口、HF Trainer、NumPy、PyTorch 都需要进入同一个 seed 链路。
- `ChronosDataset` 训练模式会随机 drop 一部分值为 NaN。自定义 loss 中不能依赖 `mask * NaN`，因为结果仍是 NaN；代码已在 target 侧清理 NaN。
- 用 bin center 连续值算 MSE 时 loss 过小，几乎没有有效梯度；当前实现改为 bin-index 空间 MSE。
- bin-index MSE 原始量级远大于 CE，联合损失必须先归一化 MSE。当前主结果使用 `ce_lambda=1e-4`；`5e-4` 是诊断点，CE 占比偏高。
- Chronos 微调优化的是 logits 分布上的损失，但推理仍走 `generate()` 采样；训练目标与推理目标并非完全一致。
- `data/arrow_training/<dataset>/` 会被复用。如果 `.tsf` 改过或怀疑数据不一致，应删除对应 Arrow 子目录后重跑。
- 基座模型权重不在仓库中，`from_pretrained()` 会走 Hugging Face cache 或联网下载。

## 结果

统一口径：

- 模型：`amazon/chronos-t5-base`
- 数据：`tourism_monthly_dataset.tsf`
- 划分：`293 train / 73 eval`
- 预测长度：`24`
- 评估样本数：`73`
- 指标：`WQL / MASE`

| 实验 | 损失 / 设置 | WQL | MASE | 讨论 |
|---|---|---:|---:|---|
| Zero-shot | 原始 Chronos，无微调 | 1.5441 | 1.6617 | 作为基线 |
| `baseline_ce_lora` | CE + LoRA | 1.2244 | 1.3806 | 最好 WQL |
| `exp1a_bin_mse_lora` | bin-MSE + LoRA | 1.2651 | 1.4176 | 可学习，但弱于 CE |
| `exp2_bin_mse_ce_lora` | bin-MSE + CE + LoRA，`ce_lambda=1e-4` | 1.2638 | 1.4147 | 只比纯 MSE 略好 |
| `exp2_bin_mse_ce_lora` | bin-MSE + CE + LoRA，`ce_lambda=5e-4` | 1.2656 | 1.4344 | CE 占比偏高，MASE 变差 |
| `exp4_bin_wass1_lora` | bin-W1 + LoRA | 1.2750 | 1.3151 | 最好 MASE |
| `exp5_bin_wass2_lora` | bin-W2 + LoRA | 1.3252 | 1.3352 | MASE 好，WQL 较弱 |
| `exp7_bin_crps_lora` | bin-CRPS + LoRA | 1.2325 | 1.3713 | 新 loss 中最均衡 |
| `exp8_bin_ordinal_ce_lora` | bin-OrdinalCE + LoRA | 1.2357 | 1.3765 | 接近 CRPS |
| `exp_6a_bin_huber_16_lora` | bin-Huber(16) + LoRA | 1.2562 | 1.3803 | 稳定但不是最优 |

## 讨论

- Chronos 的原始 CE 目标仍然非常强，`CE+LoRA` 在 WQL 上最好。
- 如果更看重点预测 MASE，`bin-W1+LoRA` 是当前最好结果。
- 如果希望概率预测和点预测都比较稳，`bin-CRPS+LoRA` 是当前最值得讨论的新损失。
- `bin-MSE+CE` 的收益不明显，主要原因是 MSE 优化期望值，而 Chronos 推理仍基于自回归采样；二者目标没有完全对齐。
- Huber、W2、OrdinalCE 都说明“分类模型上做分布式距离损失”是可行的，但当前 tourism_monthly 小规模设定下没有稳定超过 CE baseline。

## 复现入口

推荐直接使用 notebook：

- `notebook/REPRODUCE_RESULTS.ipynb`

也可以使用命令行入口：

```bash
python scripts/chronos_finetune.py \
  --model-id amazon/chronos-t5-base \
  --experiments baseline_ce_lora exp1a_bin_mse_lora exp2_bin_mse_ce_lora \
  --max-steps 200 \
  --batch-size 8 \
  --grad-accum 2 \
  --lora-lr 3e-4 \
  --ce-lambda 1e-4 \
  --prediction-length 24 \
  --eval-holdout-ratio 0.2 \
  --eval-n-series 73 \
  --eval-num-samples 20 \
  --seed 42
```
