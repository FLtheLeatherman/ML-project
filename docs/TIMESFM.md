# TimesFM 实验说明

## 实验思路

TimesFM 2.5 是回归模型：输入连续时间序列 patch，经 decoder-only transformer 输出 point / quantile forecast。它没有离散 token、词表或 softmax 分类头。

本实验从相反方向改造 TimesFM：在保留 TimesFM 主干的前提下，比较原始回归微调、重新训练原始 forecast head、以及新增 CE 分类头三条路线。

核心问题：

- 原始 TimesFM zero-shot 是否已经足够强。
- 原始回归 loss + LoRA 是否能提升 tourism_monthly。
- 把 TimesFM 改造成“Chronos 式”分箱分类模型后，CE head 是否有价值。
- CE head 预训练、bin 数、LoRA 学习率对最终 WQL / MASE 的影响。

统一 tourism_monthly 评估口径：

- 数据：`tourism_monthly_dataset.tsf`
- 来源：[Monash Time Series Forecasting Archive](https://forecastingdata.org/)
- 划分：`257 train / 36 val / 73 test`
- `context_length=128`
- `prediction_length=24`
- 指标：`WQL / MASE`

## 代码实现

入口文件：

- `scripts/timesfm_baseline.py`
- `scripts/timesfm_ce_finetune.py`
- `scripts/timesfm_ce_staged_pretrain.py`

核心实现文件：

- `src/timesfm_baseline.py`
- `src/timesfm_ce_finetune.py`
- `src/timesfm_ce_staged_pretrain.py`

### 原始回归头 / LoRA

`src/timesfm_baseline.py` 实现 TimesFM 原始回归路线：

- `load_model()` 加载 `google/timesfm-2.5-200m-pytorch` 的原生 PyTorch 权重。
- `get_hf_cache_roots()` 显式扫描 Hugging Face cache，支持 `HUGGINGFACE_HUB_CACHE`、`HF_HUB_CACHE`、`HF_HOME/hub`、`TRANSFORMERS_CACHE` 和默认 cache。
- `RandomWindowDataset` 用随机窗口训练，`LastWindowDataset` 用最后窗口评估。
- 训练时手动复刻 TimesFM decode 的 patching 和 RevIN 流程，因为官方 `decode()` 推理路径含 `torch.no_grad()`，不能直接训练。
- `run_finetune()` 支持两种模式：`mode=lora` 微调主干 LoRA；`mode=head` 冻结 backbone，只训练原始 point/quantile forecast head。
- `evaluate_model()` 输出 WQL / MASE，并保存 `predictions.npz`。

### CE head

`src/timesfm_ce_finetune.py` 实现分类头路线：

- `CEHead`：`1280 -> 1280 -> 128*K`，并带 residual projection，输出 `(B, P, 128, K)`。
- `build_bins()` 在 RevIN 归一化空间构造均匀 bins。
- `bin_targets()` 把归一化后的连续目标值映射到 bin id。
- `ce_training_step()` 手动 patch context，计算 RevIN running stats，调用 TimesFM backbone，接 CE head 后用 `F.cross_entropy` 训练。
- `ce_inference()` 对 CE logits 做 softmax，用期望值作为点预测，用 CDF 反演导出分位数。
- `evaluate_model_ce()` 统一计算 WQL / MASE，并保存 `predictions.npz`。
- `run_finetune_ce()` 支持 frozen head pretrain 和 LoRA downstream fine-tune。

CE head 参数量：

| bins | 参数量 |
|---:|---:|
| 64 | 22.6M |
| 256 | 85.6M |

### staged pretrain

`src/timesfm_ce_staged_pretrain.py` 用于分阶段训练 CE head：

- `starter` pool：较小、干净的数据池，用于快速验证。
- `large` pool：更大的 generic pool。
- large stage 若没有初始 checkpoint，会先自动 bootstrap starter，再继续 large。
- 每个 stage 会记录数据统计、split、配置和 checkpoint。

当前复现 notebook 中 `timesfm_new64_pretrain` 使用 `large` stage；如果没有传入初始 checkpoint，代码会先自动跑 `starter` bootstrap。因此数据目录需要保留：

- `starter`：`m4_monthly_dataset`、`temperature_rain_dataset_without_missing_values`、`weather_dataset`、`traffic_hourly_dataset`、`electricity_hourly_dataset`
- `large`：`kaggle_web_traffic_dataset_without_missing_values`、`temperature_rain_dataset_without_missing_values`、`m4_monthly_dataset`、`m4_daily_dataset`、`weather_dataset`、`rideshare_dataset_without_missing_values`、`traffic_hourly_dataset`、`m4_hourly_dataset`、`electricity_hourly_dataset`、`vehicle_trips_dataset_without_missing_values`、`kdd_cup_2018_dataset_without_missing_values`、`covid_deaths_dataset`、`nn5_daily_dataset_without_missing_values`、`fred_md_dataset`
- `monthly_align` 不是当前 notebook 的主线复现实验，已从 staged pretrain 当前预设中移除。

## 实验细节与注意事项

- `transformers` 里的 TimesFM 架构与 `google/timesfm-2.5-200m-pytorch` checkpoint 不兼容；本项目使用 TimesFM 官方源码里的原生 PyTorch 模块手动加载 `model.safetensors`。
- TimesFM 基模权重不在仓库里，必须来自 Hugging Face cache 或联网下载。
- `timesfm/` 目录是源码，不是模型权重。
- CE bins 定义在 RevIN 归一化空间；标签分箱必须使用与模型 forward 相同的归一化统计。
- CE 推理要从分类分布还原连续值：点预测使用 softmax 期望，分位数使用 CDF 反演。
- CE head 容量很大，尤其 256-bin 有 85.6M 参数；预训练数据不足时容易迁移不稳。
- downstream LoRA 阶段不能沿用 pretrain 的大 head LR。当前经验：pretrain head 用 `1e-3`；downstream CE64 用 `head_lr=5e-5`、`lora_lr=5e-5`；CE256 推荐 `head_lr=1e-5`、`lora_lr=5e-5`。
- 旧高学习率 CE LoRA 会明显过拟合，尤其 CE256。
- TimesFM 的 tourism_monthly split 是 `257/36/73`，不要和 Chronos 的 `293/73` 直接做严格横向比较。
- notebook 中 TimesFM direct / LoRA 实验依赖当前 `REPRO_ROOT` 下先跑出的 pretrain checkpoint；缺 checkpoint 会直接报错，不回退旧 output。

## 结果

统一 tourism_monthly 结果：

| 实验 | 设置 | WQL | MASE | 讨论 |
|---|---|---:|---:|---|
| Base zero-shot | 原始 TimesFM 2.5 | 1.1899 | 1.1688 | 最好 WQL |
| Base LoRA | 原始回归 loss + LoRA | 1.2058 | 1.1410 | MASE 改善，WQL 变差 |
| Original head direct | 原始 forecast head 从 0 预训练后直接迁移 | 1.2795 | 1.2389 | 比 CE direct 稳 |
| Original head LoRA | original head 初始化 + tourism LoRA | 1.2049 | 1.1468 | 接近 Base LoRA |
| CE64 direct | 64-bin CE head 直接迁移 | 1.4605 | 1.3809 | direct 较弱 |
| CE64 LoRA low LR | `head_lr=5e-5`，`lora_lr=5e-5` | 1.2308 | 1.1162 | 最好 MASE，当前 CE 主结果 |
| CE64 LoRA high LR | `head_lr=1e-3`，`lora_lr=1e-4` | 1.3459 | 1.2697 | 明显过拟合 |
| CE256 direct | 256-bin CE head 直接迁移 | 1.3624 | 1.2862 | direct 好于 CE64 |
| CE256 LoRA `head_lr=1e-5` | `lora_lr=5e-5` | 1.2691 | 1.2336 | 推荐 256-bin LoRA 口径 |
| CE256 LoRA `head_lr=5e-5` | `lora_lr=5e-5` | 1.3094 | 1.2605 | 不如 1e-5 |
| CE256 LoRA freeze head | `head_lr=0`，`lora_lr=5e-5` | 1.3183 | 1.3057 | 冻结 head 不如小 LR |
| CE256 LoRA high LR | `head_lr=1e-3`，`lora_lr=1e-4` | 1.3892 | 1.4606 | 严重过拟合 |
| new64 large direct | staged large 64-bin CE head | 4.5760 | 5.5845 | 直接迁移失败 |
| new64 large LoRA | low LR LoRA | 1.4641 | 1.3968 | LoRA 修复明显，但仍不推荐 |

补充：预训练池内部 test split 不是 tourism_monthly，只用于看预训练分布内表现。

| 实验 | 内部分布 WQL | 内部分布 MASE |
|---|---:|---:|
| legacy64 CE head pretrain | 1.2672 | 1.4435 |
| legacy64 original head pretrain | 0.9926 | 1.2253 |
| CE256 head pretrain | 0.6285 | 0.7708 |

## 讨论

- TimesFM zero-shot 本身很强，当前最好 WQL 仍是原始基模。
- 原始回归 LoRA 能改善 MASE，但会让 WQL 变差，说明分位数预测校准可能被破坏。
- CE64 + low LR LoRA 取得最好 MASE，但 WQL 不如 zero-shot / original-head LoRA。
- CE256 direct 迁移强于 CE64 direct，但 downstream LoRA 对学习率更敏感，`head_lr=1e-5` 明显更稳。
- original head from scratch 的表现说明“重新训练输出头”是有效对照；它比 CE direct 稳，但在 MASE 上不如 CE64 LoRA。
- new64 large 的 direct 结果很差，说明更大的 generic pretrain pool 不一定带来更好的目标域迁移；数据池和训练阶段设计仍然关键。
- CE 路线的主要价值不是全面超过原始 TimesFM，而是验证“回归架构 + 分类损失 + 分箱概率预测”这条路线在 MASE 上能取得有效改进。

## 复现入口

推荐直接使用 notebook：

- `notebook/REPRODUCE_RESULTS.ipynb`

命令行入口：

```bash
# 原始 TimesFM zero-shot / LoRA / original head
python scripts/timesfm_baseline.py --help

# CE head direct / LoRA
python scripts/timesfm_ce_finetune.py --help

# staged CE pretrain
python scripts/timesfm_ce_staged_pretrain.py --help
```
