# Loss-Function Fine-Tuning for Time-Series Foundation Models

本项目是机器学习大作业：在时间序列预训练模型上改造损失函数，并比较 zero-shot、原始微调和改造损失微调的效果。

主要模型：

- Chronos-T5：分类式时间序列模型，默认交叉熵；本项目尝试 bin-index MSE、CRPS、Wasserstein、Ordinal CE 等损失。
- TimesFM 2.5：回归式时间序列模型，默认回归头；本项目尝试预训练 / 微调 CE head，并与原始回归头比较。

## 快速开始

```bash
conda activate ml-hw1
cd ~/ML/project
jupyter notebook notebook/REPRODUCE_RESULTS.ipynb
```

环境需要包含 `torch` CUDA 版、`chronos-forecasting`、`timesfm` 原生 PyTorch 版、`peft`、`pandas`、`matplotlib`、`jupyter` 等依赖。训练建议在 GPU 环境中运行；CPU 会非常慢。

## 复现入口

推荐使用 [notebook/REPRODUCE_RESULTS.ipynb](notebook/REPRODUCE_RESULTS.ipynb) 复现当前结果。先运行第一个 setup cell，再按需要运行后续实验 cell。

Notebook 会把新结果写到 `output/notebook_repro_时间戳/`。TimesFM 的 direct / LoRA 实验依赖同一次 notebook 运行中先生成的 pretrain checkpoint，不会自动回退旧的 `output/` 目录。

如果只想加载已有 checkpoint 重新评估并展示预测图，使用 [notebook/EVALUATE_RESULTS.ipynb](notebook/EVALUATE_RESULTS.ipynb)。它不运行 pretrain 或 fine-tune，只写入 `output/notebook_eval_时间戳/`。

更详细的 notebook 用法见 [notebook/README.md](notebook/README.md)。

## 数据与模型权重

数据来自 [Monash Time Series Forecasting Archive](https://forecastingdata.org/)。当前只保留复现实验需要的 17 个 Monash `.tsf` 数据集，详见 [data/DATASETS.md](data/DATASETS.md)。下游评估固定使用 `tourism_monthly_dataset.tsf`。

Chronos 和 TimesFM 的基座模型权重不在仓库中：

- Chronos：`amazon/chronos-t5-base`
- TimesFM：`google/timesfm-2.5-200m-pytorch`

脚本会从 Hugging Face cache 读取权重；离线复现前需要提前准备好缓存。`timesfm/` 目录只是 TimesFM 官方源码参考，不包含 `model.safetensors`。

## 文档入口

- [docs/README.md](docs/README.md)：文档索引和通用注意事项。
- [docs/CHRONOS.md](docs/CHRONOS.md)：Chronos 实验思路、实现细节、结果和讨论。
- [docs/TIMESFM.md](docs/TIMESFM.md)：TimesFM 实验思路、实现细节、结果和讨论。

当前主要结论：

- Chronos：`CE+LoRA` 的 WQL 最好，`bin-W1+LoRA` 的 MASE 最好。
- TimesFM：原始基模 zero-shot 的 WQL 最好；`legacy64 CE head + low-LR LoRA` 的 MASE 最好，但 WQL 不如原始基模。
- TimesFM 的 CE head direct 迁移整体不稳定，LoRA 学习率过大时容易过拟合。

完整结果表以 `docs/CHRONOS.md` 和 `docs/TIMESFM.md` 为准。

## 目录结构

```text
ML/project/
├── chronos-forecasting/         # Chronos 官方源码参考
├── timesfm/                     # TimesFM 官方源码参考
├── data/                        # 当前保留数据与 DATASETS.md
├── docs/                        # Chronos / TimesFM 实验说明
├── notebook/                    # 一键复现 notebook
├── scripts/                     # 命令行入口，调用 src/
├── src/                         # 实验实现代码
├── output/                      # 本地 checkpoint / metrics / 图，不作为仓库交付
├── papers/reference/            # 参考论文
├── README.md
└── AGENTS.md
```

## 本地产物

`output/`、Hugging Face cache、`__pycache__/`、`.ipynb_checkpoints/` 都是本地产物。`output/` 可以保留在本地用于对照，但不应作为主要交付内容；复现与说明以 notebook 和 docs 为准。
