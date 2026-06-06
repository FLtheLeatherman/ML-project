# Agent Guide

本文件写给后续 agent。普通复现者入口是 [README.md](README.md)，实验细节入口是 [docs/README.md](docs/README.md)。

## 项目目标

机器学习大作业：在时间序列预训练模型上做损失函数改造、微调和对比分析。

- Chronos-T5：从默认 CE 出发，比较 bin-index MSE、MSE+CE、Wasserstein、CRPS、Ordinal CE、Huber 等改造损失。
- TimesFM 2.5：从默认回归头出发，比较原始 head、CE head、staged CE pretrain 和 LoRA 微调。

当前复现结果以 [notebook/REPRODUCE_RESULTS.ipynb](notebook/REPRODUCE_RESULTS.ipynb)、[docs/CHRONOS.md](docs/CHRONOS.md)、[docs/TIMESFM.md](docs/TIMESFM.md) 为准。

## 环境

```bash
conda activate ml-hw1
cd ~/ML/project
```

`ml-hw1` 已配置好项目依赖，包括 CUDA 版 `torch`、`chronos-forecasting`、TimesFM 原生 PyTorch 版、`peft`、`pandas`、`matplotlib`、`jupyter`。

不要在 agent 的普通终端里误启动长时间训练；如果用户要复现大实验，优先给出可运行命令，让用户在 GPU 环境里跑。只有用户明确要求 agent 执行时才启动训练。

## 关键入口

- `notebook/REPRODUCE_RESULTS.ipynb`：当前一键复现入口，会运行训练 / 微调。
- `notebook/EVALUATE_RESULTS.ipynb`：eval-only 入口，只加载基模或已有 checkpoint，不运行 pretrain / fine-tune。
- `notebook/README.md`：notebook 用法、数据、权重和参考结果。
- `docs/CHRONOS.md`：Chronos 实验说明、实现、结果和讨论。
- `docs/TIMESFM.md`：TimesFM 实验说明、实现、结果和讨论。
- `data/DATASETS.md`：当前保留数据清单与用途。

## 代码结构

- `src/chronos_finetune.py`：Chronos 主实现。
- `src/timesfm_baseline.py`：TimesFM 原始回归头、zero-shot、LoRA 主实现。
- `src/timesfm_ce_finetune.py`：TimesFM CE head pretrain / eval / LoRA 主实现。
- `src/timesfm_ce_staged_pretrain.py`：TimesFM staged CE head pretrain 主实现。
- `scripts/*.py`：命令行入口，原则上保持薄封装，核心逻辑放在 `src/`。
- `chronos-forecasting/` 和 `timesfm/`：官方源码参考；除非用户明确要求，不要改 vendor 源码。

## 数据规则

数据来自 [Monash Time Series Forecasting Archive](https://forecastingdata.org/)。实际数据文件不提交到 Git；clone 后需要用户自行下载并解压到 `data/extracted/`。当前本地只保留 17 个 Monash 数据集。不要按旧文档假设存在全量 58 个数据集。

- 下游评估固定使用 `data/extracted/tourism_monthly_dataset.tsf`。
- TimesFM head pretrain / staged pretrain 额外使用 [data/DATASETS.md](data/DATASETS.md) 列出的数据池。
- `monthly_align` 不是当前 notebook 主线复现实验，已从 staged pretrain 当前预设中移除。
- 如果新增或恢复数据集，必须同步更新 `data/DATASETS.md`、`notebook/README.md`，必要时更新 `docs/TIMESFM.md` 或 `docs/CHRONOS.md`。

Chronos 会复用 `data/arrow_training/<dataset>/` 下的 Arrow 文件；如果 `.tsf` 文件变动或怀疑缓存不一致，可以删除对应 Arrow 子目录后重跑。

## 模型权重规则

基座模型权重不在仓库中。

- Chronos 使用 Hugging Face 模型 `amazon/chronos-t5-base`。
- TimesFM 使用 Hugging Face 模型 `google/timesfm-2.5-200m-pytorch`。
- `timesfm/` 是源码目录，不是权重目录。
- 代码应优先使用 Hugging Face cache 和相对项目路径，不要写死用户机器上的绝对 checkpoint 路径。

## 输出与清理

- `output/` 是本地 checkpoint、metrics、图和复现结果，可以保留在本地，不作为主要交付。
- 不要主动删除 `output/`，除非用户明确要求。
- `notebook/EVALUATE_RESULTS.ipynb` 依赖本地已有 `output/*_repro_*` checkpoint；新 clone 的仓库不能直接跑 eval notebook，必须先跑复现或手动准备 checkpoint。
- 可以清理 `__pycache__/`、`.ipynb_checkpoints/` 和 notebook 临时输出。
- 修改 notebook 后应清空执行输出，保持文件轻量。

## 文档同步

如果修改实验命令、默认超参数、数据池、结果表或 checkpoint 依赖关系，需要同步检查：

- `README.md`
- `AGENTS.md`
- `notebook/README.md`
- `data/DATASETS.md`
- `docs/CHRONOS.md`
- `docs/TIMESFM.md`
- `notebook/REPRODUCE_RESULTS.ipynb`
- `notebook/EVALUATE_RESULTS.ipynb`

不要把长篇实验分析复制到 `README.md` 或 `AGENTS.md`；详细讨论放在 `docs/`。

## 常见技术细节

- Chronos 的自定义 MSE 路线使用 bin-index 空间，避免 center-value MSE 量级过小。
- Chronos 自定义 loss 要处理 `target_values` 中的 NaN；`NaN * 0` 仍是 NaN。
- Chronos 联合损失需要先对齐 MSE 与 CE 量级，再谈 lambda。
- TimesFM 2.5 不能用 `transformers` 里的不兼容架构直接加载；本项目使用 TimesFM 官方原生 PyTorch 模块。
- TimesFM CE head 的 downstream LoRA 学习率要低，尤其 CE256；高学习率容易过拟合。
- TimesFM 的 direct / LoRA 实验依赖当前 notebook 输出目录里的 pretrain checkpoint，不应静默回退旧 output。

## 最小校验

整理代码或文档后，优先做这些轻量检查：

```bash
python -m compileall -q src scripts
find src scripts -type d -name __pycache__ -prune -exec rm -rf {} +
```

如果改了数据池，再检查 `data/extracted/*.tsf`、`data/*.zip` 和 staged preset 是否一致。
