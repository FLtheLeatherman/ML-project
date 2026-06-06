"""
Chronos-T5 微调脚本 (bin-index MSE)
==================================

基于 docs/CHRONOS.md 的实施:
- 路线 A: 自定义 HF Trainer 覆写 compute_loss
- §六-A 修复: 在 **bin 索引空间** 计算 MSE, 避免 center-value MSE ≈ 0 的问题
- 支持 loss_type = "ce" | "mse" | "mse_ce"
- 默认实验全部使用 LoRA (peft) 微调, 与 TimesFM 侧对齐
- 评估: WQL + MASE

运行 (在 ml-hw1 环境中):
    cd ~/ML/project
    python scripts/chronos_finetune.py --help

也可以在 Python 中 `from src import chronos_finetune` 复用函数。

依赖:
    conda activate ml-hw1
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info

import transformers
from peft import LoraConfig, TaskType, get_peft_model
from tqdm.auto import tqdm

from chronos import ChronosConfig, ChronosPipeline
from chronos.chronos import ChronosModel
from gluonts.dataset.common import FileDataset
from gluonts.itertools import Cyclic, Filter, Map
from gluonts.transform import (
    ExpectedNumInstanceSampler,
    FilterTransformation,
    InstanceSplitter,
    LeavesMissingValues,
    TestSplitSampler,
    ValidationSplitSampler,
)
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    T5Config,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("chronos_finetune")


# =============================================================================
# 全局配置
# =============================================================================

# 频率 → pandas freq 映射 (与 notebook/chronos/02 保持一致)
FREQ_MAP = {
    "yearly": "Y",
    "quarterly": "Q",
    "monthly": "M",
    "weekly": "W",
    "daily": "D",
    "hourly": "H",
    "half_hourly": "30min",
    "10_minutes": "10min",
    "minutely": "min",
    "4_seconds": "4s",
}

# 频率 → 季节性周期 (用于 MASE 的季节性 naive 基准)
SEASONALITY = {
    "Y": 1,
    "Q": 4,
    "M": 12,
    "W": 1,
    "D": 7,
    "H": 24,
    "30min": 48,
    "10min": 144,
    "min": 60,
}

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# New seeded reproduction default. Previous base command used 5e-4, which made
# the joint loss strongly CE-dominated after the deterministic data-order fix.
DEFAULT_CE_LAMBDA = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# 0. 随机种子
# =============================================================================


def set_all_seeds(seed: int) -> None:
    """统一设置 Python / NumPy / Torch 随机种子。

    Chronos 的评估会走 sampling-based `predict_quantiles()`，如果不在每次评估
    前显式重置随机状态，结果会受实验顺序影响。
    """
    transformers.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# =============================================================================
# 1. 数据加载 / TSF → Arrow
# =============================================================================


def parse_tsf(file_path: str) -> Tuple[List[np.ndarray], Optional[str]]:
    """解析 .tsf 文件, 返回 (series_list, frequency_str)。"""
    series_list: List[np.ndarray] = []
    frequency: Optional[str] = None
    reading_data = False
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("@frequency"):
                frequency = line.split(" ", 1)[1].strip()
                continue
            if line.startswith("@data"):
                reading_data = True
                continue
            if not reading_data:
                continue
            values_str = line.split(":")[-1].strip()
            sep = "," if "," in values_str else " "
            arr: List[float] = []
            for v in values_str.split(sep):
                v = v.strip()
                if v in ("?", ""):
                    arr.append(float("nan"))
                else:
                    try:
                        arr.append(float(v))
                    except ValueError:
                        arr.append(float("nan"))
            if arr:
                series_list.append(np.asarray(arr, dtype=np.float32))
    return series_list, frequency


def tsf_to_arrow(
    tsf_path: str,
    arrow_dir: str,
    max_series: Optional[int] = None,
    eval_holdout_ratio: float = 0.2,
) -> Tuple[str, str, str]:
    """解析 TSF 并写 feather 文件, 支持 train/eval 划分。

    Parameters
    ----------
    tsf_path : str
        源 TSF 文件路径
    arrow_dir : str
        输出目录
    max_series : int, optional
        最多取多少条序列 (None 表示全部)
    eval_holdout_ratio : float
        评估集占比, 默认 0.2 (前 80% 训练, 后 20% 评估)。
        设为 0 则不划分, 所有序列同时用于训练和评估 (旧行为)。

    Returns
    -------
    (arrow_dir, train_feather_path, eval_feather_path, pd_freq)
        如果不划分, train 和 eval 指向同一文件。
    """
    series_list, freq_str = parse_tsf(tsf_path)
    pd_freq = FREQ_MAP.get(freq_str, "D")
    if max_series and len(series_list) > max_series:
        series_list = series_list[:max_series]

    arrow_dir_path = Path(arrow_dir)
    arrow_dir_path.mkdir(parents=True, exist_ok=True)

    # train/eval 划分
    if eval_holdout_ratio > 0:
        n_eval = max(1, int(len(series_list) * eval_holdout_ratio))
        n_train = len(series_list) - n_eval
        train_series = series_list[:n_train]
        eval_series = series_list[n_train:]
        log.info(
            f"Split: {n_train} train / {n_eval} eval "
            f"(holdout={eval_holdout_ratio:.0%})"
        )
    else:
        train_series = series_list
        eval_series = series_list
        n_train = len(series_list)
        n_eval = len(series_list)

    default_start = str(pd.Period("1970-01", freq=pd_freq))

    def _write(subset, filename):
        table = pa.table({
            "start": pa.array([default_start] * len(subset)),
            "target": pa.array([s.tolist() for s in subset],
                              type=pa.list_(pa.float32())),
        })
        feather_path = arrow_dir_path / filename
        feather.write_feather(table, str(feather_path))
        return str(feather_path)

    train_path = _write(train_series, "data_train.feather")
    if eval_holdout_ratio > 0:
        eval_path = _write(eval_series, "data_eval.feather")
    else:
        eval_path = train_path

    log.info(
        f"TSF→Arrow: {len(series_list)} series, freq={freq_str}→{pd_freq}, "
        f"train={n_train}, eval={n_eval}"
    )
    return str(arrow_dir_path), train_path, eval_path, pd_freq


def write_tsf(series_list: List[np.ndarray], freq_str: str, out_path: str) -> str:
    """把 series_list 写回 .tsf 格式。用于把 eval 子集导出为独立 TSF。"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"@relation EvalSplit\n")
        f.write(f"@attribute series_name numeric\n")
        f.write(f"@attribute start_timestamp date\n")
        f.write(f"@frequency {freq_str}\n")
        f.write(f"@missing false\n")
        f.write(f"@equallength false\n")
        f.write(f"@data\n")
        for i, s in enumerate(series_list):
            values = ",".join(
                "?" if np.isnan(v) else repr(float(v)) for v in s
            )
            f.write(f"series_{i}:1970-01-01 00-00-00:{values}\n")
    return out_path


# =============================================================================
# 2. ChronosDataset (输出 target_values 用于 MSE)
# =============================================================================


class PseudoShuffledIterableDataset(IterableDataset):
    """带 shuffle buffer 的 IterableDataset (复刻 train.py)。"""

    def __init__(
        self,
        base_dataset,
        shuffle_buffer_length: int = 100,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.shuffle_buffer_length = shuffle_buffer_length
        self.seed = seed

    def __iter__(self):
        generator = torch.Generator()
        if self.seed is None:
            generator.manual_seed(torch.initial_seed())
        else:
            wi = get_worker_info()
            worker_id = 0 if wi is None else wi.id
            generator.manual_seed(int(self.seed) + worker_id)

        buf = []
        for elem in self.base_dataset:
            buf.append(elem)
            if len(buf) >= self.shuffle_buffer_length:
                idx = torch.randint(len(buf), size=(), generator=generator)
                yield buf.pop(idx)
        while buf:
            idx = torch.randint(len(buf), size=(), generator=generator)
            yield buf.pop(idx)


class ShuffleMixin:
    def shuffle(self, buffer_length: int = 100, seed: Optional[int] = None):
        return PseudoShuffledIterableDataset(self, buffer_length, seed=seed)


class ChronosDataset(IterableDataset, ShuffleMixin):
    """Chronos 训练数据集, 与官方 ChronosDataset 一致但额外输出 target_values。"""

    def __init__(
        self,
        datasets,
        probabilities,
        tokenizer,
        context_length: int = 512,
        prediction_length: int = 64,
        drop_prob: float = 0.2,
        min_past: Optional[int] = None,
        model_type: str = "seq2seq",
        mode: str = "training",
        np_dtype=np.float32,
        output_target_values: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__()
        assert len(probabilities) == len(datasets)
        assert mode in ("training", "validation", "test")
        assert model_type in ("seq2seq", "causal")
        self.datasets = datasets
        self.probabilities = probabilities
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.drop_prob = drop_prob if model_type == "seq2seq" else 0.0
        self.min_past = min_past or prediction_length
        self.model_type = model_type
        self.mode = mode
        self.np_dtype = np_dtype
        self.output_target_values = output_target_values
        self.seed = seed

    def preprocess_entry(self, entry, mode):
        entry = {f: entry[f] for f in ["start", "target"]}
        entry["target"] = np.asarray(entry["target"], dtype=self.np_dtype)
        if mode == "training" and self.drop_prob > 0:
            t = entry["target"].copy()
            dp = np.random.uniform(0, self.drop_prob)
            mask = np.random.choice([True, False], size=len(t), p=[dp, 1 - dp])
            t[mask] = np.nan
            entry["target"] = t
        return entry

    def _create_instance_splitter(self, mode):
        sampler = {
            "training": ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_instances=1,
                min_past=self.min_past,
                min_future=self.prediction_length,
            ),
            "test": TestSplitSampler(),
            "validation": ValidationSplitSampler(min_future=self.prediction_length),
        }[mode]
        return InstanceSplitter(
            target_field="target",
            is_pad_field="is_pad",
            start_field="start",
            forecast_start_field="forecast_start",
            instance_sampler=sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            dummy_value=np.nan,
        )

    def create_training_data(self, data):
        data = Cyclic(data)
        t = self._create_instance_splitter("training") + FilterTransformation(
            condition=lambda e: (~np.isnan(e["past_target"])).sum() > 0
        )
        return t.apply(data, is_train=True)

    def create_validation_data(self, data):
        return self._create_instance_splitter("validation").apply(data, is_train=False)

    def create_test_data(self, data):
        return self._create_instance_splitter("test").apply(data, is_train=False)

    def to_hf_format(self, entry):
        past_target = torch.tensor(entry["past_target"]).unsqueeze(0)
        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(past_target)
        future_target = torch.tensor(entry["future_target"]).unsqueeze(0)
        labels, labels_mask = self.tokenizer.label_input_transform(future_target, scale)
        labels[labels_mask == 0] = -100

        result = {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }
        if self.output_target_values:
            # target_values 在缩放空间, 长度为 prediction_length (不含 EOS)
            tv = (future_target / scale.unsqueeze(-1)).squeeze(0)
            result["target_values"] = tv
        return result

    def __iter__(self):
        wi = get_worker_info()
        worker_id = 0 if wi is None else wi.id
        if self.seed is not None:
            worker_seed = int(self.seed) + worker_id
            np.random.seed(worker_seed % (2**32))
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        pds = [
            Map(partial(self.preprocess_entry, mode=self.mode), d)
            for d in self.datasets
        ]
        if self.mode == "training":
            iters = [self.create_training_data(d) for d in pds]
        elif self.mode == "test":
            iters = [self.create_test_data(d) for d in pds]
        else:
            iters = [self.create_validation_data(d) for d in pds]

        wi = get_worker_info()
        if wi is None:
            probs = list(self.probabilities)
        else:
            iters = list(itertools.islice(iters, wi.id, None, wi.num_workers))
            probs = list(itertools.islice(self.probabilities, wi.id, None, wi.num_workers))
        probs = [p / sum(probs) for p in probs]
        iterators = list(map(iter, iters))

        if self.mode == "training":
            while True:
                idx = np.random.choice(range(len(iterators)), p=probs)
                try:
                    yield self.to_hf_format(next(iterators[idx]))
                except StopIteration:
                    probs[idx] = 0
                    if sum(probs) == 0:
                        return
                    probs = [p / sum(probs) for p in probs]
        else:
            for entry in itertools.chain(*iterators):
                yield self.to_hf_format(entry)


# =============================================================================
# 3. BinIndexMSETrainer (核心修复)
# =============================================================================


class HighPrecisionTrainer(Trainer):
    """HF Trainer 默认把 loss round 到 4 位小数 (见 _maybe_log_save_evaluate),
    对 MSE_norm 这种量级 ~0.001 的 loss 看起来像 "都是 00"。
    这里覆写为 8 位小数, 让训练日志显示完整精度。

    只改日志显示, 不影响内部训练计算 (反向传播始终用完整精度)。
    """

    LOG_ROUND_DIGITS = 8

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch,
        ignore_keys_for_eval, start_time, learning_rate=None,
    ):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs: Dict[str, float] = {}
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            tr_loss -= tr_loss
            logs["loss"] = round(
                tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged),
                self.LOG_ROUND_DIGITS,
            )
            if grad_norm is not None:
                logs["grad_norm"] = (
                    grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                )
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs, start_time)

        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best = self._determine_best_metric(metrics, trial)
            if self.args.save_strategy == "best":
                self.control.should_save = is_new_best

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )


class BinIndexMSETrainer(HighPrecisionTrainer):
    """用 bin-index 空间计算距离型损失, 解决 center-value MSE ≈ 0 的问题。

    loss_type:
      - "ce": 默认交叉熵 (回退到 HF Trainer 行为)
      - "mse": 纯 bin-index MSE
      - "mse_ce": MSE + λ·CE 联合损失
      - "wass1": 纯 1-Wasserstein (目标视作 bin-index 空间的点质量)
      - "wass2": 纯 2-Wasserstein (目标视作 bin-index 空间的点质量)
      - "huber": 纯 bin-index Huber / SmoothL1
      - "crps": 离散 bin 上的 CRPS / Ranked Probability Score
      - "ordinal_ce": 累计分布版 ordinal CE
    """

    def __init__(
        self,
        *args,
        n_tokens: int = 4096,
        n_special_tokens: int = 2,
        low_limit: float = -15.0,
        high_limit: float = 15.0,
        loss_type: str = "mse",
        ce_lambda: float = DEFAULT_CE_LAMBDA,
        huber_delta_bins: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert loss_type in (
            "mse",
            "ce",
            "mse_ce",
            "wass1",
            "wass2",
            "huber",
            "crps",
            "ordinal_ce",
        ), (
            f"Unknown loss_type: {loss_type}"
        )
        self._loss_type = loss_type
        self._ce_lambda = ce_lambda
        self._huber_delta_bins = float(huber_delta_bins)
        self._n_special_tokens = n_special_tokens
        self._n_bins = n_tokens - n_special_tokens - 1  # 4093 for default
        self._n_numeric_tokens = n_tokens - n_special_tokens
        self._low_limit = float(low_limit)
        self._high_limit = float(high_limit)

        # v2idx: token_id → bin index (float).
        # PAD (0) / EOS (1) 映射到 0, 但会被 softmax 的 -inf mask 排除;
        # 数值 token_id t (t ≥ 2) 映射到 bin = t - 2
        v2idx = torch.arange(n_tokens, dtype=torch.float32)
        v2idx[:n_special_tokens] = 0.0
        v2idx[n_special_tokens:] = v2idx[n_special_tokens:] - n_special_tokens
        self._v2idx = v2idx

    def _build_soft_target_probs(
        self, target_idx: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """把连续 target_idx 投到相邻 numeric bins 上做线性插值。"""
        floor_idx = torch.floor(target_idx).long().clamp(
            min=0, max=self._n_numeric_tokens - 1
        )
        ceil_idx = (floor_idx + 1).clamp(max=self._n_numeric_tokens - 1)
        frac = (target_idx - floor_idx.to(target_idx.dtype)).clamp(0.0, 1.0)

        target_probs = torch.zeros(
            *target_idx.shape,
            self._n_numeric_tokens,
            device=device,
            dtype=target_idx.dtype,
        )
        target_probs.scatter_add_(-1, floor_idx.unsqueeze(-1), (1.0 - frac).unsqueeze(-1))
        target_probs.scatter_add_(-1, ceil_idx.unsqueeze(-1), frac.unsqueeze(-1))
        return target_probs

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        target_values = inputs.pop("target_values", None)

        if self._loss_type == "ce" or target_values is None:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs
            )

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
        )
        logits = outputs.logits  # [B, L, V], L = prediction_length + 1 (EOS)

        # --- 预测: bin-index 空间的 softmax 期望 ---
        logits_masked = logits.clone()
        logits_masked[..., : self._n_special_tokens] = float("-inf")
        probs = F.softmax(logits_masked, dim=-1)
        # 把 v2idx 搬到 logits 的 device (首次会触发, 后续 cache 住)
        v2idx = self._v2idx.to(probs.device)
        pred_idx = (probs * v2idx).sum(dim=-1)  # [B, L]

        # --- 真值: target_values (缩放空间的连续值) → bin-index ---
        # idx = (v - low) / (high - low) * (n_bins - 1)
        # 注意: target_values 在训练模式 (drop_prob=0.2) 下含 NaN, 这里先清成 0,
        #       后续再用 mask 排除, 避免 NaN 参与 bin 索引计算。
        tv_clean = torch.where(
            torch.isnan(target_values),
            torch.zeros_like(target_values),
            target_values,
        )
        span = self._high_limit - self._low_limit
        target_idx = (
            (tv_clean - self._low_limit) / span * (self._n_bins - 1)
        )
        target_idx = target_idx.clamp(0.0, float(self._n_bins - 1))

        # --- 对齐: 预测序列比 target_idx 多 1 个 EOS 位置, 截断 ---
        pred_valid = pred_idx[..., : target_values.shape[-1]]
        probs_valid = probs[..., : target_values.shape[-1], :]

        # mask: 排除 labels==-100 (被 tokenizer 标记为 NaN / padding / drop 的位置)
        mask = (labels[..., : target_values.shape[-1]] != -100).float()
        denom = mask.sum().clamp(min=1.0)

        # target_idx 已在源头 (tv_clean) 清掉 NaN, 此处 mask 保证不参与 loss 的 NaN 位置被乘 0 排除
        mse = ((pred_valid - target_idx) ** 2 * mask).sum() / denom

        # 关键修复 (归一化): bin-index MSE 的量级是 (n_bins-1)² ≈ 1.67e7,
        # 而 CE 的量级只有 ~5-8, 直接相加 CE 会被完全淹没。
        # 归一化 MSE 到 [0, 1]: mse_norm = mse / (n_bins - 1)²
        max_mse = (self._n_bins - 1) ** 2
        mse_norm = mse / max_mse

        numeric_probs = probs_valid[..., self._n_special_tokens :]

        # Wasserstein: 目标分布视为落在连续 target_idx 上的点质量。
        # 对一维 ordered bins, W_p(P, δ_t) = (E |X-t|^p)^(1/p)。
        distance = (v2idx.view(1, 1, -1) - target_idx.unsqueeze(-1)).abs()
        wass1_per_token = (probs_valid * distance).sum(dim=-1)
        wass1_norm = (wass1_per_token * mask).sum() / (denom * (self._n_bins - 1))

        wass2_second_moment = (probs_valid * distance.square()).sum(dim=-1)
        wass2_per_token = torch.sqrt(wass2_second_moment + 1e-12)
        wass2_norm = (wass2_per_token * mask).sum() / (denom * (self._n_bins - 1))

        huber_per_token = F.huber_loss(
            pred_valid,
            target_idx,
            reduction="none",
            delta=self._huber_delta_bins,
        )
        huber_scale = max(
            self._huber_delta_bins
            * max(self._n_bins - 1 - 0.5 * self._huber_delta_bins, 1.0),
            1.0,
        )
        huber_norm = (huber_per_token * mask).sum() / (denom * huber_scale)

        target_probs = self._build_soft_target_probs(target_idx, numeric_probs.device)
        pred_cdf = torch.cumsum(numeric_probs, dim=-1)
        target_cdf = torch.cumsum(target_probs, dim=-1)

        # 离散 bins 上用 cumulative distribution 差值近似 CRPS / RPS。
        crps_per_token = (pred_cdf - target_cdf).square().mean(dim=-1)
        crps = (crps_per_token * mask).sum() / denom

        # 参考 2511.10200 的累计分布思路，对预测 / 真值 CDF 做 soft ordinal CE。
        ordinal_ce_per_token = F.binary_cross_entropy(
            pred_cdf.clamp(1e-6, 1 - 1e-6),
            target_cdf,
            reduction="none",
        ).mean(dim=-1)
        ordinal_ce = (ordinal_ce_per_token * mask).sum() / denom

        # 仅在 loss 异常小 (疑似 NaN 或退化) 时打印诊断
        mse_val = mse_norm.item()
        if not np.isfinite(mse_val) or mse_val < 1e-12:
            log.warning(
                f"[step {getattr(self.state, 'global_step', '?')}] "
                f"loss anomaly: mse_norm={mse_val:.4e}, mask_sum={denom.item():.0f}, "
                f"pred_range=[{pred_valid.min():.2f},{pred_valid.max():.2f}], "
                f"target_range=[{target_idx.min():.2f},{target_idx.max():.2f}]"
            )

        loss = mse_norm
        if self._loss_type == "wass1":
            loss = wass1_norm
            if getattr(self.state, "global_step", 0) % 100 == 0:
                log.info(
                    f"[step {getattr(self.state, 'global_step', '?')}] "
                    f"wass1_norm={wass1_norm.item():.4e}"
                )
        elif self._loss_type == "wass2":
            loss = wass2_norm
            if getattr(self.state, "global_step", 0) % 100 == 0:
                log.info(
                    f"[step {getattr(self.state, 'global_step', '?')}] "
                    f"wass2_norm={wass2_norm.item():.4e}"
                )
        elif self._loss_type == "huber":
            loss = huber_norm
            if getattr(self.state, "global_step", 0) % 100 == 0:
                log.info(
                    f"[step {getattr(self.state, 'global_step', '?')}] "
                    f"huber_norm={huber_norm.item():.4e}"
                )
        elif self._loss_type == "crps":
            loss = crps
            if getattr(self.state, "global_step", 0) % 100 == 0:
                log.info(
                    f"[step {getattr(self.state, 'global_step', '?')}] "
                    f"crps={crps.item():.4e}"
                )
        elif self._loss_type == "ordinal_ce":
            loss = ordinal_ce
            if getattr(self.state, "global_step", 0) % 100 == 0:
                log.info(
                    f"[step {getattr(self.state, 'global_step', '?')}] "
                    f"ordinal_ce={ordinal_ce.item():.4e}"
                )
        elif self._loss_type == "mse_ce" and outputs.loss is not None:
            ce = outputs.loss
            loss = mse_norm + self._ce_lambda * ce
            # 每隔 100 步打印一次 MSE_norm / CE 的贡献比例, 帮助调 ce_lambda
            if getattr(self.state, "global_step", 0) % 100 == 0:
                log.info(
                    f"[step {getattr(self.state, 'global_step', '?')}] "
                    f"mse_norm={mse_norm.item():.4e}, ce={ce.item():.4e}, "
                    f"λ·ce={self._ce_lambda * ce.item():.4e}, "
                    f"ce_ratio={self._ce_lambda * ce.item() / max(loss.item(), 1e-12):.1%}"
                )

        return (loss, outputs) if return_outputs else loss


# =============================================================================
# 4. 模型 / LoRA 加载
# =============================================================================


def load_model_for_finetune(
    model_id: str = "amazon/chronos-t5-base",
    n_tokens: int = 4096,
    pad_token_id: int = 0,
    eos_token_id: int = 1,
    random_init: bool = False,
    use_lora: bool = False,
    lora_config: Optional[LoraConfig] = None,
):
    """加载 T5 (seq2seq) 模型, 可选 LoRA 包装。"""
    if random_init:
        config = AutoConfig.from_pretrained(model_id)
        if isinstance(config, T5Config):
            config.initializer_factor = 0.05
        model = AutoModelForSeq2SeqLM.from_config(config)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    model.resize_token_embeddings(n_tokens)
    model.config.pad_token_id = pad_token_id
    model.config.eos_token_id = eos_token_id
    if model.generation_config is not None:
        model.generation_config.pad_token_id = pad_token_id
        model.generation_config.eos_token_id = eos_token_id

    if use_lora and lora_config is not None:
        model = get_peft_model(model, lora_config)
    return model


def create_lora_config(
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
) -> LoraConfig:
    """T5 seq2seq 的 LoRA 配置: 作用于 q/k/v/o 注意力投影层。"""
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["q", "k", "v", "o"],
        task_type=TaskType.SEQ_2_SEQ_LM,
    )


def print_trainable_params(model: torch.nn.Module) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(
        f"Trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / max(total, 1):.2f}%)"
    )


DEFAULT_EXPERIMENTS = [
    "baseline_ce_lora",
    "exp1a_bin_mse_lora",
    "exp2_bin_mse_ce_lora",
    "exp4_bin_wass1_lora",
    "exp5_bin_wass2_lora",
]

EXPERIMENT_ALIASES = {
    "baseline_ce": "baseline_ce_lora",
    "exp1a_bin_mse": "exp1a_bin_mse_lora",
    "exp2_bin_mse_ce": "exp2_bin_mse_ce_lora",
    "exp3_bin_mse_lora": "exp1a_bin_mse_lora",
    "exp4_wass1_lora": "exp4_bin_wass1_lora",
    "exp5_wass2_lora": "exp5_bin_wass2_lora",
    "exp6_huber_lora": "exp_6a_bin_huber_16_lora",
    "exp6_bin_huber_lora": "exp_6a_bin_huber_16_lora",
    "exp_6a_huber_lora": "exp_6a_bin_huber_16_lora",
    "exp7_crps_lora": "exp7_bin_crps_lora",
    "exp8_ordinal_ce_lora": "exp8_bin_ordinal_ce_lora",
}

EXPERIMENT_LABELS = {
    "baseline_ce_lora": "CE+LoRA Fine-tune",
    "exp1a_bin_mse_lora": "bin-MSE+LoRA Fine-tune",
    "exp2_bin_mse_ce_lora": "bin-MSE+CE+LoRA Fine-tune",
    "exp4_bin_wass1_lora": "bin-W1+LoRA Fine-tune",
    "exp5_bin_wass2_lora": "bin-W2+LoRA Fine-tune",
    "exp_6a_bin_huber_16_lora": "bin-Huber(16)+LoRA Fine-tune",
    "exp7_bin_crps_lora": "bin-CRPS+LoRA Fine-tune",
    "exp8_bin_ordinal_ce_lora": "bin-OrdinalCE+LoRA Fine-tune",
}

ZERO_SHOT_LABEL = "Original (zero-shot)"


# =============================================================================
# 5. 训练运行器
# =============================================================================


def run_finetune_experiment(
    experiment_name: str,
    model_id: str,
    training_data_paths: List[str],
    output_dir: str,
    loss_type: str = "ce",
    ce_lambda: float = DEFAULT_CE_LAMBDA,
    huber_delta_bins: float = 1.0,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    learning_rate: float = 1e-4,
    max_steps: int = 500,
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    context_length: int = 512,
    prediction_length: int = 64,
    save_steps: int = 250,
    log_steps: int = 50,
    seed: int = 42,
    data_freq: str = "h",
) -> str:
    """运行一次微调实验, 返回最终 checkpoint 目录。"""
    log.info("=" * 60)
    log.info(f"Experiment: {experiment_name}")
    log.info(
        f"  Model: {model_id} | Loss: {loss_type} | LoRA: {use_lora} | Steps: {max_steps}"
    )
    log.info(f"  Seed: {seed}")
    if loss_type == "mse_ce":
        log.info(f"  ce_lambda: {ce_lambda}")
    if loss_type == "huber":
        log.info(f"  huber_delta_bins: {huber_delta_bins}")
    log.info("=" * 60)

    set_all_seeds(seed)

    exp_dir = Path(output_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # --- Chronos 配置 ---
    chronos_config = ChronosConfig(
        tokenizer_class="MeanScaleUniformBins",
        tokenizer_kwargs={"low_limit": -15.0, "high_limit": 15.0},
        n_tokens=4096,
        n_special_tokens=2,
        pad_token_id=0,
        eos_token_id=1,
        use_eos_token=True,
        model_type="seq2seq",
        context_length=context_length,
        prediction_length=prediction_length,
        num_samples=20,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )
    tokenizer = chronos_config.create_tokenizer()

    # --- 加载模型 ---
    log.info("Loading model...")
    lora_cfg = (
        create_lora_config(r=lora_r, lora_alpha=lora_alpha) if use_lora else None
    )
    model = load_model_for_finetune(
        model_id=model_id,
        n_tokens=4096,
        use_lora=use_lora,
        lora_config=lora_cfg,
    )
    if use_lora:
        print_trainable_params(model)
    model.config.chronos_config = chronos_config.__dict__

    # --- 加载训练数据 ---
    log.info("Loading training data...")
    train_datasets = [
        Filter(
            partial(
                lambda e, ml, mp: len(e["target"]) >= ml
                and np.isnan(e["target"]).mean() <= mp,
                ml=prediction_length,
                mp=0.9,
            ),
            FileDataset(path=Path(p), freq=data_freq),
        )
        for p in training_data_paths
    ]
    probs = [1.0 / len(train_datasets)] * len(train_datasets)

    output_tv = loss_type in (
        "mse",
        "mse_ce",
        "wass1",
        "wass2",
        "huber",
        "crps",
        "ordinal_ce",
    )
    shuffled_ds = ChronosDataset(
        datasets=train_datasets,
        probabilities=probs,
        tokenizer=tokenizer,
        context_length=context_length,
        prediction_length=prediction_length,
        min_past=prediction_length,
        model_type="seq2seq",
        mode="training",
        output_target_values=output_tv,
        seed=seed,
    ).shuffle(buffer_length=100, seed=seed + 10_000)

    # --- TrainingArguments ---
    training_args = TrainingArguments(
        output_dir=str(exp_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type="linear",
        warmup_ratio=0.0,
        optim="adamw_torch_fused",
        logging_strategy="steps",
        logging_steps=log_steps,
        save_strategy="steps",
        save_steps=save_steps,
        report_to=[],
        # 关闭 Trainer 自带 tqdm，避免训练日志延迟冲刷到后续评估阶段。
        disable_tqdm=True,
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_num_workers=0,
        tf32=torch.cuda.is_available(),
        torch_compile=False,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        seed=seed,
        data_seed=seed,
    )

    # --- 选择 Trainer ---
    if loss_type in ("mse", "mse_ce", "wass1", "wass2", "huber", "crps", "ordinal_ce"):
        trainer = BinIndexMSETrainer(
            model=model,
            args=training_args,
            train_dataset=shuffled_ds,
            n_tokens=chronos_config.n_tokens,
            n_special_tokens=chronos_config.n_special_tokens,
            low_limit=-15.0,
            high_limit=15.0,
            loss_type=loss_type,
            ce_lambda=ce_lambda,
            huber_delta_bins=huber_delta_bins,
        )
    else:
        trainer = HighPrecisionTrainer(
            model=model,
            args=training_args,
            train_dataset=shuffled_ds,
        )

    log.info(f"Starting training ({max_steps} steps)...")
    trainer.train()
    log.info(f"Finished training for {experiment_name}")

    final_dir = exp_dir / "checkpoint-final"
    model.save_pretrained(str(final_dir))
    log.info(f"Model saved to {final_dir}")
    return str(final_dir)


# =============================================================================
# 6. 评估: WQL + MASE
# =============================================================================


def load_finetuned_pipeline(checkpoint_path: str) -> ChronosPipeline:
    """加载微调后的 checkpoint 为 ChronosPipeline。

    支持普通 checkpoint (全量微调) 和 LoRA adapter checkpoint (PEFT)。
    """
    ckpt_path = Path(checkpoint_path)
    is_peft = (ckpt_path / "adapter_config.json").exists()

    # 重建 ChronosConfig
    config = AutoConfig.from_pretrained(checkpoint_path) if not is_peft else None
    if config is not None and hasattr(config, "chronos_config") and config.chronos_config is not None:
        chronos_config = ChronosConfig(**config.chronos_config)
    else:
        # PEFT adapter 没有 chronos_config, 用默认
        chronos_config = ChronosConfig(
            tokenizer_class="MeanScaleUniformBins",
            tokenizer_kwargs={"low_limit": -15.0, "high_limit": 15.0},
            n_tokens=4096,
            n_special_tokens=2,
            pad_token_id=0,
            eos_token_id=1,
            use_eos_token=True,
            model_type="seq2seq",
            context_length=512,
            prediction_length=64,
            num_samples=20,
            temperature=1.0,
            top_k=50,
            top_p=1.0,
        )

    if is_peft:
        from peft import PeftModel

        # LoRA checkpoint: 先加载基座, 再挂载 adapter
        # 基座模型 ID 从 adapter_config.json 推断
        with open(ckpt_path / "adapter_config.json") as f:
            adapter_cfg = json.load(f)
        base_model_name = adapter_cfg.get("base_model_name_or_path")
        if not base_model_name:
            raise ValueError(
                f"adapter_config.json at {ckpt_path} missing base_model_name_or_path"
            )
        log.info(f"Loading PEFT adapter: base={base_model_name}, adapter={ckpt_path}")
        inner_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        inner_model = PeftModel.from_pretrained(inner_model, str(ckpt_path))
        # 合并 LoRA 权重到基座, 让 ChronosPipeline 推理时行为一致
        inner_model = inner_model.merge_and_unload()
    else:
        inner_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)

    return ChronosPipeline(
        tokenizer=chronos_config.create_tokenizer(),
        model=ChronosModel(config=chronos_config, model=inner_model),
    )


def evaluate_model(
    pipeline: ChronosPipeline,
    tsf_path: str,
    prediction_length: int,
    n_series: int = 100,
    batch_size: int = 8,
    num_samples: int = 20,
    freq_str: Optional[str] = None,
    seed: int = 42,
    save_dir: Optional[str] = None,
) -> Dict[str, float]:
    """在 tsf 数据集上评估, 返回 WQL 和 MASE。"""
    set_all_seeds(seed)

    series_list, _freq = parse_tsf(tsf_path)
    if freq_str is None:
        freq_str = _freq
    pd_freq = FREQ_MAP.get(freq_str, "D") if freq_str else "D"
    seasonality = SEASONALITY.get(pd_freq, 1)

    def clean(s):
        return pd.Series(s).interpolate(method="linear").ffill().bfill().values

    selected = []
    for s in series_list:
        if len(s) >= prediction_length * 2:
            selected.append(clean(s))
        if len(selected) >= n_series:
            break

    pipeline.model.to(DEVICE)

    all_wql: List[float] = []
    all_mase: List[float] = []
    eval_contexts: List[np.ndarray] = []
    eval_actuals: List[np.ndarray] = []
    point_forecasts: List[np.ndarray] = []
    quantile_forecasts: List[np.ndarray] = []

    for i in tqdm(range(0, len(selected), batch_size), desc="Evaluating"):
        batch_series = selected[i : i + batch_size]
        contexts = [s[:-prediction_length] for s in batch_series]
        actuals = [s[-prediction_length:] for s in batch_series]

        context_tensors = [
            torch.tensor(c, dtype=torch.float32) for c in contexts
        ]
        quantiles, mean = pipeline.predict_quantiles(
            context_tensors,
            prediction_length=prediction_length,
            quantile_levels=QUANTILE_LEVELS,
            num_samples=num_samples,
        )
        quantiles_np = quantiles.numpy()
        mean_np = mean.numpy()

        for j, actual in enumerate(actuals):
            eval_contexts.append(np.asarray(contexts[j], dtype=np.float32))
            eval_actuals.append(np.asarray(actual, dtype=np.float32))
            point_forecasts.append(np.asarray(mean_np[j], dtype=np.float32))
            quantile_forecasts.append(np.asarray(quantiles_np[j], dtype=np.float32))

            # WQL
            wql_num = 0.0
            for qi, q in enumerate(QUANTILE_LEVELS):
                qp = quantiles_np[j, :, qi]
                diff = actual - qp
                wql_num += np.sum(2 * np.maximum(q * diff, (q - 1) * diff))
            denom = np.sum(np.abs(actual))
            wql = wql_num / denom if denom > 0 else float("nan")
            if np.isfinite(wql):
                all_wql.append(wql)

            # MASE
            mae = np.mean(np.abs(actual - mean_np[j]))
            ctx = contexts[j]
            if len(ctx) > seasonality:
                naive_errors = np.abs(ctx[seasonality:] - ctx[:-seasonality])
                naive_scale = (
                    np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
                )
            else:
                naive_scale = 1.0
            mase = mae / naive_scale if naive_scale > 0 else mae
            if np.isfinite(mase):
                all_mase.append(mase)

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path / "predictions.npz",
            contexts=np.array(eval_contexts, dtype=object),
            actuals=np.array(eval_actuals, dtype=object),
            point_forecasts=np.array(point_forecasts, dtype=object),
            quantile_forecasts=np.array(quantile_forecasts, dtype=object),
        )
        log.info(f"Predictions saved to {save_path / 'predictions.npz'}")

    return {
        "WQL": float(np.mean(all_wql)) if all_wql else float("nan"),
        "MASE": float(np.mean(all_mase)) if all_mase else float("nan"),
    }


# =============================================================================
# 7. 可视化
# =============================================================================


def plot_results(
    results: List[Dict],
    save_path: Optional[str] = None,
):
    if not results:
        log.warning("No results to plot.")
        return

    def _in_notebook():
        try:
            from IPython import get_ipython
            shell = get_ipython().__class__.__name__
            return shell in ("ZMQInteractiveShell", "Shell")
        except Exception:
            return False

    import matplotlib
    if not _in_notebook():
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["Model"] for r in results]
    wqls = [r["WQL"] for r in results]
    mases = [r["MASE"] for r in results]
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bars = ax.bar(range(len(names)), wqls, color=colors[: len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("WQL")
    ax.set_title("Weighted Quantile Loss (lower is better)")
    for b, v in zip(bars, wqls):
        if not np.isnan(v):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.002,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax = axes[1]
    bars = ax.bar(range(len(names)), mases, color=colors[: len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("MASE")
    ax.set_title("Mean Absolute Scaled Error (lower is better)")
    for b, v in zip(bars, mases):
        if not np.isnan(v):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.01,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"Figure saved to {save_path}")
    if _in_notebook():
        plt.show()


# =============================================================================
# 8. 主流程
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Chronos-T5 fine-tune (bin-index distance / probabilistic losses)"
    )
    parser.add_argument(
        "--tsf-path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "extracted" / "tourism_monthly_dataset.tsf"),
    )
    parser.add_argument(
        "--arrow-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "arrow_training"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "output" / "chronos_finetune"),
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="amazon/chronos-t5-base",
    )
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Legacy non-LoRA learning rate. Default experiment set no longer uses it.",
    )
    parser.add_argument(
        "--lora-lr",
        type=float,
        default=3e-4,
        help="Learning rate for the default LoRA-only Chronos experiments.",
    )
    parser.add_argument(
        "--ce-lambda",
        type=float,
        default=DEFAULT_CE_LAMBDA,
        help="联合损失中 CE 的权重 λ。注意 MSE 已归一化到 [0,1], "
             "最近实测 λ=0.0005 时 CE 仍常占大头, 因此默认下调到 1e-4。",
    )
    parser.add_argument(
        "--huber-delta-bins",
        type=float,
        default=1.0,
        help="Huber loss 在 bin-index 空间的 delta，单位是 bins。",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=DEFAULT_EXPERIMENTS,
        help="Subset of experiments to run",
    )
    parser.add_argument(
        "--eval-holdout-ratio",
        type=float,
        default=0.2,
        help="评估集占比 (前 1-ratio 训练, 后 ratio 评估)。设为 0 不划分 (旧行为)",
    )
    parser.add_argument("--eval-n-series", type=int, default=100)
    parser.add_argument("--eval-num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-zeroshot",
        action="store_true",
        help="Skip the explicit zero-shot baseline evaluation.",
    )
    parser.add_argument(
        "--no-eval", action="store_true", help="Skip evaluation phase"
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip plotting"
    )
    args = parser.parse_args()
    set_all_seeds(args.seed)
    log.info(f"Global seed: {args.seed}")

    tsf_path = args.tsf_path
    dataset_name = Path(tsf_path).stem
    arrow_subdir = Path(args.arrow_dir) / dataset_name

    # 解析原始 TSF (用于后续导出 eval 子集)
    series_list, freq_str = parse_tsf(tsf_path)
    pd_freq = FREQ_MAP.get(freq_str, "D")
    log.info(
        f"Dataset: {dataset_name}, freq={freq_str} (pd: {pd_freq}), "
        f"series={len(series_list)}"
    )

    # 数据准备 (train/eval split)
    train_feather = arrow_subdir / "data_train.feather"
    eval_feather = arrow_subdir / "data_eval.feather"
    if train_feather.exists() and (
        eval_feather.exists() or args.eval_holdout_ratio == 0
    ):
        arrow_path = str(arrow_subdir)
        train_data_path = str(train_feather)
        eval_data_path = str(eval_feather) if args.eval_holdout_ratio > 0 else train_data_path
        log.info(f"Reusing existing Arrow files in {arrow_subdir}")
    else:
        arrow_path, train_data_path, eval_data_path, _ = tsf_to_arrow(
            tsf_path,
            str(arrow_subdir),
            eval_holdout_ratio=args.eval_holdout_ratio,
        )

    # 导出 eval 子集为独立 TSF (供 evaluate_model 使用)
    if args.eval_holdout_ratio > 0:
        n_eval = max(1, int(len(series_list) * args.eval_holdout_ratio))
        n_train = len(series_list) - n_eval
        eval_series = series_list[n_train:]
        eval_tsf_path = str(arrow_subdir / "eval_data.tsf")
        write_tsf(eval_series, freq_str, eval_tsf_path)
        log.info(
            f"Train: {n_train} series (first {100-args.eval_holdout_ratio*100:.0f}%) | "
            f"Eval: {n_eval} series (last {args.eval_holdout_ratio*100:.0f}%) → {eval_tsf_path}"
        )
    else:
        eval_tsf_path = tsf_path
        log.info("No train/eval split (all series used for both).")

    # --- 训练实验 ---
    exp_dirs = {}
    seen_experiments = set()
    if args.skip_zeroshot:
        log.info("Zero-shot baseline evaluation: skipped (--skip-zeroshot)")
    else:
        log.info(f"Zero-shot baseline evaluation: enabled ({ZERO_SHOT_LABEL})")
    for exp in args.experiments:
        try:
            canonical_exp = EXPERIMENT_ALIASES.get(exp, exp)
            if canonical_exp != exp:
                log.info(f"Mapping legacy experiment name '{exp}' -> '{canonical_exp}'")
            if canonical_exp in seen_experiments:
                log.info(f"Skipping duplicate experiment alias: {exp} -> {canonical_exp}")
                continue
            seen_experiments.add(canonical_exp)

            if canonical_exp == "baseline_ce_lora":
                exp_dirs[exp] = run_finetune_experiment(
                    experiment_name="baseline_ce_lora",
                    model_id=args.model_id,
                    training_data_paths=[train_data_path],
                    output_dir=args.output_dir,
                    loss_type="ce",
                    use_lora=True,
                    learning_rate=args.lora_lr,
                    max_steps=args.max_steps,
                    per_device_train_batch_size=args.batch_size,
                    gradient_accumulation_steps=args.grad_accum,
                    prediction_length=args.prediction_length,
                    save_steps=args.max_steps // 2 or 100,
                    log_steps=max(args.max_steps // 8, 10),
                    seed=args.seed,
                    data_freq=pd_freq.lower(),
                )
            elif canonical_exp == "exp1a_bin_mse_lora":
                exp_dirs[exp] = run_finetune_experiment(
                    experiment_name="exp1a_bin_mse_lora",
                    model_id=args.model_id,
                    training_data_paths=[train_data_path],
                    output_dir=args.output_dir,
                    loss_type="mse",
                    use_lora=True,
                    learning_rate=args.lora_lr,
                    max_steps=args.max_steps,
                    per_device_train_batch_size=args.batch_size,
                    gradient_accumulation_steps=args.grad_accum,
                    prediction_length=args.prediction_length,
                    save_steps=args.max_steps // 2 or 100,
                    log_steps=max(args.max_steps // 8, 10),
                    seed=args.seed,
                    data_freq=pd_freq.lower(),
                )
            elif canonical_exp == "exp2_bin_mse_ce_lora":
                exp_dirs[exp] = run_finetune_experiment(
                    experiment_name="exp2_bin_mse_ce_lora",
                    model_id=args.model_id,
                    training_data_paths=[train_data_path],
                    output_dir=args.output_dir,
                    loss_type="mse_ce",
                    ce_lambda=args.ce_lambda,
                    use_lora=True,
                    learning_rate=args.lora_lr,
                    max_steps=args.max_steps,
                    per_device_train_batch_size=args.batch_size,
                    gradient_accumulation_steps=args.grad_accum,
                    prediction_length=args.prediction_length,
                    save_steps=args.max_steps // 2 or 100,
                    log_steps=max(args.max_steps // 8, 10),
                    seed=args.seed,
                    data_freq=pd_freq.lower(),
                )
            elif canonical_exp == "exp4_bin_wass1_lora":
                exp_dirs[exp] = run_finetune_experiment(
                    experiment_name="exp4_bin_wass1_lora",
                    model_id=args.model_id,
                    training_data_paths=[train_data_path],
                    output_dir=args.output_dir,
                    loss_type="wass1",
                    use_lora=True,
                    learning_rate=args.lora_lr,
                    max_steps=args.max_steps,
                    per_device_train_batch_size=args.batch_size,
                    gradient_accumulation_steps=args.grad_accum,
                    prediction_length=args.prediction_length,
                    save_steps=args.max_steps // 2 or 100,
                    log_steps=max(args.max_steps // 8, 10),
                    seed=args.seed,
                    data_freq=pd_freq.lower(),
                )
            elif canonical_exp == "exp5_bin_wass2_lora":
                exp_dirs[exp] = run_finetune_experiment(
                    experiment_name="exp5_bin_wass2_lora",
                    model_id=args.model_id,
                    training_data_paths=[train_data_path],
                    output_dir=args.output_dir,
                    loss_type="wass2",
                    use_lora=True,
                    learning_rate=args.lora_lr,
                    max_steps=args.max_steps,
                    per_device_train_batch_size=args.batch_size,
                    gradient_accumulation_steps=args.grad_accum,
                    prediction_length=args.prediction_length,
                    save_steps=args.max_steps // 2 or 100,
                    log_steps=max(args.max_steps // 8, 10),
                    seed=args.seed,
                    data_freq=pd_freq.lower(),
                )
            elif canonical_exp == "exp_6a_bin_huber_16_lora":
                exp_dirs[exp] = run_finetune_experiment(
                    experiment_name="exp_6a_bin_huber_16_lora",
                    model_id=args.model_id,
                    training_data_paths=[train_data_path],
                    output_dir=args.output_dir,
                    loss_type="huber",
                    huber_delta_bins=args.huber_delta_bins,
                    use_lora=True,
                    learning_rate=args.lora_lr,
                    max_steps=args.max_steps,
                    per_device_train_batch_size=args.batch_size,
                    gradient_accumulation_steps=args.grad_accum,
                    prediction_length=args.prediction_length,
                    save_steps=args.max_steps // 2 or 100,
                    log_steps=max(args.max_steps // 8, 10),
                    seed=args.seed,
                    data_freq=pd_freq.lower(),
                )
            elif canonical_exp == "exp7_bin_crps_lora":
                exp_dirs[exp] = run_finetune_experiment(
                    experiment_name="exp7_bin_crps_lora",
                    model_id=args.model_id,
                    training_data_paths=[train_data_path],
                    output_dir=args.output_dir,
                    loss_type="crps",
                    use_lora=True,
                    learning_rate=args.lora_lr,
                    max_steps=args.max_steps,
                    per_device_train_batch_size=args.batch_size,
                    gradient_accumulation_steps=args.grad_accum,
                    prediction_length=args.prediction_length,
                    save_steps=args.max_steps // 2 or 100,
                    log_steps=max(args.max_steps // 8, 10),
                    seed=args.seed,
                    data_freq=pd_freq.lower(),
                )
            elif canonical_exp == "exp8_bin_ordinal_ce_lora":
                exp_dirs[exp] = run_finetune_experiment(
                    experiment_name="exp8_bin_ordinal_ce_lora",
                    model_id=args.model_id,
                    training_data_paths=[train_data_path],
                    output_dir=args.output_dir,
                    loss_type="ordinal_ce",
                    use_lora=True,
                    learning_rate=args.lora_lr,
                    max_steps=args.max_steps,
                    per_device_train_batch_size=args.batch_size,
                    gradient_accumulation_steps=args.grad_accum,
                    prediction_length=args.prediction_length,
                    save_steps=args.max_steps // 2 or 100,
                    log_steps=max(args.max_steps // 8, 10),
                    seed=args.seed,
                    data_freq=pd_freq.lower(),
                )
            else:
                log.warning(f"Unknown experiment: {exp}, skipping.")
        except Exception as e:
            log.error(f"Experiment {exp} failed: {e}")
            import traceback as tb

            tb.print_exc()

    # --- 评估 ---
    if args.no_eval:
        log.info("Skipping evaluation (--no-eval).")
        return

    results = []

    # 0. 原始 zero-shot baseline
    if not args.skip_zeroshot:
        log.info(f"\n--- Evaluating: {ZERO_SHOT_LABEL} ---")
        try:
            pipeline = ChronosPipeline.from_pretrained(
                args.model_id, device_map=DEVICE, dtype=torch.float32
            )
            m = evaluate_model(
                pipeline,
                eval_tsf_path,
                args.prediction_length,
                n_series=args.eval_n_series,
                num_samples=args.eval_num_samples,
                freq_str=freq_str,
                seed=args.seed,
                save_dir=args.output_dir,
            )
            results.append({"Model": ZERO_SHOT_LABEL, **m})
            log.info(f"  WQL={m['WQL']:.4f}, MASE={m['MASE']:.4f}")
            del pipeline
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            log.error(f"Original eval failed: {e}")

    # 1..N. 微调实验
    for exp, ckpt_dir in exp_dirs.items():
        log.info(f"\n--- Evaluating: {exp} ---")
        try:
            p = load_finetuned_pipeline(ckpt_dir)
            m = evaluate_model(
                p,
                eval_tsf_path,
                args.prediction_length,
                n_series=args.eval_n_series,
                num_samples=args.eval_num_samples,
                freq_str=freq_str,
                seed=args.seed,
                save_dir=str(Path(ckpt_dir).parent),
            )
            canonical_exp = EXPERIMENT_ALIASES.get(exp, exp)
            results.append({"Model": EXPERIMENT_LABELS.get(canonical_exp, canonical_exp), **m})
            log.info(f"  WQL={m['WQL']:.4f}, MASE={m['MASE']:.4f}")
            del p
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            log.error(f"Eval for {exp} failed: {e}")

    # 汇总 + 保存
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    results_path = Path(args.output_dir) / "results_summary.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {results_path}")

    # 可视化
    if not args.no_plot and results:
        fig_path = Path(args.output_dir) / "finetune_results.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plot_results(results, save_path=str(fig_path))


if __name__ == "__main__":
    main()
