"""
TimesFM 2.5 Baseline 脚本 (Zero-shot + LoRA / Forecast-Head Fine-tune)
========================================================================

使用原生 PyTorch 模型 (TimesFM_2p5_200M_torch):
- Zero-shot 评估: 所有 Monash 数据集对 TimesFM 均为 zero-shot
- LoRA 微调: PEFT + 原始回归损失 (MSE + Quantile)
- Forecast head 微调: 冻结 backbone, 只训练原始 point/quantile 头
- 评估: WQL (概率预测) + MASE (点预测)

运行 (在 ml-hw1 环境中):
    cd ~/ML/project
    python scripts/timesfm_baseline.py --help

依赖:
    conda activate ml-hw1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 添加 timesfm 源码到 path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIMESFM_SRC = os.path.relpath(PROJECT_ROOT / "timesfm" / "src", Path.cwd())
if TIMESFM_SRC not in sys.path:
    sys.path.insert(0, TIMESFM_SRC)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("timesfm_baseline")


# =============================================================================
# 全局配置
# =============================================================================

FREQ_MAP = {
    "yearly": "Y", "quarterly": "Q", "monthly": "M", "weekly": "W",
    "daily": "D", "hourly": "H", "half_hourly": "30min", "10_minutes": "10min",
    "minutely": "min", "4_seconds": "4s",
}

SEASONALITY = {
    "Y": 1, "Q": 4, "M": 12, "W": 1, "D": 7, "H": 24,
    "30min": 48, "10min": 144, "min": 60,
}

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATCH_LEN = 32
OUTPUT_PATCH_LEN = 128
FORECAST_HEAD_FILENAME = "forecast_head.pt"


# =============================================================================
# 0. 随机种子
# =============================================================================


def set_all_seeds(seed: int) -> None:
    """Set all RNGs used by the TimesFM training scripts."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader workers from PyTorch's per-worker initial seed."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


# =============================================================================
# 1. 数据加载 / TSF 解析
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


def clean_series(s: np.ndarray) -> np.ndarray:
    arr = np.asarray(s, dtype=np.float32)
    if not np.isnan(arr).any():
        return arr
    return pd.Series(arr).interpolate(method="linear").ffill().bfill().values.astype(
        np.float32
    )


def split_train_eval(
    series_list: List[np.ndarray], eval_holdout_ratio: float = 0.2,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if eval_holdout_ratio <= 0:
        return series_list, series_list
    n_eval = max(1, int(len(series_list) * eval_holdout_ratio))
    n_train = len(series_list) - n_eval
    return series_list[:n_train], series_list[n_train:]


def split_three_way(
    series_list: List[np.ndarray],
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """三路划分: train / val (early stopping) / test (final eval)。"""
    n = len(series_list)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError(f"Not enough series ({n}) for 3-way split")
    train = series_list[:n_train]
    val = series_list[n_train : n_train + n_val]
    test = series_list[n_train + n_val :]
    return train, val, test


# =============================================================================
# 2. 数据集
# =============================================================================


class RandomWindowDataset(Dataset):
    """随机窗口采样, 避免零填充破坏 RevIN。"""

    def __init__(
        self,
        series_list: List[np.ndarray],
        context_len: int,
        horizon_len: int,
        num_samples: int = 5000,
        seed: int = 42,
    ):
        self.series_list = series_list
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.samples: List[Tuple[int, int]] = []

        rng = np.random.default_rng(seed)
        min_len = context_len + horizon_len
        valid = [i for i, s in enumerate(series_list) if len(s) >= min_len]
        if not valid:
            raise ValueError(
                f"No series long enough ({min_len}). "
                f"Shortest: {min(len(s) for s in series_list)}"
            )
        for _ in range(num_samples):
            idx = rng.choice(valid)
            series = series_list[idx]
            max_start = len(series) - min_len
            start = rng.integers(0, max_start + 1)
            self.samples.append((idx, start))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        idx, start = self.samples[i]
        series = self.series_list[idx]
        end = start + self.context_len + self.horizon_len
        context = torch.tensor(
            series[start : start + self.context_len], dtype=torch.float32
        )
        target = torch.tensor(
            series[start + self.context_len : end], dtype=torch.float32
        )
        return context, target


class LastWindowDataset(Dataset):
    def __init__(self, series_list, context_len, horizon_len):
        self.items = []
        min_len = context_len + horizon_len
        for s in series_list:
            if len(s) >= min_len:
                ctx = torch.tensor(s[-min_len:-horizon_len], dtype=torch.float32)
                tgt = torch.tensor(s[-horizon_len:], dtype=torch.float32)
                self.items.append((ctx, tgt))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


# =============================================================================
# 3. 模型加载
# =============================================================================


def get_hf_cache_roots() -> List[Path]:
    """Return Hugging Face cache roots in the same spirit as huggingface_hub.

    TimesFM needs manual `model.safetensors` loading, so we explicitly scan the
    common hub cache roots before asking `hf_hub_download`.
    """
    roots: List[Path] = []

    def add(path: object) -> None:
        if not path:
            return
        p = Path(path).expanduser()
        if p not in roots:
            roots.append(p)

    add(os.environ.get("HUGGINGFACE_HUB_CACHE"))
    add(os.environ.get("HF_HUB_CACHE"))

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        add(Path(hf_home) / "hub")

    add(os.environ.get("TRANSFORMERS_CACHE"))

    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        add(HF_HUB_CACHE)
    except Exception:
        pass

    add(Path.home() / ".cache" / "huggingface" / "hub")

    try:
        import pwd

        real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
        if real_home != Path.home():
            add(real_home / ".cache" / "huggingface" / "hub")
    except Exception:
        pass

    return roots


def load_model(model_id: str = "google/timesfm-2.5-200m-pytorch"):
    """加载原生 PyTorch TimesFM 2.5 模型 (手动加载, 绕过 HubMixin 兼容性问题)。"""
    from huggingface_hub import hf_hub_download
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module

    log.info(f"Loading native PyTorch model: {model_id}")
    repo_cache_name = f"models--{model_id.replace('/', '--')}"

    weights_path = None
    for hub_root in get_hf_cache_roots():
        cache_root = hub_root / repo_cache_name / "snapshots"
        local_candidates = sorted(cache_root.glob("*/model.safetensors"))
        if local_candidates:
            weights_path = str(local_candidates[-1])
            log.info(f"Loaded TimesFM weights from local snapshot: {weights_path}")
            break

    if weights_path is None:
        download_kwargs = {"repo_id": model_id, "filename": "model.safetensors"}
        try:
            # Prefer the local HF cache first. In restricted environments, a remote
            # HEAD request can fail even when the weights are already cached.
            weights_path = hf_hub_download(local_files_only=True, **download_kwargs)
            log.info("Loaded TimesFM weights from local Hugging Face cache")
        except Exception as exc:
            log.warning(f"Local cache lookup failed, falling back to remote download: {exc}")
            weights_path = hf_hub_download(**download_kwargs)
    nn_module = TimesFM_2p5_200M_torch_module()
    nn_module.load_checkpoint(weights_path, torch_compile=False)
    return nn_module


def apply_lora(nn_module, r=4, lora_alpha=8, lora_dropout=0.05):
    """用 PEFT 包装 LoRA, 返回 PeftModel。"""
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        lora_dropout=lora_dropout,
        bias="none",
    )
    peft_model = get_peft_model(nn_module, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def save_adapter(peft_model, path: str):
    peft_model.save_pretrained(path)


def load_adapter_on_model(model_id: str, adapter_path: str):
    """加载 base model + LoRA adapter, 返回 PeftModel。"""
    from peft import PeftModel

    nn_module = load_model(model_id)
    head_ckpt_path = Path(adapter_path) / FORECAST_HEAD_FILENAME
    if head_ckpt_path.exists():
        log.info(f"Loading forecast head init from adapter dir: {head_ckpt_path}")
        load_forecast_head_into_model(nn_module, adapter_path)
    log.info(f"Loading LoRA adapter: {adapter_path}")
    peft_model = PeftModel.from_pretrained(nn_module, adapter_path)
    peft_model.eval()
    return peft_model


def get_raw_module(peft_or_module):
    """从 PeftModel 中取出原始 nn.Module (用于 decode 推理)。
    PEFT wrapping: PeftModel.base_model.model -> original nn.Module
    """
    if hasattr(peft_or_module, "base_model"):
        base = peft_or_module.base_model
        if hasattr(base, "model"):
            return base.model
        return base
    return peft_or_module


def _reset_module_tree(module) -> None:
    for submodule in module.modules():
        reset_fn = getattr(submodule, "reset_parameters", None)
        if callable(reset_fn):
            reset_fn()


def reset_forecast_head(nn_module) -> None:
    """随机重置 TimesFM 原始 point/quantile 预测头。"""
    _reset_module_tree(nn_module.output_projection_point)
    _reset_module_tree(nn_module.output_projection_quantiles)


def freeze_backbone_except_forecast_head(nn_module) -> None:
    """冻结 backbone, 仅保留原始 forecast head 可训练。"""
    for param in nn_module.parameters():
        param.requires_grad = False
    for param in nn_module.output_projection_point.parameters():
        param.requires_grad = True
    for param in nn_module.output_projection_quantiles.parameters():
        param.requires_grad = True


def count_forecast_head_params(nn_module) -> int:
    return sum(
        p.numel()
        for name, p in nn_module.named_parameters()
        if name.startswith("output_projection_point")
        or name.startswith("output_projection_quantiles")
    )


def save_forecast_head(nn_module, path: str) -> None:
    payload = {
        "output_projection_point": nn_module.output_projection_point.state_dict(),
        "output_projection_quantiles": nn_module.output_projection_quantiles.state_dict(),
    }
    torch.save(payload, Path(path) / FORECAST_HEAD_FILENAME)


def load_forecast_head_into_model(nn_module, checkpoint_dir: str) -> None:
    ckpt_path = Path(checkpoint_dir) / FORECAST_HEAD_FILENAME
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Forecast head checkpoint not found: {ckpt_path}")
    payload = torch.load(str(ckpt_path), map_location=DEVICE)
    nn_module.output_projection_point.load_state_dict(payload["output_projection_point"])
    nn_module.output_projection_quantiles.load_state_dict(
        payload["output_projection_quantiles"]
    )


def load_forecast_head_on_model(model_id: str, checkpoint_dir: str):
    nn_module = load_model(model_id)
    log.info(f"Loading forecast head checkpoint: {checkpoint_dir}")
    load_forecast_head_into_model(nn_module, checkpoint_dir)
    nn_module.eval()
    return nn_module


def describe_forecast_head_checkpoint(checkpoint_dir: str) -> str:
    config_path = Path(checkpoint_dir) / "forecast_head_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        init = config.get("head_init")
        if init == "scratch":
            return "Original Head (from-scratch)"
        if init == "checkpoint":
            return "Original Head (continued)"
    return "Original Head"


# =============================================================================
# 4. 训练辅助: 手动 patching + RevIN + forward
# =============================================================================


def _compute_patch_stats(inputs_patched, masks_patched):
    """计算每个 patch 的累积 running stats (不梯度)。

    inputs_patched: (B, num_patches, p)
    masks_patched: (B, num_patches, p) bool, True=masked
    Returns: mu, sigma 各 (B, num_patches)
    """
    from timesfm.torch.util import update_running_stats

    batch_size, num_patches, p = inputs_patched.shape
    device = inputs_patched.device
    n = torch.zeros(batch_size, device=device)
    mu = torch.zeros(batch_size, device=device)
    sigma = torch.zeros(batch_size, device=device)
    patch_mu, patch_sigma = [], []
    for i in range(num_patches):
        (n, mu, sigma), _ = update_running_stats(
            n, mu, sigma, inputs_patched[:, i], masks_patched[:, i]
        )
        patch_mu.append(mu)
        patch_sigma.append(sigma)
    return torch.stack(patch_mu, dim=1), torch.stack(patch_sigma, dim=1)


def _revin(x, mu, sigma, reverse=False):
    """可逆实例归一化 (复刻 timesfm.torch.util.revin)。"""
    tol = 1e-6
    if len(mu.shape) == len(x.shape) - 1:
        mu = mu[..., None]
        sigma = sigma[..., None]
    elif len(mu.shape) == len(x.shape) - 2:
        mu = mu[..., None, None]
        sigma = sigma[..., None, None]
    if reverse:
        return x * sigma + mu
    else:
        return (x - mu) / torch.where(sigma < tol, 1.0, sigma)


def training_step(model, context, future_values, context_len, horizon_len):
    """一次训练前向步: 手动 patching + RevIN + forward + MSE + Quantile loss。

    Loss 与官方 TimesFM 一致 (modeling_timesfm.py forward):
        loss = MSE(mean_pred, target) + QuantileLoss(quantile_preds, target)
    在原始尺度 (反 RevIN 后) 上计算, 与 official HF model 行为一致。

    model: PeftModel 或 raw nn.Module
    context: (B, context_len) float32
    future_values: (B, horizon_len) float32
    Returns: loss (scalar tensor)
    """
    nn_module = get_raw_module(model)
    p = PATCH_LEN
    batch_size = context.shape[0]
    device = context.device

    # 1. Patch context
    num_patches = context_len // p
    patched_inputs = context.reshape(batch_size, num_patches, p)
    patched_masks = torch.zeros_like(patched_inputs, dtype=torch.bool)

    # 2. 计算 running stats (不需要梯度)
    with torch.no_grad():
        mu, sigma = _compute_patch_stats(patched_inputs, patched_masks)

    # 3. RevIN 归一化
    normed_inputs = _revin(patched_inputs, mu, sigma, reverse=False)
    normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)

    # 4. 前向传播
    (_, _, normed_outputs, _), _ = nn_module(normed_inputs, patched_masks)

    # 5. Reshape + RevIN 反归一化 → 原始尺度
    q = 10
    normed_outputs = normed_outputs.reshape(batch_size, num_patches, OUTPUT_PATCH_LEN, q)
    renormed_outputs = _revin(normed_outputs, mu, sigma, reverse=True)

    # 6. 取最后一个 patch 的前 horizon_len 步
    last_patch = renormed_outputs[:, -1, :horizon_len, :]
    pred_mean = last_patch[:, :, 0]           # (B, horizon_len) mean
    pred_quants = last_patch[:, :, 1:10]      # (B, horizon_len, 9) quantiles

    # 7. MSE loss (与官方一致)
    mse_loss = F.mse_loss(pred_mean, future_values)

    # 8. Quantile loss (与官方 _quantile_loss 一致)
    quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    q_losses = []
    for i, ql in enumerate(quantile_levels):
        errors = future_values - pred_quants[:, :, i]
        q_loss = torch.max((ql - 1) * errors, ql * errors)
        q_losses.append(q_loss.mean())
    quantile_loss = torch.stack(q_losses).mean()

    loss = mse_loss + quantile_loss
    return loss


# =============================================================================
# 5. 批量推理 (用 decode)
# =============================================================================


def batched_inference(model, series_list, context_len, horizon_len, batch_size=16):
    """批量 zero-shot 推理, 返回 (point_forecasts, quantile_forecasts)。

    model: PeftModel 或 raw nn.Module (TimesFM_2p5_200M_torch_module)
    point_forecasts: list[np.ndarray], 每条 (horizon_len,)
    quantile_forecasts: list[np.ndarray], 每条 (horizon_len, 9) for q0.1~q0.9
    """
    nn_module = get_raw_module(model)
    p = PATCH_LEN

    contexts = []
    for s in series_list:
        if len(s) >= context_len:
            ctx = s[-context_len:]
        else:
            ctx = np.zeros(context_len, dtype=np.float32)
            ctx[-len(s):] = s
        contexts.append(ctx)

    point_forecasts = []
    quantile_forecasts = []

    for i in range(0, len(contexts), batch_size):
        batch_ctx = np.array(contexts[i : i + batch_size], dtype=np.float32)

        inputs_t = torch.tensor(batch_ctx, dtype=torch.float32, device=DEVICE)
        masks_t = torch.zeros_like(inputs_t, dtype=torch.bool)

        with torch.no_grad():
            pf_outputs, _, ar_outputs = nn_module.decode(
                horizon_len, inputs_t, masks_t
            )

        last_patch = pf_outputs[:, -1, :horizon_len, :]
        point = last_patch[:, :, 5].cpu().numpy()
        quants = last_patch[:, :, 1:10].cpu().numpy()

        for j in range(len(batch_ctx)):
            point_forecasts.append(point[j])
            quantile_forecasts.append(quants[j])

    return point_forecasts, quantile_forecasts


# =============================================================================
# 6. 评估: WQL + MASE
# =============================================================================


def evaluate_model(
    model,
    series_list: List[np.ndarray],
    prediction_length: int,
    context_length: int,
    n_series: int = 100,
    batch_size: int = 16,
    freq_str: Optional[str] = None,
    save_dir: Optional[str] = None,
    predictions_name: str = "predictions.npz",
) -> Dict[str, float]:
    """评估模型, 返回 WQL + MASE。"""
    pd_freq = FREQ_MAP.get(freq_str, "D") if freq_str else "D"
    seasonality = SEASONALITY.get(pd_freq, 1)

    selected = []
    for s in series_list:
        cleaned = clean_series(s)
        if len(cleaned) >= context_length + prediction_length:
            selected.append(cleaned)
        if len(selected) >= n_series:
            break

    log.info(f"Evaluation: {len(selected)} series, ctx={context_length}, horizon={prediction_length}")

    # 对每条序列, 取 context + actual
    eval_contexts = [s[-(context_length + prediction_length) : -prediction_length] for s in selected]
    eval_actuals = [s[-prediction_length:] for s in selected]

    point_forecasts, quantile_forecasts = batched_inference(
        model, eval_contexts, context_length, prediction_length, batch_size=batch_size,
    )

    all_wql, all_mase = [], []
    for j in range(len(selected)):
        actual = eval_actuals[j]
        pred_mean = point_forecasts[j]
        pred_quants = quantile_forecasts[j]
        ctx = eval_contexts[j]

        # WQL
        wql_num = 0.0
        for qi, q in enumerate(QUANTILE_LEVELS):
            qp = pred_quants[:, qi]
            diff = actual - qp
            wql_num += np.sum(2 * np.maximum(q * diff, (q - 1) * diff))
        denom = np.sum(np.abs(actual))
        wql = wql_num / denom if denom > 0 else float("nan")
        if np.isfinite(wql):
            all_wql.append(wql)

        # MASE
        mae = np.mean(np.abs(actual - pred_mean))
        if len(ctx) > seasonality:
            naive_errors = np.abs(ctx[seasonality:] - ctx[:-seasonality])
            naive_scale = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
        else:
            naive_scale = 1.0
        mase = mae / naive_scale if naive_scale > 0 else mae
        if np.isfinite(mase):
            all_mase.append(mase)

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_path / predictions_name,
            contexts=np.array(eval_contexts, dtype=object),
            actuals=np.array(eval_actuals, dtype=object),
            point_forecasts=np.array(point_forecasts, dtype=object),
            quantile_forecasts=np.array(quantile_forecasts, dtype=object),
        )
        log.info(f"Predictions saved to {save_path / predictions_name}")

    return {
        "WQL": float(np.mean(all_wql)) if all_wql else float("nan"),
        "MASE": float(np.mean(all_mase)) if all_mase else float("nan"),
        "n_series": len(selected),
    }


# =============================================================================
# 7. LoRA 微调
# =============================================================================


def run_finetune(
    model_id: str,
    train_series: List[np.ndarray],
    val_series: List[np.ndarray],
    output_dir: str,
    context_len: int = 128,
    horizon_len: int = 24,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    lora_r: int = 4,
    lora_alpha: int = 8,
    lora_dropout: float = 0.05,
    num_samples: int = 5000,
    num_workers: int = 0,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    mode: str = "lora",
    reset_head: bool = False,
    init_forecast_head_checkpoint: Optional[str] = None,
) -> str:
    """TimesFM 微调, 返回 checkpoint 保存路径。"""
    log.info("=" * 60)
    log.info("TimesFM 2.5 Fine-tune (native PyTorch)")
    log.info(f"  Mode: {mode} | Model: {model_id}")
    if mode == "lora":
        log.info(f"  LoRA r={lora_r}, alpha={lora_alpha}")
    else:
        log.info(f"  Reset forecast head: {reset_head}")
    log.info(f"  Context: {context_len} | Horizon: {horizon_len}")
    log.info(f"  Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    log.info("=" * 60)

    set_all_seeds(seed)
    log.info(f"  Seed: {seed}")

    exp_dir = Path(output_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型 + LoRA
    nn_module = load_model(model_id)
    nn_module.to(DEVICE)
    trainable_model = nn_module
    if mode == "lora":
        if init_forecast_head_checkpoint:
            load_forecast_head_into_model(nn_module, init_forecast_head_checkpoint)
            head_init = "checkpoint"
            log.info(f"Initialized forecast head from {init_forecast_head_checkpoint}")
        else:
            head_init = "pretrained"
        peft_model = apply_lora(
            nn_module, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )
        optimizer_params = list(peft_model.parameters())
        trainable_model = peft_model
        checkpoint_kind = "adapter"
    elif mode == "head":
        if init_forecast_head_checkpoint:
            load_forecast_head_into_model(nn_module, init_forecast_head_checkpoint)
            head_init = "checkpoint"
            log.info(f"Initialized forecast head from {init_forecast_head_checkpoint}")
        elif reset_head:
            reset_forecast_head(nn_module)
            head_init = "scratch"
            log.info("Reset original forecast head to random init")
        else:
            head_init = "pretrained"
            log.info("Using pre-trained original forecast head as init")
        freeze_backbone_except_forecast_head(nn_module)
        optimizer_params = [
            p for p in nn_module.parameters() if p.requires_grad
        ]
        checkpoint_kind = "forecast_head"
        log.info(f"Forecast head params: {count_forecast_head_params(nn_module):,}")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # 数据集
    min_len = context_len + horizon_len
    train_clean = [clean_series(s) for s in train_series if len(s) >= min_len]
    val_clean = [clean_series(s) for s in val_series if len(s) >= min_len]

    train_ds = RandomWindowDataset(
        train_clean, context_len, horizon_len, num_samples=num_samples, seed=seed,
    )
    val_ds = LastWindowDataset(val_clean, context_len, horizon_len)
    pin_memory = DEVICE == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(seed),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(seed + 1),
    )

    log.info(f"Train: {len(train_ds)} ({len(train_loader)} batches) | Val: {len(val_ds)}")

    # 优化器 + 调度器
    optimizer = torch.optim.AdamW(optimizer_params, lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader),
    )

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, epochs + 1):
        trainable_model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for context, target_vals in train_loader:
            context = context.to(DEVICE)
            target_vals = target_vals.to(DEVICE)

            loss = training_step(
                trainable_model, context, target_vals, context_len, horizon_len,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optimizer_params, max_norm=max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # 验证
        trainable_model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for context, target_vals in val_loader:
                context = context.to(DEVICE)
                target_vals = target_vals.to(DEVICE)
                vl = training_step(
                    trainable_model, context, target_vals, context_len, horizon_len,
                )
                val_loss += vl.item()
                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(current_lr)

        log.info(
            f"Epoch {epoch}/{epochs} ({n_batches} steps, {elapsed:.1f}s) — "
            f"train: {avg_train_loss:.4f}, val: {avg_val_loss:.4f}, lr: {current_lr:.2e}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if mode == "lora":
                save_adapter(trainable_model, str(exp_dir))
                if init_forecast_head_checkpoint:
                    shutil.copy2(
                        Path(init_forecast_head_checkpoint) / FORECAST_HEAD_FILENAME,
                        exp_dir / FORECAST_HEAD_FILENAME,
                    )
            else:
                save_forecast_head(nn_module, str(exp_dir))
            log.info(f"  -> saved best {checkpoint_kind} (val={avg_val_loss:.4f})")

    log.info(f"Training complete. Best val loss: {best_val_loss:.4f}")

    train_config = {
        "mode": mode,
        "head_init": head_init,
        "context_len": context_len,
        "horizon_len": horizon_len,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "num_samples": num_samples,
        "seed": seed,
    }
    if mode == "head":
        with open(exp_dir / "forecast_head_config.json", "w") as f:
            json.dump(train_config, f, indent=2)
    elif mode == "lora":
        with open(exp_dir / "adapter_config_extra.json", "w") as f:
            json.dump(train_config, f, indent=2)
    with open(exp_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return str(exp_dir)


# =============================================================================
# 8. 可视化
# =============================================================================


def _in_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell in ("ZMQInteractiveShell", "Shell")
    except Exception:
        return False


def plot_results(results: List[Dict], save_path: Optional[str] = None):
    if not results:
        return
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
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    bars = ax.bar(range(len(names)), mases, color=colors[: len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("MASE")
    ax.set_title("Mean Absolute Scaled Error (lower is better)")
    for b, v in zip(bars, mases):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"Figure saved to {save_path}")
    if _in_notebook():
        plt.show()


# =============================================================================
# 9. 主流程
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="TimesFM 2.5 Baseline (Zero-shot + LoRA / Forecast-Head Fine-tune)"
    )
    parser.add_argument(
        "--tsf-path", type=str,
        default=str(PROJECT_ROOT / "data" / "extracted" / "tourism_monthly_dataset.tsf"),
    )
    parser.add_argument(
        "--tsf-paths", type=str, nargs="+", default=None,
        help="Multiple TSF paths for multi-dataset pre-training. Overrides --tsf-path.",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "output" / "timesfm_finetune"),
    )
    parser.add_argument(
        "--model-id", type=str,
        default="google/timesfm-2.5-200m-pytorch",
    )
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--context-length", type=int, default=128,
                        help="Must be multiple of 32 (patch_length)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mode", choices=["lora", "head"], default="lora")
    parser.add_argument("--reset-forecast-head", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--eval-n-series", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument("--skip-zeroshot", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--forecast-head-checkpoint", type=str, default=None)
    parser.add_argument(
        "--init-forecast-head-checkpoint",
        type=str,
        default=None,
        help="Initialize training from a saved forecast_head.pt checkpoint dir.",
    )
    args = parser.parse_args()

    assert args.context_length % 32 == 0, "context_length must be multiple of 32"
    set_all_seeds(args.seed)
    log.info(f"Global seed: {args.seed}")

    if args.tsf_paths:
        tsf_paths = args.tsf_paths
    else:
        tsf_paths = [args.tsf_path]

    all_series_clean = []
    freq_str = None
    dataset_names = []
    for tsf_path in tsf_paths:
        series_list, f = parse_tsf(tsf_path)
        dataset_name = Path(tsf_path).stem
        dataset_names.append(dataset_name)
        cleaned = [clean_series(s) for s in series_list]
        all_series_clean.extend(cleaned)
        log.info(f"  Loaded {dataset_name}: {len(series_list)} series, freq={f}")
        if freq_str is None:
            freq_str = f

    pd_freq = FREQ_MAP.get(freq_str, "D")
    log.info(
        f"Combined: {len(all_series_clean)} series from {len(tsf_paths)} dataset(s), "
        f"eval_freq={freq_str} (pd: {pd_freq})"
    )

    train_series, val_series, test_series = split_three_way(
        all_series_clean, args.val_ratio, args.test_ratio,
    )
    log.info(
        f"Split: {len(train_series)} train / {len(val_series)} val / "
        f"{len(test_series)} test ({args.val_ratio:.0%}/{args.test_ratio:.0%})"
    )

    results = []

    # --- 1. Zero-shot ---
    if not args.skip_zeroshot:
        log.info("\n--- Evaluating: Zero-shot ---")
        model = load_model(args.model_id)
        model.to(DEVICE)
        model.eval()
        m = evaluate_model(
            model, test_series,
            prediction_length=args.prediction_length,
            context_length=args.context_length,
            n_series=args.eval_n_series,
            freq_str=freq_str,
            save_dir=args.output_dir,
            predictions_name="zeroshot_predictions.npz",
        )
        results.append({"Model": "Zero-shot", **m})
        log.info(f"  WQL={m['WQL']:.4f}, MASE={m['MASE']:.4f} (n={m['n_series']})")
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # --- 2. LoRA Fine-tune ---
    adapter_path = args.adapter_path
    forecast_head_checkpoint = args.forecast_head_checkpoint
    eval_head_only = args.mode == "head" and forecast_head_checkpoint is not None
    if not args.skip_finetune and adapter_path is None and not eval_head_only:
        log.info(f"\n--- {args.mode.upper()} Fine-tune ---")
        checkpoint_path = run_finetune(
            model_id=args.model_id,
            train_series=train_series,
            val_series=val_series,
            output_dir=args.output_dir,
            context_len=args.context_length,
            horizon_len=args.prediction_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            num_samples=args.num_samples,
            num_workers=args.num_workers,
            seed=args.seed,
            mode=args.mode,
            reset_head=args.reset_forecast_head,
            init_forecast_head_checkpoint=args.init_forecast_head_checkpoint,
        )
        if args.mode == "lora":
            adapter_path = checkpoint_path
        else:
            forecast_head_checkpoint = checkpoint_path

    # --- 3. Fine-tuned eval ---
    if adapter_path and Path(adapter_path).exists():
        log.info("\n--- Evaluating: LoRA Fine-tuned ---")
        ft_model = load_adapter_on_model(args.model_id, adapter_path)
        ft_model.to(DEVICE)
        m = evaluate_model(
            ft_model, test_series,
            prediction_length=args.prediction_length,
            context_length=args.context_length,
            n_series=args.eval_n_series,
            freq_str=freq_str,
            save_dir=args.output_dir,
        )
        results.append({"Model": "LoRA Fine-tune", **m})
        log.info(f"  WQL={m['WQL']:.4f}, MASE={m['MASE']:.4f} (n={m['n_series']})")
        del ft_model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    if forecast_head_checkpoint and Path(forecast_head_checkpoint).exists():
        eval_label = describe_forecast_head_checkpoint(forecast_head_checkpoint)
        log.info(f"\n--- Evaluating: {eval_label} ---")
        ft_model = load_forecast_head_on_model(args.model_id, forecast_head_checkpoint)
        ft_model.to(DEVICE)
        m = evaluate_model(
            ft_model, test_series,
            prediction_length=args.prediction_length,
            context_length=args.context_length,
            n_series=args.eval_n_series,
            freq_str=freq_str,
            save_dir=args.output_dir,
        )
        results.append({"Model": eval_label, **m})
        log.info(f"  WQL={m['WQL']:.4f}, MASE={m['MASE']:.4f} (n={m['n_series']})")
        del ft_model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # --- 汇总 ---
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

    if not args.no_plot and results:
        fig_path = Path(args.output_dir) / "baseline_results.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plot_results(results, save_path=str(fig_path))


if __name__ == "__main__":
    main()
