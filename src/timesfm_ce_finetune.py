"""
TimesFM 2.5 CE (Cross-Entropy) Loss 微调脚本
=============================================

把回归模型 TimesFM 改用交叉熵 (分类损失) 微调:
- 新增 CEHead 分类头: backbone output_embeddings (1280维) → logits (128步 × K bins)
- 把连续目标值分箱为离散 bin, 用 F.cross_entropy 训练
- 推理: softmax 期望做点预测, CDF 反演做分位数预测

与 baseline (MSE + Quantile) 形成对照实验。

运行:
    cd ~/ML/project
    python scripts/timesfm_ce_finetune.py --help

依赖:
    conda activate ml-hw1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIMESFM_SRC = os.path.relpath(PROJECT_ROOT / "timesfm" / "src", Path.cwd())
if TIMESFM_SRC not in sys.path:
    sys.path.insert(0, TIMESFM_SRC)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.timesfm_baseline import (
    parse_tsf,
    clean_series,
    split_three_way,
    load_model,
    apply_lora,
    get_raw_module,
    _compute_patch_stats,
    _revin,
    plot_results,
    _in_notebook,
    evaluate_model,
    PATCH_LEN,
    OUTPUT_PATCH_LEN,
    DEVICE,
    FREQ_MAP,
    SEASONALITY,
    QUANTILE_LEVELS,
    RandomWindowDataset,
    LastWindowDataset,
    make_torch_generator,
    seed_worker,
    set_all_seeds,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("timesfm_ce")

DEFAULT_PRETRAIN_HEAD_LR = 1e-3
DEFAULT_DOWNSTREAM_HEAD_LR = 5e-5
DEFAULT_DOWNSTREAM_LORA_LR = 5e-5


# =============================================================================
# 1. 分箱工具
# =============================================================================


def build_bins(
    n_bins: int = 64,
    low: float = -10.0,
    high: float = 10.0,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """构建均匀分箱的 centers 和 boundaries。"""
    centers = torch.linspace(low, high, n_bins, device=device)
    boundaries = torch.cat([
        torch.tensor([-1e20], device=device),
        (centers[1:] + centers[:-1]) / 2,
        torch.tensor([1e20], device=device),
    ])
    return centers, boundaries


def bin_targets(values: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    """连续值 → bin ID。values: (B, H), boundaries: (K+1,) → (B, H) long."""
    ids = torch.bucketize(values, boundaries, right=True) - 1
    return ids.clamp(0, len(boundaries) - 2)


# =============================================================================
# 2. CEHead
# =============================================================================


class CEHead(nn.Module):
    """ResidualBlock 风格分类头: backbone output_embeddings → K-bin logits。

    与 TimesFM 原始 output_projection_point (ResidualBlock) 结构对齐:
      Layer1: Linear(d_model, d_model) + Swish
      Layer2: Linear(d_model, output_patch_len * n_bins)
      Residual: Linear(d_model, output_patch_len * n_bins) (维度不同时的投影)

    Input:  (B, num_patches, 1280)
    Output: (B, num_patches, 128, K)
    """

    def __init__(self, d_model: int = 1280, output_patch_len: int = 128, n_bins: int = 64):
        super().__init__()
        self.o = output_patch_len
        self.K = n_bins
        out_dims = output_patch_len * n_bins
        self.layer1 = nn.Linear(d_model, d_model)
        self.layer2 = nn.Linear(d_model, out_dims)
        self.residual_proj = nn.Linear(d_model, out_dims)

    def forward(self, output_embeddings: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.layer1(output_embeddings))     # (B, P, 1280)
        h = self.layer2(h)                              # (B, P, 128*K)
        h = h + self.residual_proj(output_embeddings)   # residual connection
        return h.reshape(*h.shape[:2], self.o, self.K)  # (B, P, 128, K)


# =============================================================================
# 3. CE 训练步
# =============================================================================


def ce_training_step(
    model,
    ce_head: CEHead,
    context: torch.Tensor,
    future_values: torch.Tensor,
    context_len: int,
    horizon_len: int,
    bin_centers: torch.Tensor,
    bin_boundaries: torch.Tensor,
) -> torch.Tensor:
    """CE 训练步: patch + RevIN + backbone forward + CE head + cross_entropy。"""
    nn_module = get_raw_module(model)
    p = PATCH_LEN
    batch_size = context.shape[0]
    device = context.device

    # 1. Patch context
    num_patches = context_len // p
    patched_inputs = context.reshape(batch_size, num_patches, p)
    patched_masks = torch.zeros_like(patched_inputs, dtype=torch.bool)

    # 2. RevIN running stats
    with torch.no_grad():
        mu, sigma = _compute_patch_stats(patched_inputs, patched_masks)

    # 3. RevIN normalize
    normed_inputs = _revin(patched_inputs, mu, sigma, reverse=False)
    normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)

    # 4. Backbone forward → output_embeddings (B, num_patches, 1280)
    (_, output_embeddings, _, _), _ = nn_module(normed_inputs, patched_masks)

    # 5. CE head → logits (B, num_patches, 128, K)
    logits = ce_head(output_embeddings)

    # 6. Normalize targets with last patch RevIN stats
    mu_last = mu[:, -1:]       # (B, 1)
    sigma_last = sigma[:, -1:]  # (B, 1)
    safe_sigma = torch.where(sigma_last < 1e-6, torch.ones_like(sigma_last), sigma_last)
    normed_targets = (future_values - mu_last) / safe_sigma

    # 7. Bin targets → labels (B, horizon_len)
    boundaries = bin_boundaries.to(device)
    labels = bin_targets(normed_targets, boundaries)

    # 8. CE loss on last patch, first horizon_len steps
    step_logits = logits[:, -1, :horizon_len, :]  # (B, H, K)
    loss = F.cross_entropy(
        step_logits.reshape(-1, ce_head.K),
        labels.reshape(-1),
    )
    return loss


def compute_diagnostic_mae(
    model,
    ce_head: CEHead,
    val_loader,
    context_len: int,
    horizon_len: int,
    bin_centers: torch.Tensor,
    max_batches: int = 5,
) -> float:
    """诊断指标: 用 softmax 期望做点预测, 在原始尺度上算 MAE。

    不走完整 evaluate_model (太慢), 只取几个 batch 快速估算。
    """
    nn_module = get_raw_module(model)
    p = PATCH_LEN
    centers = bin_centers.to(DEVICE)
    total_mae = 0.0
    n_samples = 0

    ce_head.eval()
    nn_module.eval()

    with torch.no_grad():
        for i, (context, future_values) in enumerate(val_loader):
            if i >= max_batches:
                break
            context = context.to(DEVICE)
            future_values = future_values.to(DEVICE)
            B = context.shape[0]
            num_patches = context_len // p

            patched = context.reshape(B, num_patches, p)
            masks = torch.zeros_like(patched, dtype=torch.bool)
            mu, sigma = _compute_patch_stats(patched, masks)
            normed = _revin(patched, mu, sigma, reverse=False)
            normed = torch.where(masks, 0.0, normed)

            (_, output_emb, _, _), _ = nn_module(normed, masks)
            logits = ce_head(output_emb)  # (B, P, 128, K)

            step_logits = logits[:, -1, :horizon_len, :]  # (B, H, K)
            probs = F.softmax(step_logits, dim=-1)
            normed_point = (probs * centers).sum(dim=-1)  # (B, H)

            # RevIN reverse
            mu_last = mu[:, -1:]
            sigma_last = sigma[:, -1:]
            safe_sigma = torch.where(sigma_last < 1e-6, torch.ones_like(sigma_last), sigma_last)
            point = normed_point * safe_sigma + mu_last

            mae = (point - future_values).abs().mean().item()
            total_mae += mae * B
            n_samples += B

    return total_mae / max(n_samples, 1)


# =============================================================================
# 4. CE 推理
# =============================================================================


def ce_inference(
    model,
    ce_head: CEHead,
    series_list: List[np.ndarray],
    context_len: int,
    horizon_len: int,
    bin_centers: torch.Tensor,
    bin_boundaries: torch.Tensor,
    batch_size: int = 16,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """CE 推理: backbone → CE head → softmax → point + quantile forecasts。"""
    nn_module = get_raw_module(model)
    p = PATCH_LEN

    # 准备 contexts
    contexts = []
    for s in series_list:
        if len(s) >= context_len:
            ctx = s[-context_len:]
        else:
            ctx = np.zeros(context_len, dtype=np.float32)
            ctx[-len(s):] = s
        contexts.append(ctx)

    centers = bin_centers.to(DEVICE)
    point_forecasts = []
    quantile_forecasts = []

    for i in range(0, len(contexts), batch_size):
        batch_ctx = np.array(contexts[i : i + batch_size], dtype=np.float32)
        B = batch_ctx.shape[0]

        inputs_t = torch.tensor(batch_ctx, dtype=torch.float32, device=DEVICE)
        num_patches = context_len // p
        patched = inputs_t.reshape(B, num_patches, p)
        masks = torch.zeros_like(patched, dtype=torch.bool)

        with torch.no_grad():
            mu, sigma = _compute_patch_stats(patched, masks)
            normed = _revin(patched, mu, sigma, reverse=False)
            normed = torch.where(masks, 0.0, normed)

            (_, output_emb, _, _), _ = nn_module(normed, masks)
            logits = ce_head(output_emb)  # (B, P, 128, K)

        # 取最后 patch 前 horizon_len 步
        step_logits = logits[:, -1, :horizon_len, :]  # (B, H, K)
        probs = F.softmax(step_logits, dim=-1)          # (B, H, K)

        # 点预测: softmax 期望 (归一化空间)
        normed_point = (probs * centers).sum(dim=-1)    # (B, H)

        # 分位数预测: CDF 反演
        cdf = probs.cumsum(dim=-1)                      # (B, H, K)
        normed_quants = []
        for q in QUANTILE_LEVELS:
            q_tensor = torch.full((*cdf.shape[:2], 1), q, device=DEVICE)
            idx = torch.searchsorted(cdf, q_tensor).clamp(0, len(centers) - 1)
            q_values = centers[idx.squeeze(-1)]          # (B, H)
            normed_quants.append(q_values)
        normed_quants = torch.stack(normed_quants, dim=-1)  # (B, H, 9)

        # RevIN 反归一化 → 原始尺度
        mu_last = mu[:, -1:]
        sigma_last = sigma[:, -1:]
        safe_sigma = torch.where(sigma_last < 1e-6, torch.ones_like(sigma_last), sigma_last)

        point = normed_point * safe_sigma + mu_last               # (B, H)
        quants = normed_quants * safe_sigma.unsqueeze(-1) + mu_last.unsqueeze(-1)  # (B, H, 9)

        for j in range(B):
            point_forecasts.append(point[j].cpu().numpy())
            quantile_forecasts.append(quants[j].cpu().numpy())

    return point_forecasts, quantile_forecasts


# =============================================================================
# 5. CE 评估
# =============================================================================


def evaluate_model_ce(
    model,
    ce_head: CEHead,
    series_list: List[np.ndarray],
    prediction_length: int,
    context_length: int,
    bin_centers: torch.Tensor,
    bin_boundaries: torch.Tensor,
    n_series: int = 100,
    batch_size: int = 16,
    freq_str: Optional[str] = None,
    save_dir: Optional[str] = None,
) -> Dict[str, float]:
    """评估 CE 模型的 WQL + MASE。"""
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

    eval_contexts = [s[-(context_length + prediction_length) : -prediction_length] for s in selected]
    eval_actuals = [s[-prediction_length:] for s in selected]

    point_forecasts, quantile_forecasts = ce_inference(
        model, ce_head, eval_contexts, context_length, prediction_length,
        bin_centers, bin_boundaries, batch_size=batch_size,
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
        "n_series": len(selected),
    }


# =============================================================================
# 6. CE 微调训练循环
# =============================================================================


def run_finetune_ce(
    model_id: str,
    train_series: List[np.ndarray],
    val_series: List[np.ndarray],
    output_dir: str,
    context_len: int = 128,
    horizon_len: int = 24,
    epochs: int = 10,
    batch_size: int = 32,
    n_bins: int = 64,
    bin_low: float = -10.0,
    bin_high: float = 10.0,
    mode: str = "lora",
    head_lr: Optional[float] = None,
    lora_lr: Optional[float] = None,
    lora_r: int = 4,
    lora_alpha: int = 8,
    lora_dropout: float = 0.05,
    num_samples: int = 5000,
    num_workers: int = 0,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    ce_head_checkpoint: Optional[str] = None,
) -> Tuple[str, CEHead, torch.Tensor, torch.Tensor]:
    """CE 微调, 返回 (checkpoint_dir, ce_head, bin_centers, bin_boundaries)。"""
    if head_lr is None:
        head_lr = DEFAULT_PRETRAIN_HEAD_LR if mode == "frozen" else DEFAULT_DOWNSTREAM_HEAD_LR
    if lora_lr is None:
        lora_lr = DEFAULT_DOWNSTREAM_LORA_LR

    log.info("=" * 60)
    log.info("TimesFM 2.5 CE Fine-tune")
    log.info(f"  Mode: {mode} | Bins: {n_bins} [{bin_low}, {bin_high}]")
    log.info(f"  Context: {context_len} | Horizon: {horizon_len}")
    lr_msg = f"  Epochs: {epochs} | Batch: {batch_size} | Head LR: {head_lr}"
    if mode == "lora":
        lr_msg += f" | LoRA LR: {lora_lr}"
    log.info(lr_msg)
    log.info("=" * 60)

    set_all_seeds(seed)
    log.info(f"  Seed: {seed}")

    exp_dir = Path(output_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Bins
    bin_centers, bin_boundaries = build_bins(n_bins, bin_low, bin_high)

    # Model
    nn_module = load_model(model_id)
    nn_module.to(DEVICE)

    if mode == "lora":
        peft_model = apply_lora(nn_module, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        trainable = peft_model
    else:
        for param in nn_module.parameters():
            param.requires_grad = False
        trainable = nn_module

    # CE head
    ce_head = CEHead(d_model=1280, output_patch_len=OUTPUT_PATCH_LEN, n_bins=n_bins)
    if ce_head_checkpoint:
        ckpt_path = Path(ce_head_checkpoint) / "ce_head.pt"
        if ckpt_path.exists():
            ce_head.load_state_dict(torch.load(str(ckpt_path), map_location=DEVICE))
            log.info(f"Loaded pre-trained CE head from {ckpt_path}")
        else:
            log.warning(f"CE head checkpoint not found: {ckpt_path}")
    ce_head.to(DEVICE)
    ce_params = sum(p.numel() for p in ce_head.parameters())
    log.info(f"CE head params: {ce_params:,}")

    # Optimizer (parameter groups)
    param_groups = [{"params": ce_head.parameters(), "lr": head_lr}]
    if mode == "lora":
        param_groups.append({"params": peft_model.parameters(), "lr": lora_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    # Data
    min_len = context_len + horizon_len
    train_clean = [clean_series(s) for s in train_series if len(s) >= min_len]
    val_clean = [clean_series(s) for s in val_series if len(s) >= min_len]

    train_ds = RandomWindowDataset(train_clean, context_len, horizon_len, num_samples=num_samples, seed=seed)
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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader),
    )

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "diag_mae": [], "lr": []}

    for epoch in range(1, epochs + 1):
        ce_head.train()
        if mode == "lora":
            peft_model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for context, target_vals in train_loader:
            context = context.to(DEVICE, non_blocking=True)
            target_vals = target_vals.to(DEVICE, non_blocking=True)

            loss = ce_training_step(
                trainable, ce_head, context, target_vals,
                context_len, horizon_len, bin_centers, bin_boundaries,
            )
            loss.backward()
            all_params = list(ce_head.parameters())
            if mode == "lora":
                all_params += list(peft_model.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)

        # Validation
        ce_head.eval()
        if mode == "lora":
            peft_model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for context, target_vals in val_loader:
                context = context.to(DEVICE, non_blocking=True)
                target_vals = target_vals.to(DEVICE, non_blocking=True)
                vl = ce_training_step(
                    trainable, ce_head, context, target_vals,
                    context_len, horizon_len, bin_centers, bin_boundaries,
                )
                val_loss += vl.item()
                val_batches += 1

        avg_val = val_loss / max(val_batches, 1)
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        # Diagnostic MAE (softmax expectation, original scale)
        diag_mae = compute_diagnostic_mae(
            trainable, ce_head, val_loader,
            context_len, horizon_len, bin_centers, max_batches=5,
        )

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["diag_mae"].append(diag_mae)
        history["lr"].append(current_lr)

        log.info(
            f"Epoch {epoch}/{epochs} ({n_batches} steps, {elapsed:.1f}s) — "
            f"train: {avg_train:.4f}, val: {avg_val:.4f}, MAE: {diag_mae:.2f}, lr: {current_lr:.2e}"
        )

        # 每个 epoch 保存一次 history, 方便实时监控
        with open(exp_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            # Save CE head
            torch.save(ce_head.state_dict(), exp_dir / "ce_head.pt")
            # Save LoRA adapter if applicable
            if mode == "lora":
                peft_model.save_pretrained(str(exp_dir))
            log.info(f"  -> saved best checkpoint (val={avg_val:.4f})")

    log.info(f"Training complete. Best val loss: {best_val_loss:.4f}")

    # Save config + history
    config = {
        "n_bins": n_bins, "bin_low": bin_low, "bin_high": bin_high,
        "mode": mode, "context_len": context_len, "horizon_len": horizon_len,
    }
    with open(exp_dir / "ce_config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(exp_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return str(exp_dir), ce_head, bin_centers, bin_boundaries


# =============================================================================
# 7. Checkpoint 加载
# =============================================================================


def load_ce_checkpoint(
    model_id: str,
    checkpoint_dir: str,
) -> Tuple[object, CEHead, torch.Tensor, torch.Tensor]:
    """加载 CE checkpoint, 返回 (model, ce_head, bin_centers, bin_boundaries)。"""
    exp_dir = Path(checkpoint_dir)

    with open(exp_dir / "ce_config.json") as f:
        config = json.load(f)

    bin_centers, bin_boundaries = build_bins(
        config["n_bins"], config["bin_low"], config["bin_high"],
    )

    # Load model
    nn_module = load_model(model_id)
    nn_module.to(DEVICE)

    adapter_path = exp_dir / "adapter_config.json"
    if adapter_path.exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(nn_module, str(exp_dir))
        model.eval()
    else:
        model = nn_module
        model.eval()

    # Load CE head
    ce_head = CEHead(
        d_model=1280,
        output_patch_len=OUTPUT_PATCH_LEN,
        n_bins=config["n_bins"],
    )
    ce_head.load_state_dict(torch.load(exp_dir / "ce_head.pt", map_location=DEVICE))
    ce_head.to(DEVICE)
    ce_head.eval()

    return model, ce_head, bin_centers, bin_boundaries


def describe_ce_checkpoint(checkpoint_dir: str) -> str:
    """Return a human-readable label for a CE checkpoint."""
    exp_dir = Path(checkpoint_dir)
    if (exp_dir / "adapter_config.json").exists():
        return "CE Fine-tune (lora)"
    return "CE Head (pretrained)"


# =============================================================================
# 8. 主流程
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="TimesFM 2.5 CE Fine-tune (cross-entropy loss)"
    )
    parser.add_argument(
        "--tsf-path", type=str,
        default=str(PROJECT_ROOT / "data" / "extracted" / "tourism_monthly_dataset.tsf"),
    )
    parser.add_argument(
        "--tsf-paths", type=str, nargs="+", default=None,
        help="Multiple TSF paths for multi-dataset pre-training. Overrides --tsf-path.",
    )
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "output" / "timesfm_ce_finetune"))
    parser.add_argument("--model-id", type=str, default="google/timesfm-2.5-200m-pytorch")
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--n-bins", type=int, default=64,
                        help="Number of bins. Current CEHead uses layer1 + layer2 + residual_proj. "
                             "K=64 → 22.6M, K=128 → 43.6M, K=256 → 85.6M params")
    parser.add_argument("--bin-range", type=float, nargs=2, default=[-10.0, 10.0])
    parser.add_argument("--mode", choices=["frozen", "lora"], default="lora")
    parser.add_argument(
        "--head-lr",
        type=float,
        default=None,
        help="CE head learning rate. Default: 1e-3 for --mode frozen, 5e-5 for --mode lora.",
    )
    parser.add_argument(
        "--lora-lr",
        type=float,
        default=None,
        help="LoRA learning rate. Default: 5e-5.",
    )
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--eval-n-series", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument("--skip-zeroshot", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--ce-checkpoint", type=str, default=None)
    parser.add_argument("--ce-head-checkpoint", type=str, default=None,
                        help="Path to pre-trained CE head checkpoint (ce_head.pt dir) to initialize from")
    args = parser.parse_args()

    assert args.context_length % 32 == 0, "context_length must be multiple of 32"
    set_all_seeds(args.seed)
    log.info(f"Global seed: {args.seed}")

    # Multi-dataset or single-dataset
    if args.tsf_paths:
        tsf_paths = args.tsf_paths
    else:
        tsf_paths = [args.tsf_path]

    all_series_clean = []
    freq_str = None
    for tsf_path in tsf_paths:
        series_list, f = parse_tsf(tsf_path)
        name = Path(tsf_path).stem
        cleaned = [clean_series(s) for s in series_list]
        all_series_clean.extend(cleaned)
        log.info(f"  Loaded {name}: {len(series_list)} series, freq={f}")
        if freq_str is None:
            freq_str = f  # use first dataset's freq for eval

    log.info(f"Combined: {len(all_series_clean)} series from {len(tsf_paths)} dataset(s)")

    train_series, val_series, test_series = split_three_way(
        all_series_clean, args.val_ratio, args.test_ratio,
    )
    log.info(
        f"Split: {len(train_series)} train / {len(val_series)} val / "
        f"{len(test_series)} test ({args.val_ratio:.0%}/{args.test_ratio:.0%})"
    )

    results = []

    # --- 1. Zero-shot (用 baseline 的 evaluate_model, 走 decode) ---
    if not args.skip_zeroshot:
        log.info("\n--- Evaluating: Zero-shot ---")
        zs_model = load_model(args.model_id)
        zs_model.to(DEVICE)
        zs_model.eval()
        m = evaluate_model(
            zs_model, test_series,
            prediction_length=args.prediction_length,
            context_length=args.context_length,
            n_series=args.eval_n_series,
            freq_str=freq_str,
        )
        results.append({"Model": "Zero-shot", **m})
        log.info(f"  WQL={m['WQL']:.4f}, MASE={m['MASE']:.4f} (n={m['n_series']})")

        # 保存 zero-shot 预测
        from src.timesfm_baseline import batched_inference as baseline_inference
        zs_contexts = [s[-(args.context_length + args.prediction_length):-args.prediction_length]
                       for s in test_series[:args.eval_n_series]
                       if len(s) >= args.context_length + args.prediction_length]
        zs_point, zs_quants = baseline_inference(
            zs_model, zs_contexts, args.context_length, args.prediction_length,
        )
        zs_save = Path(args.output_dir)
        zs_save.mkdir(parents=True, exist_ok=True)
        np.savez(
            zs_save / "zeroshot_predictions.npz",
            contexts=np.array(zs_contexts, dtype=object),
            point_forecasts=np.array(zs_point, dtype=object),
            quantile_forecasts=np.array(zs_quants, dtype=object),
        )
        log.info(f"Zero-shot predictions saved to {zs_save / 'zeroshot_predictions.npz'}")

        del zs_model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # --- 2. CE Fine-tune ---
    ce_checkpoint = args.ce_checkpoint
    if not args.skip_finetune and ce_checkpoint is None:
        log.info("\n--- CE Fine-tune ---")
        ckpt_dir, ce_head, bin_centers, bin_boundaries = run_finetune_ce(
            model_id=args.model_id,
            train_series=train_series,
            val_series=val_series,
            output_dir=args.output_dir,
            context_len=args.context_length,
            horizon_len=args.prediction_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            n_bins=args.n_bins,
            bin_low=args.bin_range[0],
            bin_high=args.bin_range[1],
            mode=args.mode,
            head_lr=args.head_lr,
            lora_lr=args.lora_lr,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            num_samples=args.num_samples,
            seed=args.seed,
            ce_head_checkpoint=args.ce_head_checkpoint,
        )
        ce_checkpoint = ckpt_dir

    # --- 3. CE evaluation ---
    if ce_checkpoint and Path(ce_checkpoint).exists():
        eval_label = describe_ce_checkpoint(ce_checkpoint)
        log.info(f"\n--- Evaluating: {eval_label} ---")
        ft_model, ce_head, bin_centers, bin_boundaries = load_ce_checkpoint(
            args.model_id, ce_checkpoint,
        )
        m = evaluate_model_ce(
            ft_model, ce_head, test_series,
            prediction_length=args.prediction_length,
            context_length=args.context_length,
            bin_centers=bin_centers,
            bin_boundaries=bin_boundaries,
            n_series=args.eval_n_series,
            freq_str=freq_str,
            save_dir=args.output_dir,
        )
        results.append({"Model": eval_label, **m})
        log.info(f"  WQL={m['WQL']:.4f}, MASE={m['MASE']:.4f} (n={m['n_series']})")
        del ft_model, ce_head
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
        fig_path = Path(args.output_dir) / "ce_results.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plot_results(results, save_path=str(fig_path))


if __name__ == "__main__":
    main()
