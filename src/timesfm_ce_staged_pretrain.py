"""
TimesFM 2.5 CE head staged pretraining (64-bin by default)
==========================================================

Purpose:
- Train the CE head from scratch with a frozen TimesFM backbone.
- Large stage can auto-bootstrap from the old starter pool when no checkpoint is
  provided, so users only need to request `large`.

Run:
    cd ~/ML/project
    /home/istina/miniconda3/envs/ml-hw1/bin/python \
        scripts/timesfm_ce_staged_pretrain.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.timesfm_baseline import (
    clean_series,
    parse_tsf,
    set_all_seeds,
    split_three_way,
)
from src.timesfm_ce_finetune import run_finetune_ce

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("timesfm_ce_staged_pretrain")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "extracted"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "timesfm_ce_staged_pretrain"

STAGE_PRESETS: Dict[str, Dict[str, object]] = {
    "legacy64": {
        "datasets": [
            "m4_monthly_dataset.tsf",
            "m4_daily_dataset.tsf",
            "weather_dataset.tsf",
            "electricity_hourly_dataset.tsf",
            "traffic_hourly_dataset.tsf",
        ],
        "epochs": 5,
        "batch_size": 256,
        "num_samples": 200_000,
    },
    "starter": {
        "datasets": [
            "m4_monthly_dataset.tsf",
            "temperature_rain_dataset_without_missing_values.tsf",
            "weather_dataset.tsf",
            "traffic_hourly_dataset.tsf",
            "electricity_hourly_dataset.tsf",
        ],
        "epochs": 3,
        "batch_size": 128,
        "num_samples": 100_000,
    },
    "large": {
        "datasets": [
            "kaggle_web_traffic_dataset_without_missing_values.tsf",
            "temperature_rain_dataset_without_missing_values.tsf",
            "m4_monthly_dataset.tsf",
            "m4_daily_dataset.tsf",
            "weather_dataset.tsf",
            "rideshare_dataset_without_missing_values.tsf",
            "traffic_hourly_dataset.tsf",
            "m4_hourly_dataset.tsf",
            "electricity_hourly_dataset.tsf",
            "vehicle_trips_dataset_without_missing_values.tsf",
            "kdd_cup_2018_dataset_without_missing_values.tsf",
            "covid_deaths_dataset.tsf",
            "nn5_daily_dataset_without_missing_values.tsf",
            "fred_md_dataset.tsf",
        ],
        "epochs": 5,
        "batch_size": 128,
        "num_samples": 300_000,
    },
}


def load_pool_series(
    dataset_names: List[str],
    min_len: int,
) -> Tuple[List[np.ndarray], List[Dict[str, object]]]:
    """Load and clean one pool, dropping series shorter than min_len."""
    all_series: List[np.ndarray] = []
    dataset_stats: List[Dict[str, object]] = []

    for name in dataset_names:
        path = DATA_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        series_list, freq = parse_tsf(str(path))
        valid_series: List[np.ndarray] = []
        lengths: List[int] = []
        for series in series_list:
            cleaned = clean_series(series)
            lengths.append(len(cleaned))
            if len(cleaned) >= min_len:
                valid_series.append(cleaned)

        raw_count = len(series_list)
        valid_count = len(valid_series)
        avg_len = round(float(np.mean(lengths)), 1) if lengths else 0.0
        dataset_stats.append({
            "dataset": name,
            "freq": freq,
            "raw_series": raw_count,
            "valid_series": valid_count,
            "avg_len": avg_len,
            "min_len": min(lengths) if lengths else 0,
            "max_len": max(lengths) if lengths else 0,
        })
        all_series.extend(valid_series)

        log.info(
            f"  Loaded {name}: raw={raw_count}, valid>={min_len}={valid_count}, "
            f"freq={freq}, avg_len={avg_len}"
        )

        del series_list
        del valid_series

    return all_series, dataset_stats


def stage_output_dir(output_root: Path, stage_idx: int, stage_name: str) -> Path:
    return output_root / f"{stage_idx:02d}_{stage_name}"


def stage_complete_marker(stage_dir: Path) -> Path:
    return stage_dir / "stage_complete.json"


def save_stage_metadata(
    stage_dir: Path,
    stage_name: str,
    stage_stats: List[Dict[str, object]],
    train_count: int,
    val_count: int,
    test_count: int,
    config: Dict[str, object],
) -> None:
    payload = {
        "stage": stage_name,
        "datasets": stage_stats,
        "split": {
            "train_series": train_count,
            "val_series": val_count,
            "test_series": test_count,
        },
        "config": config,
    }
    stage_dir.mkdir(parents=True, exist_ok=True)
    with open(stage_dir / "stage_plan.json", "w") as f:
        json.dump(payload, f, indent=2)


def mark_stage_complete(stage_dir: Path, stage_name: str, checkpoint_dir: str) -> None:
    payload = {
        "stage": stage_name,
        "checkpoint_dir": checkpoint_dir,
        "status": "completed",
    }
    with open(stage_complete_marker(stage_dir), "w") as f:
        json.dump(payload, f, indent=2)


def maybe_bootstrap_large_stage(
    stage_name: str,
    stage_idx: int,
    output_root: Path,
    prev_checkpoint: Optional[str],
    args: argparse.Namespace,
) -> Tuple[Optional[str], Optional[Path]]:
    """Auto-run starter warmup before large when no init checkpoint is given.

    We no longer surface `starter` as a first-class public result for new64
    reproduction, but the old large checkpoint was trained as a continuation
    from starter. This helper preserves that behavior.
    """
    if stage_name != "large" or prev_checkpoint is not None:
        return prev_checkpoint, None

    bootstrap_dir = output_root / f"{stage_idx:02d}_large_bootstrap_starter"
    if args.reuse_checkpoints and stage_complete_marker(bootstrap_dir).exists():
        log.info(f"Reusing large bootstrap checkpoint: {bootstrap_dir}")
        return str(bootstrap_dir), bootstrap_dir

    log.info("=" * 72)
    log.info("Large stage requested without init checkpoint; auto-bootstrapping from starter")
    bootstrap_args = argparse.Namespace(**vars(args))
    setattr(bootstrap_args, "starter_epochs", args.starter_epochs)
    setattr(bootstrap_args, "starter_batch_size", args.starter_batch_size)
    setattr(bootstrap_args, "starter_num_samples", args.starter_num_samples)
    ckpt_dir = run_stage(
        stage_name="starter",
        stage_idx=stage_idx,
        output_root=output_root,
        prev_checkpoint=None,
        args=bootstrap_args,
        stage_dir_override=bootstrap_dir,
        display_name="large_bootstrap_starter",
    )
    return ckpt_dir, bootstrap_dir


def run_stage(
    stage_name: str,
    stage_idx: int,
    output_root: Path,
    prev_checkpoint: Optional[str],
    args: argparse.Namespace,
    stage_dir_override: Optional[Path] = None,
    display_name: Optional[str] = None,
) -> str:
    preset = STAGE_PRESETS[stage_name]
    min_len = args.context_length + args.prediction_length
    epochs = getattr(args, f"{stage_name}_epochs")
    batch_size = getattr(args, f"{stage_name}_batch_size")
    num_samples = getattr(args, f"{stage_name}_num_samples")
    stage_dir = stage_dir_override or stage_output_dir(output_root, stage_idx, stage_name)
    stage_label = display_name or stage_name

    if args.reuse_checkpoints and stage_complete_marker(stage_dir).exists():
        log.info(f"Reusing existing stage checkpoint: {stage_dir}")
        return str(stage_dir)

    log.info("=" * 72)
    log.info(f"Stage {stage_idx}: {stage_label}")
    log.info(
        f"  ctx={args.context_length}, horizon={args.prediction_length}, "
        f"min_len={min_len}, bins={args.n_bins}"
    )
    log.info(
        f"  epochs={epochs}, batch={batch_size}, num_samples={num_samples}, "
        f"head_lr={args.head_lr:.2e}"
    )
    if prev_checkpoint is None:
        log.info("  init=from_scratch")
    else:
        log.info(f"  init={prev_checkpoint}")

    pool_series, stage_stats = load_pool_series(
        preset["datasets"], min_len=min_len
    )
    if not pool_series:
        raise ValueError(
            f"Stage '{stage_label}' has 0 valid series with min_len={min_len}. "
            "Lower context/horizon or change the dataset pool."
        )

    set_all_seeds(args.seed)
    log.info(f"  seed={args.seed}")
    train_series, val_series, test_series = split_three_way(
        pool_series, args.val_ratio, args.test_ratio
    )

    stage_config = {
        "model_id": args.model_id,
        "context_length": args.context_length,
        "prediction_length": args.prediction_length,
        "n_bins": args.n_bins,
        "bin_range": list(args.bin_range),
        "epochs": epochs,
        "batch_size": batch_size,
        "num_samples": num_samples,
        "head_lr": args.head_lr,
        "seed": args.seed,
        "prev_checkpoint": prev_checkpoint,
    }
    save_stage_metadata(
        stage_dir,
        stage_label,
        stage_stats,
        train_count=len(train_series),
        val_count=len(val_series),
        test_count=len(test_series),
        config=stage_config,
    )

    if args.dry_run:
        log.info(f"Dry run only; skipping training for stage '{stage_label}'.")
        return str(stage_dir)

    ckpt_dir, _, _, _ = run_finetune_ce(
        model_id=args.model_id,
        train_series=train_series,
        val_series=val_series,
        output_dir=str(stage_dir),
        context_len=args.context_length,
        horizon_len=args.prediction_length,
        epochs=epochs,
        batch_size=batch_size,
        n_bins=args.n_bins,
        bin_low=args.bin_range[0],
        bin_high=args.bin_range[1],
        mode="frozen",
        head_lr=args.head_lr,
        num_samples=num_samples,
        num_workers=args.num_workers,
        seed=args.seed,
        ce_head_checkpoint=prev_checkpoint,
    )
    mark_stage_complete(stage_dir, stage_label, ckpt_dir)
    return ckpt_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage-wise TimesFM CE head pretraining from scratch"
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(STAGE_PRESETS.keys()),
        default=["large"],
        help="Pretraining stages to run in order. `large` now auto-bootstraps "
             "from starter if no init checkpoint is provided.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(OUTPUT_ROOT),
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/timesfm-2.5-200m-pytorch",
    )
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--n-bins", type=int, default=64)
    parser.add_argument("--bin-range", type=float, nargs=2, default=[-10.0, 10.0])
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--initial-ce-head-checkpoint",
        type=str,
        default=None,
        help="Optional starting checkpoint. If omitted, stage 1 starts from scratch.",
    )
    parser.add_argument(
        "--reuse-checkpoints",
        action="store_true",
        help="Reuse stage output dirs that already contain ce_head.pt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load pools and write stage plans, but do not start training",
    )

    for stage_name in STAGE_PRESETS:
        parser.add_argument(
            f"--{stage_name}-epochs",
            type=int,
            default=int(STAGE_PRESETS[stage_name]["epochs"]),
        )
        parser.add_argument(
            f"--{stage_name}-batch-size",
            type=int,
            default=int(STAGE_PRESETS[stage_name]["batch_size"]),
        )
        parser.add_argument(
            f"--{stage_name}-num-samples",
            type=int,
            default=int(STAGE_PRESETS[stage_name]["num_samples"]),
        )

    args = parser.parse_args()
    assert args.context_length % 32 == 0, "context_length must be multiple of 32"
    set_all_seeds(args.seed)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    log.info("TimesFM CE staged pretraining")
    log.info(f"  stages={' -> '.join(args.stages)}")
    log.info(f"  output_root={output_root}")

    prev_checkpoint = args.initial_ce_head_checkpoint
    stage_records: List[Dict[str, object]] = []

    for stage_idx, stage_name in enumerate(args.stages, start=1):
        bootstrap_dir = None
        prev_checkpoint, bootstrap_dir = maybe_bootstrap_large_stage(
            stage_name=stage_name,
            stage_idx=stage_idx,
            output_root=output_root,
            prev_checkpoint=prev_checkpoint,
            args=args,
        )
        ckpt_dir = run_stage(
            stage_name=stage_name,
            stage_idx=stage_idx,
            output_root=output_root,
            prev_checkpoint=prev_checkpoint,
            args=args,
        )
        record = {
            "stage": stage_name,
            "checkpoint_dir": ckpt_dir,
        }
        if bootstrap_dir is not None:
            record["bootstrap_checkpoint_dir"] = str(bootstrap_dir)
            record["bootstrap_stage"] = "starter"
        stage_records.append(record)
        prev_checkpoint = ckpt_dir

    with open(output_root / "staged_pretrain_summary.json", "w") as f:
        json.dump(stage_records, f, indent=2)
    log.info(f"Summary saved to {output_root / 'staged_pretrain_summary.json'}")


if __name__ == "__main__":
    main()
