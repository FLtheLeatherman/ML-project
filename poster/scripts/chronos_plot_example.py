"""
Poster-oriented Chronos base-model prediction example.

This module generates one clean forecast figure for `amazon/chronos-t5-base`
on the tourism_monthly holdout split.  It is intentionally inference-only:
no fine-tuning or checkpoint creation is performed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FREQ_MAP = {
    "hourly": "H",
    "daily": "D",
    "weekly": "W",
    "monthly": "M",
    "quarterly": "Q",
    "yearly": "Y",
}


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def parse_tsf(file_path: str) -> tuple[list[np.ndarray], Optional[str]]:
    """Parse a Monash .tsf file into series arrays and a frequency label."""
    series_list: list[np.ndarray] = []
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
            values: list[float] = []
            for raw_value in values_str.split(sep):
                raw_value = raw_value.strip()
                if raw_value in ("?", ""):
                    values.append(float("nan"))
                    continue
                try:
                    values.append(float(raw_value))
                except ValueError:
                    values.append(float("nan"))
            if values:
                series_list.append(np.asarray(values, dtype=np.float32))
    return series_list, frequency


def clean_series(series: np.ndarray) -> np.ndarray:
    arr = np.asarray(series, dtype=np.float32)
    if not np.isnan(arr).any():
        return arr
    return pd.Series(arr).interpolate(method="linear").ffill().bfill().values.astype(
        np.float32
    )


def select_eval_series(
    tsf_path: Path,
    eval_holdout_ratio: float,
    sample_index: int,
    prediction_length: int,
) -> tuple[np.ndarray, Optional[str], int]:
    series_list, freq = parse_tsf(str(tsf_path))
    if not series_list:
        raise ValueError(f"No series found in {tsf_path}")

    if eval_holdout_ratio > 0:
        n_eval = max(1, int(len(series_list) * eval_holdout_ratio))
        candidates = series_list[-n_eval:]
    else:
        candidates = series_list

    valid = [
        clean_series(series)
        for series in candidates
        if len(series) >= prediction_length * 2
    ]
    if not valid:
        raise ValueError(
            f"No series long enough for prediction_length={prediction_length}"
        )

    idx = sample_index % len(valid)
    return valid[idx], freq, idx


def make_prediction(
    model_id: str,
    series: np.ndarray,
    prediction_length: int,
    num_samples: int,
    seed: int,
    local_files_only: bool = False,
):
    from chronos import ChronosPipeline

    set_all_seeds(seed)
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    pipeline = ChronosPipeline.from_pretrained(
        model_id,
        device_map=DEVICE,
        dtype=torch.float32,
        local_files_only=local_files_only,
    )
    pipeline.model.to(DEVICE)

    context = np.asarray(series[:-prediction_length], dtype=np.float32)
    actual = np.asarray(series[-prediction_length:], dtype=np.float32)
    quantiles, mean = pipeline.predict_quantiles(
        [torch.tensor(context, dtype=torch.float32)],
        prediction_length=prediction_length,
        quantile_levels=QUANTILE_LEVELS,
        num_samples=num_samples,
    )
    return context, actual, mean.numpy()[0], quantiles.numpy()[0]


def save_prediction_npz(
    path: Path,
    context: np.ndarray,
    actual: np.ndarray,
    point: np.ndarray,
    quantiles: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        contexts=np.array([context], dtype=object),
        actuals=np.array([actual], dtype=object),
        point_forecasts=np.array([point], dtype=object),
        quantile_forecasts=np.array([quantiles], dtype=object),
    )


def load_prediction_npz(path: Path, sample_index: int = 0):
    payload = np.load(path, allow_pickle=True)
    n = len(payload["contexts"])
    idx = sample_index % n
    context = np.asarray(payload["contexts"][idx], dtype=np.float32)
    actual = np.asarray(payload["actuals"][idx], dtype=np.float32)
    point = np.asarray(payload["point_forecasts"][idx], dtype=np.float32)
    quantiles = np.asarray(payload["quantile_forecasts"][idx], dtype=np.float32)
    return context, actual, point, quantiles


def plot_prediction(
    context: np.ndarray,
    actual: np.ndarray,
    point: np.ndarray,
    quantiles: np.ndarray,
    output_path: Path,
    title: str,
    context_tail: int,
    dpi: int,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ml-project")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    context_tail = min(context_tail, len(context))
    ctx = context[-context_tail:]
    x_ctx = np.arange(-context_tail, 0)
    x_pred = np.arange(len(point))

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.plot(x_ctx, ctx, color="#334155", linewidth=2.2, label="context")
    ax.plot(
        [-1, 0],
        [ctx[-1], actual[0]],
        color="#111827",
        linewidth=2.6,
        label="_nolegend_",
    )
    ax.plot(x_pred, actual, color="#111827", linewidth=2.6, label="actual")
    ax.plot(x_pred, point, color="#dc2626", linewidth=2.6, label="prediction")

    if quantiles.ndim == 2 and quantiles.shape[1] >= 9:
        ax.fill_between(
            x_pred,
            quantiles[:, 0],
            quantiles[:, -1],
            color="#fecaca",
            alpha=0.48,
            linewidth=0,
            label="q10-q90",
        )

    ax.axvline(-0.5, color="#94a3b8", linestyle="--", linewidth=1.4)
    ax.text(
        -0.4,
        ax.get_ylim()[1],
        " forecast start",
        color="#64748b",
        fontsize=10,
        va="top",
    )
    if title:
        ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("Time index relative to forecast start", fontsize=11)
    ax.set_ylabel("Value", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", ncol=3, fontsize=9, frameon=False)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a poster-ready Chronos base-model prediction example."
    )
    parser.add_argument(
        "--tsf-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "extracted" / "tourism_monthly_dataset.tsf",
    )
    parser.add_argument("--model-id", default="amazon/chronos-t5-base")
    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--eval-holdout-ratio", type=float, default=0.2)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only local Hugging Face cache; avoids remote HEAD checks.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-tail", type=int, default=60)
    parser.add_argument("--title", default="", help="Optional plot title.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "poster" / "figures" / "chronos_base_prediction_example.png",
    )
    parser.add_argument(
        "--save-npz",
        type=Path,
        default=None,
        help="Optional path to cache the generated prediction arrays.",
    )
    parser.add_argument(
        "--from-npz",
        type=Path,
        default=None,
        help="Plot an existing predictions.npz instead of running Chronos inference.",
    )
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    if args.from_npz is not None:
        context, actual, point, quantiles = load_prediction_npz(
            args.from_npz, sample_index=args.sample_index
        )
        freq = None
        eval_idx = args.sample_index
    else:
        series, freq, eval_idx = select_eval_series(
            args.tsf_path,
            eval_holdout_ratio=args.eval_holdout_ratio,
            sample_index=args.sample_index,
            prediction_length=args.prediction_length,
        )
        context, actual, point, quantiles = make_prediction(
            args.model_id,
            series,
            prediction_length=args.prediction_length,
            num_samples=args.num_samples,
            seed=args.seed,
            local_files_only=args.local_files_only,
        )
        if args.save_npz is not None:
            save_prediction_npz(args.save_npz, context, actual, point, quantiles)

    title = args.title
    plot_prediction(
        context=context,
        actual=actual,
        point=point,
        quantiles=quantiles,
        output_path=args.output,
        title=title,
        context_tail=args.context_tail,
        dpi=args.dpi,
    )

    summary = {
        "output": str(args.output),
        "sample_index": int(eval_idx),
        "prediction_length": int(len(point)),
        "context_length_plotted": int(min(args.context_tail, len(context))),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
