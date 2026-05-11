import json
import logging
import math
import re
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from ts_benchmark.common.constant import ROOT_PATH

plt.switch_backend("agg")

MIN_METRIC_KEYS = (
    "loss",
    "error",
    "err",
    "mae",
    "mse",
    "rmse",
    "mape",
    "wape",
    "wer",
    "per",
)
MAX_METRIC_KEYS = (
    "acc",
    "accuracy",
    "f1",
    "precision",
    "recall",
    "auc",
    "auroc",
    "iou",
    "dice",
    "bleu",
    "rouge",
    "spearman",
    "kendall",
    "tau",
    "r2",
    "ndcg",
    "map",
)


@dataclass
class ExperimentPaths:
    project_root: Path
    exp_dir: Path

    @property
    def config_path(self) -> Path:
        return self.exp_dir / "config.json"

    @property
    def summary_all_runs_path(self) -> Path:
        return self.exp_dir / "summary_all_runs.json"

    @property
    def exp_runlog_path(self) -> Path:
        return self.exp_dir / "run.log"

    @property
    def project_runlog_path(self) -> Path:
        return self.project_root / "run.log"

    def detail_log_path(self, run_id: int) -> Path:
        return self.exp_dir / "details" / f"run{run_id}.log"

    def history_path(self, run_id: int) -> Path:
        return self.exp_dir / f"history_run{run_id}.json"

    def curve_path(self, run_id: int) -> Path:
        return self.exp_dir / f"training_curve_run{run_id}.png"

    def best_ckpt_path(self, run_id: int) -> Path:
        return self.exp_dir / "checkpoints" / f"best_model_run{run_id}.pt"


def get_cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def set_cfg_value(cfg: Any, key: str, value: Any) -> None:
    if isinstance(cfg, dict):
        cfg[key] = value
    else:
        setattr(cfg, key, value)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text.isdigit():
        return bool(int(text))
    return text in {"true", "yes", "on", "y"}


def to_serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return to_serializable(value.item())
    if isinstance(value, np.ndarray):
        return [to_serializable(v) for v in value.tolist()]
    if torch.is_tensor(value):
        if value.numel() == 1:
            return to_serializable(value.detach().cpu().item())
        return to_serializable(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(v) for v in value]
    if hasattr(value, "__dict__"):
        return {
            str(k): to_serializable(v)
            for k, v in vars(value).items()
            if not k.startswith("_")
        }
    return str(value)


def sanitize_component(value: Any, default: str) -> str:
    text = str(value or default)
    if text.endswith(".csv"):
        text = Path(text).stem
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or default


def create_experiment_paths(dataset: str, model: str, timestamp: Optional[str] = None) -> ExperimentPaths:
    dataset_name = sanitize_component(dataset, "unknown_dataset")
    model_name = sanitize_component(model, "unknown_model")
    timestamp = timestamp or datetime.now().strftime("%m%d_%H%M")
    exp_dir = Path(ROOT_PATH) / "results" / dataset_name / model_name / timestamp
    paths = ExperimentPaths(project_root=Path(ROOT_PATH), exp_dir=exp_dir)
    (exp_dir / "details").mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return paths


def build_logger(name: str, log_path: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def close_logger(logger: Optional[logging.Logger]) -> None:
    if logger is None:
        return
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_serializable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def append_summary(path: Path, record: Dict[str, Any]) -> None:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = []
    else:
        data = []
    if not isinstance(data, list):
        data = []
    data.append(to_serializable(record))
    write_json(path, data)


def update_summary_record(path: Path, run_id: int, updates: Dict[str, Any]) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return
    for item in reversed(data):
        if item.get("run_id") == run_id:
            item.update(to_serializable(updates))
            break
    write_json(path, data)


def infer_metric_direction(metric_name: str) -> Optional[str]:
    name = metric_name.lower()
    if any(key in name for key in MIN_METRIC_KEYS):
        return "min"
    if any(key in name for key in MAX_METRIC_KEYS):
        return "max"
    return None


def is_improved(current: float, best: Optional[float], direction: str, min_delta: float = 0.0) -> bool:
    if best is None:
        return True
    if direction == "max":
        return current > best + min_delta
    return current < best - min_delta


def compute_grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        param_norm = parameter.grad.detach().data.norm(2).item()
        total += param_norm ** 2
    return total ** 0.5


def get_gpu_memory_gb(device: torch.device) -> Optional[float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.max_memory_allocated(index) / (1024 ** 3)


def reset_gpu_peak_memory(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(index)


def format_seconds(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(value) < 1e-4 and value != 0:
        return f"{value:.6e}"
    return f"{value:.6f}"


def runtime_env(device: torch.device, seed: int) -> Dict[str, Any]:
    if device.type == "cuda" and torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        gpu = "MPS"
    else:
        gpu = "CPU only"
    return {
        "gpu": gpu,
        "seed": seed,
        "pytorch": torch.__version__,
        "python": sys.version.split()[0],
    }


def log_runtime_env(logger: logging.Logger, env: Dict[str, Any]) -> None:
    logger.info("========== Runtime Env ==========")
    logger.info(f"GPU          : {env['gpu']}")
    logger.info(f"Random Seed  : {env['seed']}")
    logger.info(f"PyTorch      : {env['pytorch']}")
    logger.info(f"Python       : {env['python']}")
    logger.info("================================")


def log_model_info(
    logger: logging.Logger,
    total_params: int,
    trainable_params: int,
    model_name: str,
    device: torch.device,
) -> None:
    logger.info("========== Model Info ==========")
    logger.info(f"Total Params   : {total_params / 1_000_000:.3f} M")
    logger.info(f"Trainable      : {trainable_params / 1_000_000:.3f} M")
    logger.info("FLOPs          : N/A")
    logger.info(f"Model          : {model_name}")
    logger.info(f"Run in         : {device}")
    logger.info("================================")


def log_dataset_info(
    logger: logging.Logger,
    train_points: int,
    valid_points: int,
    train_windows: int,
    valid_windows: int,
    train_batches: int,
    valid_batches: int,
    first_batch_sec: Optional[float],
) -> None:
    loader_time = "N/A" if first_batch_sec is None else f"{first_batch_sec:.3f}s (first batch)"
    logger.info("========== Dataset Info ==========")
    logger.info(f"Train samples : {train_points}")
    logger.info(f"Val samples   : {valid_points}")
    logger.info("Test samples  : N/A")
    logger.info(f"Train windows : {train_windows}")
    logger.info(f"Val windows   : {valid_windows}")
    logger.info(f"Train batches : {train_batches}")
    logger.info(f"Val batches   : {valid_batches}")
    logger.info(f"DataLoader    : {loader_time}")
    logger.info("=================================")


def save_training_curve(history: List[Dict[str, Any]], path: Path, primary_metric: str) -> None:
    if not history:
        return

    epochs = [item["epoch"] for item in history]
    panels = []

    train_loss = [get_nested(item, ("train", "loss")) for item in history]
    val_loss = [get_nested(item, ("val", "loss")) for item in history]
    if any(v is not None for v in train_loss + val_loss):
        panels.append(("Loss", [("train", train_loss), ("val", val_loss)]))

    if primary_metric != "loss":
        train_metric = [get_nested(item, ("train", primary_metric)) for item in history]
        val_metric = [get_nested(item, ("val", primary_metric)) for item in history]
        if any(v is not None for v in train_metric + val_metric):
            panels.append((primary_metric, [("train", train_metric), ("val", val_metric)]))

    lr = [item.get("lr") for item in history]
    if any(v is not None for v in lr):
        panels.append(("Learning Rate", [("lr", lr)]))

    grad_norm = [item.get("grad_norm") for item in history]
    if any(v is not None for v in grad_norm):
        panels.append(("Gradient Norm", [("grad_norm", grad_norm)]))

    gpu_mem = [item.get("gpu_mem_gb") for item in history]
    if any(v is not None for v in gpu_mem):
        panels.append(("GPU Memory GB", [("gpu_mem", gpu_mem)]))

    if not panels:
        return

    cols = 2
    rows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.6 * rows))
    axes_array = np.array(axes).reshape(-1)

    for axis, (title, series_list) in zip(axes_array, panels):
        for label, values in series_list:
            y_values = [np.nan if value is None else float(value) for value in values]
            axis.plot(epochs, y_values, marker="o", label=label)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
        axis.legend()

    for axis in axes_array[len(panels):]:
        axis.axis("off")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def get_nested(item: Dict[str, Any], keys: Iterable[str]) -> Any:
    cur = item
    for key in keys:
        if cur is None or not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def append_aggregate_runlog(
    paths: ExperimentPaths,
    dataset: str,
    model: str,
    summaries: List[Dict[str, Any]],
    metric_names: List[str],
) -> None:
    if not summaries:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"[{timestamp}] ******************** Experiment Results ********************",
        f"[{timestamp}] Experiment Detail: dataset={dataset}   model={model}",
        f"[{timestamp}] ------------------------------------------------------------",
    ]

    aggregate_keys = list(dict.fromkeys(metric_names + ["best_epoch", "train_time_sec"]))
    for key in aggregate_keys:
        values = []
        for summary in summaries:
            value = summary.get(key)
            if value is None and key in summary.get("best", {}):
                value = summary["best"][key]
            if value is None:
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        if not values:
            continue
        mean = statistics.fmean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        lines.append(f"[{timestamp}] {key:<14}: {mean:.6f} +/- {std:.6f}")
    lines.append("")

    text = "\n".join(lines)
    for path in (paths.exp_runlog_path, paths.project_runlog_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(text)
