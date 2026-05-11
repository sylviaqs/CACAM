import copy
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from ts_benchmark.baselines.device import get_torch_device
from ts_benchmark.baselines.CACAM.models import (
    FeatureTemporalTransformerModel,
    GrangerFeatureTemporalTransformerModel,
    CACAMModel,
)
from ts_benchmark.baselines.utils import anomaly_detection_data_provider
from ts_benchmark.baselines.utils import train_val_split
from ts_benchmark.baselines.catch.utils.observability import (
    append_aggregate_runlog,
    append_summary,
    as_bool,
    build_logger,
    close_logger,
    compute_grad_norm,
    create_experiment_paths,
    format_metric,
    format_seconds,
    get_cfg_value,
    get_gpu_memory_gb,
    infer_metric_direction,
    is_improved,
    log_dataset_info,
    log_model_info,
    log_runtime_env,
    reset_gpu_peak_memory,
    runtime_env,
    save_training_curve,
    set_cfg_value,
    to_serializable,
    update_summary_record,
    write_json,
)
from ts_benchmark.utils.random_utils import fix_all_random_seed


DEFAULT_CACAM_HYPER_PARAMS = {
    "lr": 0.0005,
    "e_layers": 2,
    "n_heads": 4,
    "d_model": 64,
    "d_ff": 128,
    "dropout": 0.1,
    "num_epochs": 5,
    "batch_size": 128,
    "patience": 3,
    "early_stopping": True,
    "min_delta": 0.0,
    "print_freq": 1,
    "num_runs": 1,
    "debug": False,
    "debug_num_runs": 1,
    "debug_max_train_batches": 2,
    "debug_max_val_batches": 2,
    "seq_len": 192,
    "model_variant": "CACAM",
    "feature_layers": 1,
    "temporal_layers": 2,
    "granger_mask_type": "soft",
    "granger_bias": 2.0,
    "granger_graph_path": None,
    "experiment_model_name": None,
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
}


CACAM_MODEL_VARIANTS = {
    "CACAM": CACAMModel,
    "feature_temporal_transformer": FeatureTemporalTransformerModel,
    "granger_feature_temporal_transformer": GrangerFeatureTemporalTransformerModel,
}


class CACAMConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_CACAM_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)


class CACAM:
    def __init__(self, **kwargs):
        self.config = CACAMConfig(**kwargs)
        self.scaler = StandardScaler()
        self.device = get_torch_device()
        self.criterion = nn.MSELoss()
        self.score_criterion = nn.MSELoss(reduction="none")
        self.model = None
        self.model_name = "CACAM"
        self.experiment_context = {}
        self.experiment_paths = None
        self.experiment_timestamp = None
        self.metric_spec = {
            "task_type": "anomaly_detection",
            "metrics": ["loss"],
            "primary_metric": "loss",
            "direction": "min",
        }
        self.best_checkpoint_path = None
        self.best_checkpoint_state = None
        self.best_run_id = None
        self.best_epoch = None
        self.best_metric_value = None
        self.early_stopping = SimpleNamespace(check_point=None, early_stop=False)

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    def __repr__(self) -> str:
        return "CACAM"

    def set_experiment_context(self, **context):
        experiment_model_name = get_cfg_value(self.config, "experiment_model_name", None)
        self.experiment_context.update({k: v for k, v in context.items() if v is not None})
        if experiment_model_name:
            self.model_name = str(experiment_model_name)
            self.experiment_context["model_name"] = self.model_name
        elif context.get("model_name"):
            self.model_name = str(context["model_name"]).split(".")[-1]

    def _normalize_observability_config(self):
        config = self.config
        debug = as_bool(get_cfg_value(config, "debug", False))
        set_cfg_value(config, "debug", debug)

        num_epochs = int(get_cfg_value(config, "num_epochs", get_cfg_value(config, "epochs", 1)))
        set_cfg_value(config, "num_epochs", max(1, num_epochs))
        set_cfg_value(config, "epochs", max(1, num_epochs))

        print_freq = int(get_cfg_value(config, "print_freq", 1))
        set_cfg_value(config, "print_freq", max(1, print_freq))

        num_runs = int(get_cfg_value(config, "num_runs", 1))
        if debug:
            num_runs = max(1, int(get_cfg_value(config, "debug_num_runs", 1)))
            set_cfg_value(config, "num_epochs", 1)
            set_cfg_value(config, "epochs", 1)
            set_cfg_value(config, "print_freq", 1)
        set_cfg_value(config, "num_runs", max(1, num_runs))

        set_cfg_value(config, "early_stopping", as_bool(get_cfg_value(config, "early_stopping", True)))
        set_cfg_value(config, "patience", max(1, int(get_cfg_value(config, "patience", 3))))
        set_cfg_value(config, "min_delta", float(get_cfg_value(config, "min_delta", 0.0)))

    def _prepare_metric_spec(self):
        primary_metric = str(get_cfg_value(self.config, "best_metric", "loss"))
        direction = get_cfg_value(self.config, "metric_direction", None)
        if direction is None:
            direction = infer_metric_direction(primary_metric) or "min"
        self.metric_spec = {
            "task_type": "anomaly_detection",
            "metrics": ["loss"],
            "primary_metric": primary_metric,
            "direction": direction,
        }
        return self.metric_spec

    def _prepare_experiment_paths(self):
        dataset = (
            self.experiment_context.get("series_name")
            or get_cfg_value(self.config, "dataset", "unknown_dataset")
        )
        model = self.experiment_context.get("model_name") or self.model_name
        self.experiment_timestamp = self.experiment_timestamp or time.strftime("%m%d_%H%M")
        self.experiment_paths = create_experiment_paths(dataset, model, self.experiment_timestamp)
        write_json(
            self.experiment_paths.config_path,
            {
                "dataset": dataset,
                "model": model,
                "config": to_serializable(self.config),
                "context": to_serializable(self.experiment_context),
                "metric_spec": self.metric_spec,
            },
        )
        return self.experiment_paths

    @staticmethod
    def _limited_batches(data_loader, max_batches):
        for batch_idx, batch in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            yield batch_idx, batch

    def _load_best_checkpoint(self):
        if self.model is None:
            raise ValueError("Model not trained. Call detect_fit first.")
        if self.best_checkpoint_path is not None:
            try:
                checkpoint = torch.load(
                    self.best_checkpoint_path,
                    map_location=self.device,
                    weights_only=False,
                )
            except TypeError:
                checkpoint = torch.load(self.best_checkpoint_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
        elif self.best_checkpoint_state is not None:
            state_dict = self.best_checkpoint_state
        elif self.early_stopping.check_point is not None:
            state_dict = self.early_stopping.check_point
        else:
            raise ValueError("No trained checkpoint is available.")
        result = self.model.load_state_dict(state_dict, strict=False)
        logger = getattr(self, "_active_logger", None)
        if logger is not None and (result.missing_keys or result.unexpected_keys):
            logger.info(
                f"[Load   ]    missing_keys={result.missing_keys} unexpected_keys={result.unexpected_keys}"
            )

    def _build_model(self):
        variant = str(get_cfg_value(self.config, "model_variant", "CACAM"))
        if variant not in CACAM_MODEL_VARIANTS:
            raise ValueError(
                f"Unknown CACAM model_variant '{variant}'. "
                f"Available variants: {sorted(CACAM_MODEL_VARIANTS)}"
            )
        return CACAM_MODEL_VARIANTS[variant](self.config)

    def detect_hyper_param_tune(self, train_data: pd.DataFrame):
        self.config.c_in = train_data.shape[1]

    def detect_validate(self, valid_data_loader, max_batches=None):
        total_loss = []
        total_samples = 0
        start = time.time()
        self.model.eval()
        with torch.no_grad():
            for _, (input, _) in self._limited_batches(valid_data_loader, max_batches):
                input = input.float().to(self.device)
                total_samples += input.shape[0]
                output = self.model(input)
                loss = self.criterion(output, input).detach().cpu().numpy()
                total_loss.append(loss)
        self.model.train()
        elapsed = time.time() - start
        return {
            "loss": float(np.mean(total_loss)) if total_loss else float("nan"),
            "elapsed_sec": elapsed,
            "samples": total_samples,
            "samples_per_sec": total_samples / elapsed if elapsed > 0 else None,
            "latency_ms_per_sample": elapsed / total_samples * 1000 if total_samples > 0 else None,
        }

    def detect_fit(self, train_data: pd.DataFrame, train_label=None):
        self._normalize_observability_config()
        self.detect_hyper_param_tune(train_data)
        self._prepare_metric_spec()
        paths = self._prepare_experiment_paths()

        config = self.config
        train_data_value, valid_data = train_val_split(train_data, 0.8, None)
        self.scaler.fit(train_data_value.values)

        train_data_value = pd.DataFrame(
            self.scaler.transform(train_data_value.values),
            columns=train_data_value.columns,
            index=train_data_value.index,
        )
        valid_data = pd.DataFrame(
            self.scaler.transform(valid_data.values),
            columns=valid_data.columns,
            index=valid_data.index,
        )

        self.train_data_loader = anomaly_detection_data_provider(
            train_data_value,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="train",
        )
        self.valid_data_loader = anomaly_detection_data_provider(
            valid_data,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="val",
        )

        num_runs = int(get_cfg_value(config, "num_runs", 1))
        base_seed = int(get_cfg_value(config, "seed", self.experiment_context.get("seed", 2021)))
        all_run_summaries = []
        self.best_metric_value = None
        self.best_checkpoint_path = None
        self.best_checkpoint_state = None

        for run_id in range(num_runs):
            run_seed = base_seed + run_id
            fix_all_random_seed(run_seed)
            summary = self._run_single_experiment(
                run_id=run_id,
                run_seed=run_seed,
                train_points=len(train_data_value),
                valid_points=len(valid_data),
                paths=paths,
            )
            append_summary(paths.summary_all_runs_path, summary)
            all_run_summaries.append(summary)

            current_best = summary["best"]["best_metric_value"]
            if is_improved(
                current_best,
                self.best_metric_value,
                self.metric_spec["direction"],
                float(get_cfg_value(config, "min_delta", 0.0)),
            ):
                self.best_metric_value = current_best
                self.best_checkpoint_path = summary["best_checkpoint_path"]
                self.best_checkpoint_state = copy.deepcopy(self.model.state_dict())
                self.best_run_id = run_id
                self.best_epoch = summary["best"]["best_epoch"]

        dataset = self.experiment_context.get("series_name") or get_cfg_value(config, "dataset", "unknown_dataset")
        model = self.experiment_context.get("model_name") or self.model_name
        append_aggregate_runlog(
            paths,
            dataset=dataset,
            model=model,
            summaries=all_run_summaries,
            metric_names=[self.metric_spec["primary_metric"]],
        )

        self.early_stopping = SimpleNamespace(
            check_point=self.best_checkpoint_state,
            early_stop=any(item.get("early_stopped", False) for item in all_run_summaries),
        )
        self._load_best_checkpoint()

    def _run_single_experiment(self, run_id, run_seed, train_points, valid_points, paths):
        config = self.config
        logger = build_logger(f"CACAM.run{run_id}.{id(self)}", paths.detail_log_path(run_id))
        self._active_logger = logger
        history = []
        run_started = time.time()
        early_stopped = False
        patience_counter = 0
        run_best_value = None
        run_best_epoch = None
        run_best_state = None
        run_best_path = paths.best_ckpt_path(run_id)

        try:
            self.model = self._build_model().to(self.device)
            reset_gpu_peak_memory(self.device)
            train_steps = len(self.train_data_loader)
            valid_steps = len(self.valid_data_loader)
            debug = as_bool(get_cfg_value(config, "debug", False))
            max_train_batches = (
                int(get_cfg_value(config, "debug_max_train_batches", 2)) if debug else None
            )
            max_val_batches = (
                int(get_cfg_value(config, "debug_max_val_batches", 2)) if debug else None
            )
            effective_train_steps = min(train_steps, max_train_batches) if max_train_batches else train_steps
            effective_valid_steps = min(valid_steps, max_val_batches) if max_val_batches else valid_steps
            effective_train_steps = max(1, effective_train_steps)

            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            log_runtime_env(logger, runtime_env(self.device, run_seed))
            log_model_info(logger, total_params, trainable_params, self.model_name, self.device)
            first_batch_sec = self._measure_first_batch_time(self.train_data_loader)
            log_dataset_info(
                logger,
                train_points=train_points,
                valid_points=valid_points,
                train_windows=len(self.train_data_loader.dataset),
                valid_windows=len(self.valid_data_loader.dataset),
                train_batches=effective_train_steps,
                valid_batches=effective_valid_steps,
                first_batch_sec=first_batch_sec,
            )

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

            for epoch in range(config.num_epochs):
                epoch_result = self._train_one_epoch(
                    epoch=epoch,
                    run_id=run_id,
                    train_steps=effective_train_steps,
                    max_train_batches=max_train_batches,
                )
                val_metrics = self.detect_validate(
                    self.valid_data_loader,
                    max_batches=max_val_batches,
                )
                current_metric = float(val_metrics["loss"])
                improved = is_improved(
                    current_metric,
                    run_best_value,
                    self.metric_spec["direction"],
                    float(get_cfg_value(config, "min_delta", 0.0)),
                )
                if improved:
                    run_best_value = current_metric
                    run_best_epoch = epoch + 1
                    run_best_state = copy.deepcopy(self.model.state_dict())
                    checkpoint = {
                        "model_state_dict": run_best_state,
                        "epoch": run_best_epoch,
                        "best_metric_name": self.metric_spec["primary_metric"],
                        "best_metric_value": run_best_value,
                        "metric_spec": self.metric_spec,
                        "config": to_serializable(config),
                        "run_id": run_id,
                        "run_seed": run_seed,
                    }
                    torch.save(checkpoint, run_best_path)
                    patience_counter = 0
                else:
                    patience_counter += 1

                lr = self.optimizer.param_groups[0]["lr"]
                gpu_mem = get_gpu_memory_gb(self.device)
                record = {
                    "epoch": epoch + 1,
                    "train": {"loss": epoch_result["loss"]},
                    "val": {"loss": current_metric},
                    "lr": lr,
                    "grad_norm": epoch_result["grad_norm"],
                    "epoch_time_sec": epoch_result["epoch_time_sec"],
                    "train_samples_per_sec": epoch_result["samples_per_sec"],
                    "val_samples_per_sec": val_metrics["samples_per_sec"],
                    "latency_ms_per_sample": val_metrics["latency_ms_per_sample"],
                    "gpu_mem_gb": gpu_mem,
                    "best_metric": run_best_value,
                    "best_epoch": run_best_epoch,
                }
                history.append(record)
                write_json(paths.history_path(run_id), history)

                if (epoch + 1) % int(get_cfg_value(config, "print_freq", 1)) == 0 or improved:
                    self._log_progress_block(
                        logger=logger,
                        run_id=run_id,
                        run_seed=run_seed,
                        epoch=epoch + 1,
                        total_epochs=config.num_epochs,
                        train_metrics=record["train"],
                        val_metrics=record["val"],
                        lr=lr,
                        grad_norm=epoch_result["grad_norm"],
                        train_speed=epoch_result["samples_per_sec"],
                        val_speed=val_metrics["samples_per_sec"],
                        gpu_mem=gpu_mem,
                        patience_counter=patience_counter,
                        best_value=run_best_value,
                        best_epoch=run_best_epoch,
                        improved=improved,
                        checkpoint_path=run_best_path,
                        elapsed=time.time() - run_started,
                    )

                if as_bool(get_cfg_value(config, "early_stopping", True)) and patience_counter >= int(config.patience):
                    early_stopped = True
                    logger.info("Early stopping")
                    break

            if run_best_state is None:
                run_best_state = copy.deepcopy(self.model.state_dict())
                run_best_value = float("nan")
                run_best_epoch = len(history)
                torch.save(
                    {
                        "model_state_dict": run_best_state,
                        "epoch": run_best_epoch,
                        "best_metric_name": self.metric_spec["primary_metric"],
                        "best_metric_value": run_best_value,
                        "metric_spec": self.metric_spec,
                        "config": to_serializable(config),
                        "run_id": run_id,
                        "run_seed": run_seed,
                    },
                    run_best_path,
                )

            self.model.load_state_dict(run_best_state, strict=False)
            save_training_curve(history, paths.curve_path(run_id), self.metric_spec["primary_metric"])
            train_time = time.time() - run_started
            summary = {
                "run_id": run_id,
                "run_seed": run_seed,
                "dataset": self.experiment_context.get("series_name")
                or get_cfg_value(config, "dataset", "unknown_dataset"),
                "model": self.experiment_context.get("model_name") or self.model_name,
                "metric_spec": self.metric_spec,
                "best": {
                    "best_metric_name": self.metric_spec["primary_metric"],
                    "best_metric_value": run_best_value,
                    "best_epoch": run_best_epoch,
                },
                "loss": run_best_value,
                "best_epoch": run_best_epoch,
                "train_time_sec": train_time,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "early_stopped": early_stopped,
                "best_checkpoint_path": str(run_best_path),
                "history_path": str(paths.history_path(run_id)),
                "curve_path": str(paths.curve_path(run_id)),
                "test_results_by_ratio": {},
            }
            self._log_training_summary(logger, summary, history)
            return summary
        finally:
            close_logger(logger)
            self._active_logger = None

    @staticmethod
    def _measure_first_batch_time(data_loader):
        start = time.time()
        try:
            next(iter(data_loader))
        except StopIteration:
            return None
        return time.time() - start

    def _train_one_epoch(self, epoch=0, run_id=0, train_steps=None, max_train_batches=None):
        train_loss = []
        grad_norms = []
        total_samples = 0
        epoch_time = time.time()
        self.model.train()

        batches = self._limited_batches(self.train_data_loader, max_train_batches)
        progress = tqdm(
            batches,
            total=train_steps,
            desc=f"ours run{run_id} epoch {epoch + 1}/{self.config.num_epochs}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        )
        for _, (input, _) in progress:
            self.optimizer.zero_grad()
            input = input.float().to(self.device)
            total_samples += input.shape[0]
            output = self.model(input)
            loss = self.criterion(output, input)
            loss.backward()
            grad_norm = compute_grad_norm(self.model)
            grad_norms.append(grad_norm)
            self.optimizer.step()

            train_loss.append(loss.item())
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{np.average(train_loss):.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
            )

        elapsed = time.time() - epoch_time
        return {
            "loss": float(np.average(train_loss)) if train_loss else float("nan"),
            "grad_norm": float(np.average(grad_norms)) if grad_norms else None,
            "epoch_time_sec": elapsed,
            "samples": total_samples,
            "samples_per_sec": total_samples / elapsed if elapsed > 0 else None,
        }

    def _log_progress_block(
        self,
        logger,
        run_id,
        run_seed,
        epoch,
        total_epochs,
        train_metrics,
        val_metrics,
        lr,
        grad_norm,
        train_speed,
        val_speed,
        gpu_mem,
        patience_counter,
        best_value,
        best_epoch,
        improved,
        checkpoint_path,
        elapsed,
    ):
        dataset = self.experiment_context.get("series_name") or get_cfg_value(
            self.config, "dataset", "unknown_dataset"
        )
        model = self.experiment_context.get("model_name") or self.model_name
        patience_state = (
            f"{patience_counter}/{self.config.patience}"
            if as_bool(get_cfg_value(self.config, "early_stopping", True))
            else "off"
        )
        mem_text = "N/A" if gpu_mem is None else f"{format_metric(gpu_mem)}GB"
        logger.info(
            f"-------------------- Run{run_id}  Epoch {epoch}/{total_epochs} --------------------"
        )
        logger.info(f"  [Info   ]    dataset={dataset}   model={model}   run_seed={run_seed}")
        logger.info(
            "  [Metrics]    "
            + "   ".join(f"{key}={format_metric(value)}" for key, value in train_metrics.items())
            + "   (train)"
        )
        logger.info(
            "  [Metrics]    "
            + "   ".join(f"{key}={format_metric(value)}" for key, value in val_metrics.items())
            + "   (val)"
        )
        logger.info(f"  [Optim  ]    lr={format_metric(lr)}   gnorm={format_metric(grad_norm)}")
        logger.info(
            "  [System ]    "
            f"t_spd={format_metric(train_speed)}/s   "
            f"v_spd={format_metric(val_speed)}/s   "
            f"mem={mem_text}   "
            f"patience={patience_state}   "
            f"elapsed={format_seconds(elapsed)}"
        )
        logger.info(
            f"  [Best   ]    best_{self.metric_spec['primary_metric']}={format_metric(best_value)}   "
            f"best_epoch={best_epoch}"
        )
        if improved:
            logger.info(
                f"  [Saved  ]    best model updated at epoch {epoch}   "
                f"{self.metric_spec['primary_metric']}={format_metric(best_value)}   -> {checkpoint_path}"
            )

    def _log_training_summary(self, logger, summary, history):
        total_time = summary["train_time_sec"]
        avg_train_speed = np.mean(
            [item["train_samples_per_sec"] for item in history if item.get("train_samples_per_sec")]
        ) if history else None
        avg_val_speed = np.mean(
            [item["val_samples_per_sec"] for item in history if item.get("val_samples_per_sec")]
        ) if history else None
        avg_latency = np.mean(
            [item["latency_ms_per_sample"] for item in history if item.get("latency_ms_per_sample")]
        ) if history else None
        peak_mem = max(
            [item["gpu_mem_gb"] for item in history if item.get("gpu_mem_gb") is not None],
            default=None,
        )
        avg_epoch_time = np.mean(
            [item["epoch_time_sec"] for item in history if item.get("epoch_time_sec")]
        ) if history else None

        logger.info("========== Training Summary ==========")
        logger.info(f"Best Epoch     : {summary['best']['best_epoch']}")
        logger.info(
            f"Best {summary['best']['best_metric_name']}  : "
            f"{format_metric(summary['best']['best_metric_value'])}"
        )
        logger.info(f"Total Time     : {format_seconds(total_time)}")
        logger.info(f"Early Stop     : {'Yes' if summary.get('early_stopped') else 'No'}")
        params = summary.get("trainable_params")
        logger.info(
            f"Params         : {params / 1_000_000:.3f} M" if params is not None else "Params         : N/A"
        )
        logger.info("FLOPs          : N/A")
        logger.info(f"Avg Train Spd  : {format_metric(avg_train_speed)} samples/s")
        logger.info(f"Avg Infer Spd  : {format_metric(avg_val_speed)} samples/s")
        logger.info(f"Avg Latency    : {format_metric(avg_latency)} ms/sample")
        peak_mem_text = "N/A" if peak_mem is None else f"{format_metric(peak_mem)} GB"
        logger.info(f"Peak GPU Mem   : {peak_mem_text}")
        logger.info(f"Avg Epoch Time : {format_seconds(avg_epoch_time or 0)}")
        logger.info("======================================")

    def record_evaluation_result(self, ratio, metrics, fit_time_sec, inference_time_sec):
        if self.experiment_paths is None:
            return
        run_id = self.best_run_id if self.best_run_id is not None else 0
        path = self.experiment_paths.summary_all_runs_path
        existing = {}
        if path.exists():
            import json

            data = json.loads(path.read_text(encoding="utf-8"))
            for item in reversed(data):
                if item.get("run_id") == run_id:
                    existing = item.get("test_results_by_ratio", {})
                    break
        ratio_key = str(ratio)
        existing[ratio_key] = {
            "metrics": to_serializable(metrics),
            "fit_time_sec": fit_time_sec,
            "inference_time_sec": inference_time_sec,
        }
        update_summary_record(path, run_id, {"test_results_by_ratio": existing})
        self._log_test_results(
            run_id=run_id,
            ratio=ratio,
            metrics=metrics,
            fit_time_sec=fit_time_sec,
            inference_time_sec=inference_time_sec,
        )

    def _log_test_results(self, run_id, ratio, metrics, fit_time_sec, inference_time_sec):
        logger = build_logger(
            f"CACAM.test.run{run_id}.{id(self)}",
            self.experiment_paths.detail_log_path(run_id),
        )
        try:
            logger.info(f"========== Test Results (Run{run_id}) ==========")
            logger.info(f"Source         : {self.best_checkpoint_path}")
            logger.info(f"Best Epoch     : {self.best_epoch}")
            logger.info(f"Anomaly Ratio  : {ratio}")
            for metric_name, metric_value in metrics.items():
                logger.info(f"{metric_name:<14}: {format_metric(metric_value)}")
            logger.info(f"Fit Time       : {format_seconds(fit_time_sec)}")
            logger.info(f"Inference Time : {format_seconds(inference_time_sec)}")
            logger.info("=================================================")
        finally:
            close_logger(logger)

    def _score_loader(self, loader, max_batches=None):
        scores = []
        self.model.eval()
        with torch.no_grad():
            for _, (batch_x, _) in self._limited_batches(loader, max_batches):
                batch_x = batch_x.float().to(self.device)
                output = self.model(batch_x)
                score = torch.mean(self.score_criterion(batch_x, output), dim=-1)
                scores.append(score.detach().cpu().numpy())
        return np.concatenate(scores, axis=0).reshape(-1)

    def detect_score(self, test: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call detect_fit first.")
        self._load_best_checkpoint()
        max_eval_batches = (
            int(get_cfg_value(self.config, "debug_max_val_batches", 2))
            if as_bool(get_cfg_value(self.config, "debug", False))
            else None
        )

        test = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )
        thre_loader = anomaly_detection_data_provider(
            test,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=1,
            mode="thre",
        )
        test_energy = self._score_loader(thre_loader, max_batches=max_eval_batches)
        return test_energy, test_energy

    def detect_label(self, test: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call detect_fit first.")
        self._load_best_checkpoint()
        max_eval_batches = (
            int(get_cfg_value(self.config, "debug_max_val_batches", 2))
            if as_bool(get_cfg_value(self.config, "debug", False))
            else None
        )

        test = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )
        test_data_loader = anomaly_detection_data_provider(
            test,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=1,
            mode="test",
        )
        thre_loader = anomaly_detection_data_provider(
            test,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=1,
            mode="thre",
        )

        train_energy = self._score_loader(self.train_data_loader, max_batches=max_eval_batches)
        test_energy_for_threshold = self._score_loader(test_data_loader, max_batches=max_eval_batches)
        combined_energy = np.concatenate([train_energy, test_energy_for_threshold], axis=0)
        test_energy = self._score_loader(thre_loader, max_batches=max_eval_batches)

        anomaly_ratio = self.config.anomaly_ratio
        if not isinstance(anomaly_ratio, list):
            anomaly_ratio = [anomaly_ratio]

        preds = {}
        for ratio in anomaly_ratio:
            threshold = np.percentile(combined_energy, 100 - ratio)
            preds[ratio] = (test_energy > threshold).astype(int)
        return preds, test_energy
