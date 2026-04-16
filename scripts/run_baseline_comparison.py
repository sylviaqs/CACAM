import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path


DATASETS = [
    "CalIt2",
    "NYC",
    "CICIDS",
    "Creditcard",
    "MSL",
    "PSM",
    "SMAP",
]

MODELS = {
    "CACAM": "CACAM.sh",
    "AutoEncoder": "AutoEncoder.sh",
    "DLinear": "DLinear.sh",
    "NLinear": "NLinear.sh",
    "TimesNet": "TimesNet.sh",
    "IsolationForest": "IsolationForest.sh",
    "PCA": "pcaodetectorski.sh",
    "HBOS": "hbosski.sh",
    "OCSVM": "ocsvmski.sh",
}

CONFIG_PATH = "unfixed_detect_compare_config.json"
SAVE_ROOT = "compare_10x10"
EPOCH_CAP_MODELS = {"CACAM", "DLinear", "NLinear", "TimesNet"}


def replace_option(args, option, value):
    if option in args:
        args[args.index(option) + 1] = value
    else:
        args.extend([option, value])


def cap_training_cost(args, model):
    if "--model-hyper-params" not in args:
        return
    idx = args.index("--model-hyper-params") + 1
    try:
        params = json.loads(args[idx])
    except json.JSONDecodeError:
        return
    if model == "CACAM" and not params:
        params["train_epochs"] = 3
    if model in EPOCH_CAP_MODELS:
        cap = 3
        if "num_epochs" in params:
            params["num_epochs"] = min(int(params["num_epochs"]), cap)
        if "train_epochs" in params:
            params["train_epochs"] = min(int(params["train_epochs"]), cap)
    args[idx] = json.dumps(params)


def find_script(dataset, script_name):
    exact = (
        Path("scripts")
        / "multivariate_detection"
        / "detect_label"
        / f"{dataset}_script"
        / script_name
    )
    if exact.exists():
        return exact
    matches = sorted(
        (Path("scripts") / "multivariate_detection" / "detect_label").glob(
            f"*_script/{script_name}"
        )
    )
    if not matches:
        raise FileNotFoundError(exact)
    return matches[0]


def has_report(dataset, model):
    model_dir = Path("result") / SAVE_ROOT / model
    if not model_dir.exists():
        return False
    return any(model_dir.glob(f"*_{dataset}_*_report.csv"))


def command_from_script(dataset, model, script_name):
    script_path = find_script(dataset, script_name)
    line = script_path.read_text().strip().splitlines()[0]
    args = shlex.split(line)
    args[0] = "/opt/anaconda3/envs/cacam/bin/python"
    replace_option(args, "--config-path", CONFIG_PATH)
    replace_option(args, "--data-name-list", f"{dataset}.csv")
    replace_option(args, "--save-path", f"{SAVE_ROOT}/{model}")
    replace_option(args, "--timeout", "60000")
    cap_training_cost(args, model)
    return args


def main():
    failures = []
    total = len(DATASETS) * len(MODELS)
    index = 0

    for dataset in DATASETS:
        for model, script_name in MODELS.items():
            index += 1
            print(f"[{index}/{total}] {dataset} / {model}", flush=True)
            try:
                if has_report(dataset, model):
                    print(f"SKIP existing {dataset} / {model}", flush=True)
                    continue
                args = command_from_script(dataset, model, script_name)
                subprocess.run(args, check=True)
            except Exception as exc:
                print(f"FAILED {dataset} / {model}: {exc}", flush=True)
                failures.append(
                    {"dataset": dataset, "model": model, "error": repr(exc)}
                )

    failure_path = Path("result") / SAVE_ROOT / "failures.csv"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    with failure_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "model", "error"])
        writer.writeheader()
        writer.writerows(failures)

    if failures:
        print(f"{len(failures)} failures written to {failure_path}", flush=True)
        return 1
    print("All runs completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
