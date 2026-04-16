# -*- coding: utf-8 -*-
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from ts_benchmark.recording import read_record_file


DATASET_ORDER = [
    ("Genesis", "Genesis"),
    ("CICIDS", "CICIDS"),
    ("Creditcard", "Credit"),
    ("MSL", "MSL"),
    ("PSM", "PSM"),
    ("SMAP", "SMAP"),
    ("CalIt2", "CalIt2"),
]

MODEL_ORDER = ["CATCH", "DUET", "ATrans", "AE", "PCA", "Ocsvm", "IF", "HBOS", "CACAM"]

MODEL_NAME_MAP = {
    "CATCH": "CATCH",
    "DUET": "DUET",
    "AnomalyTransformer": "ATrans",
    "AutoEncoder": "AE",
    "pcaodetectorski": "PCA",
    "ocsvmski": "Ocsvm",
    "IsolationForest": "IF",
    "hbosski": "HBOS",
    "CACAM": "CACAM",
}


def iter_result_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.endswith("_report.csv"):
            continue
        if path.suffix == ".csv" or path.name.endswith(".tar.gz"):
            yield path


def load_metric_records(root: Path, metric_name: str) -> pd.DataFrame:
    records = []
    for file_path in iter_result_files(root):
        try:
            record_df = read_record_file(str(file_path))
        except Exception:
            continue

        required_columns = {"model_name", "file_name", metric_name}
        if not required_columns.issubset(record_df.columns):
            continue

        current_df = record_df.loc[:, [column for column in record_df.columns if column in {
            "model_name",
            "file_name",
            "typical_anomaly_ratio",
            metric_name,
        }]].copy()
        current_df["source_path"] = str(file_path)
        current_df["source_mtime"] = file_path.stat().st_mtime
        current_df["dataset_key"] = current_df["file_name"].map(
            lambda value: Path(str(value)).stem
        )
        current_df["model_display"] = current_df["model_name"].map(MODEL_NAME_MAP)
        current_df = current_df[current_df["model_display"].notna()]
        records.append(current_df)

    if not records:
        return pd.DataFrame()

    return pd.concat(records, axis=0, ignore_index=True)


def pick_latest_metric_values(metric_df: pd.DataFrame, metric_name: str) -> dict:
    if metric_df.empty:
        return {}

    latest_values = {}
    sorted_df = metric_df.sort_values(["source_mtime", "source_path", "file_name"])
    for (dataset_key, model_display), group in sorted_df.groupby(
        ["dataset_key", "model_display"], dropna=False
    ):
        latest_group = group[group["source_mtime"] == group["source_mtime"].max()]
        if "typical_anomaly_ratio" in latest_group.columns:
            preferred_group = latest_group[latest_group["typical_anomaly_ratio"] == 1.0]
            if preferred_group.empty:
                preferred_group = latest_group
        else:
            preferred_group = latest_group
        latest_row = preferred_group.iloc[-1]
        latest_values[(dataset_key, model_display)] = latest_row[metric_name]
    return latest_values


def build_table(label_root: Path, score_root: Path) -> pd.DataFrame:
    label_values = pick_latest_metric_values(
        load_metric_records(label_root, "affiliation_f"),
        "affiliation_f",
    )
    score_values = pick_latest_metric_values(
        load_metric_records(score_root, "auc_roc"),
        "auc_roc",
    )

    rows = []
    for dataset_key, dataset_display in DATASET_ORDER:
        aff_row = {"Dataset": dataset_display, "Metric": "Aff-F"}
        auc_row = {"Dataset": "", "Metric": "A-R"}
        for model_name in MODEL_ORDER:
            aff_value = label_values.get((dataset_key, model_name))
            auc_value = score_values.get((dataset_key, model_name))
            aff_row[model_name] = "" if pd.isna(aff_value) else round(float(aff_value), 4)
            auc_row[model_name] = "" if pd.isna(auc_value) else round(float(auc_value), 4)
        rows.extend([aff_row, auc_row])
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label-root",
        default=str(ROOT_DIR / "result" / "label"),
    )
    parser.add_argument(
        "--score-root",
        default=str(ROOT_DIR / "result" / "score"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT_DIR / "result" / "figure_table.csv"),
    )
    args = parser.parse_args()

    table_df = build_table(Path(args.label_root), Path(args.score_root))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()
