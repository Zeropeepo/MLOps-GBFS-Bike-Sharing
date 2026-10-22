"""Train chronological GBFS availability regressors and track them in MLflow.

Run from any working directory with the project's Python environment:

    python src/modeling/train.py

The script expects labels created by ``src/preprocess.py`` and needs at least
two timestamp groups for a chronological holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data/processed/station_features.csv"
POINTER_PATH = ROOT_DIR / "data/processed.dvc"
TARGETS = [
    "num_bikes_available_t_plus_30m",
    "num_docks_available_t_plus_30m",
]
NUMERIC_FEATURES = [
    "capacity",
    "num_bikes_available",
    "num_docks_available",
    "is_installed",
    "is_renting",
    "is_returning",
    "collected_hour_utc",
    "collected_day_of_week_utc",
    "is_weekend_utc",
    "bike_availability_ratio",
    "dock_availability_ratio",
    "previous_num_bikes_available",
    "previous_num_docks_available",
]
CATEGORICAL_FEATURES = ["station_id"]
EXPERIMENT_NAME = "GBFS Bike Availability Forecasting"
RUNS = (
    {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 2},
    {"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 2},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 4},
)
REGISTERED_MODEL_NAME = "GBFSBikeAvailabilityRegressor"
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.compose._column_transformer._RemainderColsList",
    "sklearn.tree._tree.Tree",
]
LOGGER = logging.getLogger("gbfs_train")


@dataclass
class ExperimentResult:
    run_id: str
    model_uri: str
    metrics: dict[str, float]
    parameters: dict[str, int | None]


def git_revision() -> str:
    """Return the current Git commit when available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def git_worktree_dirty() -> bool:
    """Report whether uncommitted changes are present in the repository."""
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT_DIR, text=True
        )
        return bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data(
    data_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load labeled rows and keep timestamps for the chronological split."""
    if not data_path.exists():
        raise SystemExit(f"Dataset tidak ditemukan: {data_path}")

    frame = pd.read_csv(data_path)
    required_columns = {
        *TARGETS,
        "collected_at_utc",
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise SystemExit(f"Kolom dataset tidak lengkap: {', '.join(missing_columns)}")

    for target in TARGETS:
        frame[target] = pd.to_numeric(frame[target], errors="coerce")
    frame["collected_at_utc"] = pd.to_datetime(
        frame["collected_at_utc"], utc=True, errors="coerce"
    )

    labeled = frame.dropna(subset=[*TARGETS, "collected_at_utc"]).copy()
    labeled = labeled.sort_values("collected_at_utc")
    timestamp_groups = labeled["collected_at_utc"].drop_duplicates().sort_values()
    if labeled.empty or len(timestamp_groups) < 2:
        raise SystemExit(
            "Training memerlukan minimal dua waktu observasi berlabel. "
            "Lanjutkan polling GBFS, lalu jalankan kembali src/preprocess.py."
        )

    feature_columns = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    features = labeled[feature_columns].copy()
    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce").astype(
            "float64"
        )
    features["station_id"] = features["station_id"].fillna("unknown").astype(str)

    targets = labeled[TARGETS].astype(float)
    timestamps = labeled["collected_at_utc"]
    return features, targets, timestamps


def chronological_split(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    timestamps: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Hold out the latest timestamp groups without shuffling observations."""
    unique_times = timestamps.drop_duplicates().sort_values().to_list()
    first_test_group = min(
        len(unique_times) - 1,
        max(1, int(np.ceil(len(unique_times) * 0.8))),
    )
    cutoff = unique_times[first_test_group]
    train_mask = timestamps < cutoff
    test_mask = timestamps >= cutoff
    if not train_mask.any() or not test_mask.any():
        raise SystemExit("Pembagian kronologis gagal membentuk train dan test set.")

    return (
        features.loc[train_mask],
        features.loc[test_mask],
        targets.loc[train_mask],
        targets.loc[test_mask],
        timestamps.loc[train_mask],
        timestamps.loc[test_mask],
    )


def build_model(parameters: dict[str, int | None]) -> Pipeline:
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    regressor = RandomForestRegressor(
        **parameters,
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocessing", preprocessing), ("regressor", regressor)])


def calculate_metrics(
    targets: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict[str, float]:
    metrics = {}
    target_names = ("bikes", "docks")
    for target, name in zip(TARGETS, target_names):
        metrics[f"mae_{name}"] = float(
            mean_absolute_error(targets[target], predictions[target])
        )
        metrics[f"rmse_{name}"] = float(
            np.sqrt(mean_squared_error(targets[target], predictions[target]))
        )
        metrics[f"r2_{name}"] = float(r2_score(targets[target], predictions[target]))
    metrics["mean_rmse"] = float(
        np.mean([metrics["rmse_bikes"], metrics["rmse_docks"]])
    )
    return metrics


def run_experiment(
    index: int,
    parameters: dict[str, int | None],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    train_times: pd.Series,
    test_times: pd.Series,
    tags: dict[str, str],
) -> ExperimentResult:
    with mlflow.start_run(run_name=f"random-forest-{index}") as run:
        model = build_model(parameters)
        model.fit(X_train, y_train)
        predictions = pd.DataFrame(
            model.predict(X_test), columns=TARGETS, index=X_test.index
        )
        metrics = calculate_metrics(y_test, predictions)

        mlflow.log_params(
            {
                **parameters,
                "model": "RandomForestRegressor",
                "random_state": 42,
                "split_strategy": "chronological_80_20_by_capture_time",
                "targets": ",".join(TARGETS),
                "feature_count": X_train.shape[1],
                "train_rows": len(y_train),
                "test_rows": len(y_test),
                "train_end_utc": train_times.max().isoformat(),
                "test_start_utc": test_times.min().isoformat(),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags(tags)

        signature = infer_signature(X_test, predictions)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="gbfs-bike-availability-regressor",
            signature=signature,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

    LOGGER.info(
        "run=%s bikes_rmse=%.4f docks_rmse=%.4f mean_rmse=%.4f",
        run.info.run_id,
        metrics["rmse_bikes"],
        metrics["rmse_docks"],
        metrics["mean_rmse"],
    )
    return ExperimentResult(
        run_id=run.info.run_id,
        model_uri=model_info.model_uri,
        metrics=metrics,
        parameters=parameters,
    )


def write_results_report(
    experiment: str,
    revision: str,
    worktree_dirty: bool,
    source_hash: str,
    data_hash: str,
    pointer_hash: str,
    train_rows: int,
    test_rows: int,
    results: list[ExperimentResult],
    best_result: ExperimentResult,
    registered_version: str,
) -> Path:
    comparison_rows = [
        "| Run ID | Hyperparameters | Bikes RMSE | Docks RMSE | Mean RMSE |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for result in results:
        comparison_rows.append(
            f"| `{result.run_id}` | `{result.parameters}` | "
            f"{result.metrics['rmse_bikes']:.4f} | "
            f"{result.metrics['rmse_docks']:.4f} | "
            f"{result.metrics['mean_rmse']:.4f} |"
        )

    metrics = best_result.metrics
    report_lines = [
        "# Hasil Eksperimen MLflow",
        "",
        f"Experiment: `{experiment}`",
        f"Git revision: `{revision}`",
        f"Git worktree dirty: `{str(worktree_dirty).lower()}`",
        f"Training script SHA-256: `{source_hash}`",
        f"Feature table SHA-256: `{data_hash}`",
        f"Pointer DVC SHA-256: `{pointer_hash}`",
        f"Train rows: {train_rows}; validation rows: {test_rows}.",
        "",
        *comparison_rows,
        "",
        "## Model terpilih",
        "",
        f"- Nama registry: `{REGISTERED_MODEL_NAME}`",
        f"- Versi registry: `{registered_version}`",
        f"- Run ID: `{best_result.run_id}`",
        f"- Parameter: `{best_result.parameters}`",
        f"- MAE bikes: {metrics['mae_bikes']:.4f}",
        f"- RMSE bikes: {metrics['rmse_bikes']:.4f}",
        f"- MAE docks: {metrics['mae_docks']:.4f}",
        f"- RMSE docks: {metrics['rmse_docks']:.4f}",
        f"- Mean RMSE used for selection: {metrics['mean_rmse']:.4f}",
        "",
        "Validation memakai waktu observasi terbaru dan tidak diacak. "
        "Perbandingan ini merupakan eksperimen awal; evaluasi deployment "
        "perlu periode test yang terpisah.",
        "",
    ]

    report_path = ROOT_DIR / "reports/mlflow_experiment_results.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="Lokasi feature table (default: data/processed/station_features.csv).",
    )
    parser.add_argument(
        "--experiment",
        default=EXPERIMENT_NAME,
        help="Nama experiment MLflow.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    data_path = args.data.resolve()
    X, y, timestamps = load_data(data_path)
    X_train, X_test, y_train, y_test, train_times, test_times = chronological_split(
        X, y, timestamps
    )

    tracking_db = ROOT_DIR / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")
    mlflow.set_experiment(args.experiment)

    data_hash = sha256_file(data_path)
    pointer_hash = sha256_file(POINTER_PATH) if POINTER_PATH.exists() else "unavailable"
    source_hash = sha256_file(Path(__file__).resolve())
    revision = git_revision()
    worktree_dirty = git_worktree_dirty()
    tags = {
        "git_revision": revision,
        "git_worktree_dirty": str(worktree_dirty).lower(),
        "train_script_sha256": source_hash,
        "feature_table_sha256": data_hash,
        "processed_dvc_pointer_sha256": pointer_hash,
        "selection_metric": "rmse_minimize",
    }

    LOGGER.info(
        "Dataset=%s labeled=%d train=%d test=%d train_until=%s test_from=%s",
        data_path,
        len(y),
        len(y_train),
        len(y_test),
        train_times.max().isoformat(),
        test_times.min().isoformat(),
    )

    results = [
        run_experiment(
            index,
            parameters,
            X_train,
            X_test,
            y_train,
            y_test,
            train_times,
            test_times,
            tags,
        )
        for index, parameters in enumerate(RUNS, start=1)
    ]
    best_result = min(results, key=lambda result: result.metrics["mean_rmse"])
    LOGGER.info(
        "Best run by lowest mean RMSE: %s (bikes RMSE=%.4f, docks RMSE=%.4f)",
        best_result.run_id,
        best_result.metrics["rmse_bikes"],
        best_result.metrics["rmse_docks"],
    )

    registered = mlflow.register_model(best_result.model_uri, REGISTERED_MODEL_NAME)
    LOGGER.info(
        "Registered model=%s version=%s params=%s",
        REGISTERED_MODEL_NAME,
        registered.version,
        best_result.parameters,
    )

    report_path = write_results_report(
        experiment=args.experiment,
        revision=revision,
        worktree_dirty=worktree_dirty,
        source_hash=source_hash,
        data_hash=data_hash,
        pointer_hash=pointer_hash,
        train_rows=len(y_train),
        test_rows=len(y_test),
        results=results,
        best_result=best_result,
        registered_version=registered.version,
    )
    LOGGER.info("Markdown report written: %s", report_path)


if __name__ == "__main__":
    main()
