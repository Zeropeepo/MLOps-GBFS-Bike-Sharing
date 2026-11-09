"""Validate the DVC feature table before starting automated training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.modeling.train import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    NUMERIC_FEATURES,
    TARGETS,
    chronological_split,
    load_data,
    load_label_times,
)


def validate_dataset(path: Path) -> dict:
    result = {"dataset": str(path), "errors": [], "warnings": []}
    errors = result["errors"]
    warnings = result["warnings"]
    if not path.is_file():
        errors.append("Dataset tidak ditemukan.")
        return result

    try:
        frame = pd.read_csv(path, dtype={"station_id": "string"})
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        errors.append(f"Dataset tidak dapat dibaca: {exc}")
        return result

    result["rows"] = len(frame)
    required = {
        *NUMERIC_FEATURES,
        *CATEGORICAL_FEATURES,
        *TARGETS,
        "collected_at_utc",
        "target_timestamp_utc",
        "target_observation_at_utc",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        errors.append(f"Kolom wajib tidak tersedia: {', '.join(missing)}")
        return result
    if frame.empty:
        errors.append("Dataset kosong.")
        return result

    collected = pd.to_datetime(frame["collected_at_utc"], utc=True, errors="coerce")
    target_times = pd.to_datetime(frame["target_timestamp_utc"], utc=True, errors="coerce")
    observed = pd.to_datetime(
        frame["target_observation_at_utc"], utc=True, errors="coerce"
    )
    if collected.isna().any() or target_times.isna().any():
        errors.append("Waktu pengambilan atau target tidak valid.")
    if frame["station_id"].isna().any() or frame["station_id"].str.strip().eq("").any():
        errors.append("Terdapat station_id kosong.")
    if pd.DataFrame({"station": frame["station_id"], "time": collected}).duplicated().any():
        errors.append("Terdapat duplikasi stasiun pada waktu pengambilan yang sama.")

    numeric = {}
    for column in [*NUMERIC_FEATURES, *TARGETS]:
        values = pd.to_numeric(frame[column], errors="coerce")
        if (frame[column].notna() & values.isna()).any() or np.isinf(values).any():
            errors.append(f"Kolom {column} mengandung nilai nonnumerik atau tak hingga.")
        numeric[column] = values

    for column in ["capacity", "num_bikes_available", "num_docks_available"]:
        values = numeric[column]
        if values.isna().any() or (values < 0).any() or (values % 1 != 0).any():
            errors.append(f"Kolom {column} harus berisi bilangan bulat nonnegatif.")
    if (numeric["capacity"] <= 0).any():
        errors.append("Kapasitas stasiun harus lebih besar dari nol.")
    for column in ["is_installed", "is_renting", "is_returning", "is_weekend_utc"]:
        if not numeric[column].dropna().isin([0, 1]).all():
            errors.append(f"Kolom {column} harus bernilai 0 atau 1 jika tersedia.")

    for column in ["num_bikes_available", "num_docks_available"]:
        count = int((numeric[column] > numeric["capacity"]).sum())
        if count:
            warnings.append(f"{count} baris memiliki {column} melebihi kapasitas.")

    present = [numeric[target].notna() for target in TARGETS]
    labeled = present[0] & present[1]
    result["labeled_rows"] = int(labeled.sum())
    if (present[0] != present[1]).any():
        errors.append("Sebagian baris hanya memiliki satu dari dua target.")
    for target in TARGETS:
        values = numeric[target].dropna()
        if (values < 0).any() or (values % 1 != 0).any():
            errors.append(f"Target {target} harus berupa jumlah bulat nonnegatif.")
    if not collected.isna().any() and not target_times.isna().any():
        if ((target_times - collected - pd.Timedelta(minutes=30)).abs() > pd.Timedelta(milliseconds=1)).any():
            errors.append("Waktu target tidak sesuai dengan horizon 30 menit.")
        if observed[labeled].isna().any():
            errors.append("Waktu observasi target untuk data berlabel tidak tersedia.")
        elif ((observed[labeled] - target_times[labeled]).abs() > pd.Timedelta(seconds=150)).any():
            errors.append("Waktu observasi target melewati toleransi 150 detik.")

    if errors:
        return result
    try:
        features, targets, timestamps = load_data(path)
        label_times = load_label_times(path, timestamps.index)
        x_train, x_test, _, _, train_times, test_times = chronological_split(
            features, targets, timestamps, label_times
        )
    except (SystemExit, ValueError, OSError) as exc:
        errors.append(str(exc))
        return result
    if len(x_train) < 2 or len(x_test) < 2:
        errors.append("Training dan validasi masing-masing memerlukan minimal dua baris.")
    cutoff = test_times.min()
    purged = int(
        ((timestamps < cutoff) & (label_times + pd.Timedelta(seconds=1) > cutoff)).sum()
    )
    if purged:
        warnings.append(f"{purged} baris training dikeluarkan untuk mencegah kebocoran waktu.")
    result.update(
        train_rows=len(x_train),
        validation_rows=len(x_test),
        train_last_timestamp_utc=train_times.max().isoformat(),
        train_last_label_timestamp_utc=label_times.loc[train_times.index].max().isoformat(),
        validation_first_timestamp_utc=cutoff.isoformat(),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    result = validate_dataset(args.data.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
