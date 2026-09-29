"""Preprocess snapshot GBFS dan siapkan feature table stasiun."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
SNAPSHOT_PATTERN = re.compile(
    r"^(station_information|station_status)_(\d{8}T\d{6}(?:\d{6})?Z)$"
)
COUNT_COLUMNS = ["num_bikes_available", "num_docks_available"]
REJECTION_COLUMNS = ["snapshot_id", "feed", "station_id", "reason"]
OBSERVATION_COLUMNS = [
    "source_capture_id",
    "collected_at_utc",
    "station_id",
    "name",
    "lat",
    "lon",
    "capacity",
    *COUNT_COLUMNS,
    "is_installed",
    "is_renting",
    "is_returning",
    "last_reported_utc",
]
FEATURE_COLUMNS = [
    *OBSERVATION_COLUMNS,
    "collected_hour_utc",
    "collected_day_of_week_utc",
    "is_weekend_utc",
    "bike_availability_ratio",
    "dock_availability_ratio",
    "previous_num_bikes_available",
    "previous_num_docks_available",
    "target_timestamp_utc",
    "target_observation_at_utc",
    "num_bikes_available_t_plus_30m",
    "num_docks_available_t_plus_30m",
]


def project_path(path: str) -> Path:
    """Resolve relative path dari repository root."""
    result = Path(path).expanduser()
    return result if result.is_absolute() else ROOT_DIR / result


def read_stations(path: Path) -> pd.DataFrame:
    """Read daftar stasiun dari raw GBFS JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        raise TypeError(f"Format JSON tidak sesuai: {path.name}")

    stations = payload["data"].get("stations")
    if not isinstance(stations, list):
        raise TypeError(f"Daftar data.stations tidak ditemukan: {path.name}")
    return pd.DataFrame(stations)


def reject(
    rejected: list[dict[str, str]],
    snapshot_id: str,
    feed: str,
    station_id: Any,
    reason: str,
) -> None:
    """Tambahkan satu rejected row ke rejection report."""
    rejected.append(
        {
            "snapshot_id": snapshot_id,
            "feed": feed,
            "station_id": "" if pd.isna(station_id) else str(station_id),
            "reason": reason,
        }
    )


def prepare_feed(
    frame: pd.DataFrame,
    feed_name: str,
    snapshot_id: str,
    rejected: list[dict[str, str]],
) -> pd.DataFrame:
    """Remove row tanpa ID dan keep record terakhir untuk duplicate ID."""
    if "station_id" not in frame:
        frame["station_id"] = pd.NA

    frame["station_id"] = frame["station_id"].astype("string").str.strip()
    missing_id = frame["station_id"].isna() | frame["station_id"].eq("")
    for station_id in frame.loc[missing_id, "station_id"]:
        reject(rejected, snapshot_id, feed_name, station_id, "missing_station_id")
    frame = frame.loc[~missing_id].copy()

    duplicate_id = frame["station_id"].duplicated(keep="last")
    for station_id in frame.loc[duplicate_id, "station_id"]:
        reject(
            rejected,
            snapshot_id,
            feed_name,
            station_id,
            "duplicate_station_id; kept_last_record",
        )
    return frame.loc[~duplicate_id].copy()


def capture_time(snapshot_id: str) -> str:
    """Convert timestamp di filename menjadi ISO UTC timestamp."""
    if len(snapshot_id) == 22:
        date_format = "%Y%m%dT%H%M%S%fZ"
    else:
        date_format = "%Y%m%dT%H%M%SZ"
    captured_at = datetime.strptime(snapshot_id, date_format).replace(
        tzinfo=timezone.utc
    )
    return captured_at.isoformat()


def clean_snapshot(
    information: pd.DataFrame,
    status: pd.DataFrame,
    snapshot_id: str,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Join satu pair feed dan reject row yang gagal basic validation."""
    rejected: list[dict[str, str]] = []
    information = prepare_feed(
        information, "station_information", snapshot_id, rejected
    )
    status = prepare_feed(status, "station_status", snapshot_id, rejected)

    # Pilih kolom yang dipakai pipeline. Missing column diisi blank supaya row
    # bisa dicatat sebagai rejected data.
    information_fields = ["station_id", "name", "lat", "lon", "capacity"]
    status_fields = [
        "station_id",
        *COUNT_COLUMNS,
        "is_installed",
        "is_renting",
        "is_returning",
        "last_reported",
    ]
    for column in information_fields:
        if column not in information:
            information[column] = pd.NA
    for column in status_fields:
        if column not in status:
            status[column] = pd.NA

    joined = information[information_fields].merge(
        status[status_fields],
        on="station_id",
        how="outer",
        indicator=True,
        suffixes=("_information", "_status"),
    )

    for _, row in joined.loc[joined["_merge"] == "left_only"].iterrows():
        reject(
            rejected,
            snapshot_id,
            "station_information",
            row["station_id"],
            "no_matching_station_status",
        )
    for _, row in joined.loc[joined["_merge"] == "right_only"].iterrows():
        reject(
            rejected,
            snapshot_id,
            "station_status",
            row["station_id"],
            "no_matching_station_information",
        )

    clean = joined.loc[joined["_merge"] == "both"].copy()
    for column in ["capacity", *COUNT_COLUMNS, "lat", "lon", "last_reported"]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    valid_rows = []
    for index, row in clean.iterrows():
        problems = []
        if pd.isna(row["capacity"]):
            problems.append("missing_or_non_numeric_capacity")
        elif row["capacity"] <= 0:
            problems.append("capacity_not_positive")
        elif row["capacity"] % 1 != 0:
            problems.append("non_integer_capacity")

        for column in COUNT_COLUMNS:
            if pd.isna(row[column]):
                problems.append(f"missing_or_non_numeric_{column}")
            elif row[column] < 0:
                problems.append(f"negative_{column}")
            elif row[column] % 1 != 0:
                problems.append(f"non_integer_{column}")

        if problems:
            reject(
                rejected,
                snapshot_id,
                "joined_snapshot",
                row["station_id"],
                ";".join(problems),
            )
        else:
            valid_rows.append(index)

    clean = clean.loc[valid_rows].copy()
    clean["source_capture_id"] = snapshot_id
    clean["collected_at_utc"] = capture_time(snapshot_id)
    clean["last_reported_utc"] = pd.to_datetime(
        clean["last_reported"], unit="s", utc=True, errors="coerce"
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    for column in ["is_installed", "is_renting", "is_returning"]:
        flags = pd.to_numeric(clean[column], errors="coerce")
        clean[column] = flags.where(flags.isin([0, 1])).astype("Int64")
    for column in COUNT_COLUMNS:
        clean[column] = clean[column].astype("Int64")

    clean = clean[OBSERVATION_COLUMNS]
    clean = clean.sort_values("station_id").reset_index(drop=True)
    return clean, rejected


def make_features(
    observations: pd.DataFrame, target_tolerance_seconds: int
) -> pd.DataFrame:
    """Tambahkan time dan availability features, plus target jika tersedia."""
    if observations.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    features = observations.copy()
    features["collected_at_utc"] = pd.to_datetime(
        features["collected_at_utc"], utc=True, format="mixed"
    )
    features = features.sort_values(["station_id", "collected_at_utc"])

    # Calendar dan ratio features hanya memakai current dan previous observation.
    features["collected_hour_utc"] = features["collected_at_utc"].dt.hour
    features["collected_day_of_week_utc"] = features["collected_at_utc"].dt.dayofweek
    features["is_weekend_utc"] = (features["collected_day_of_week_utc"] >= 5).astype(
        "int8"
    )
    features["bike_availability_ratio"] = (
        features["num_bikes_available"] / features["capacity"]
    )
    features["dock_availability_ratio"] = (
        features["num_docks_available"] / features["capacity"]
    )

    by_station = features.groupby("station_id")
    features["previous_num_bikes_available"] = by_station["num_bikes_available"].shift(
        1
    )
    features["previous_num_docks_available"] = by_station["num_docks_available"].shift(
        1
    )

    # Cari snapshot terdekat ke waktu t + 30 menit. Tolerance 150 detik
    # mengakomodasi sedikit pergeseran polling interval lima menit.
    features["target_timestamp_utc"] = features["collected_at_utc"] + pd.Timedelta(
        minutes=30
    )
    future = features[["station_id", "collected_at_utc", *COUNT_COLUMNS]].rename(
        columns={
            "collected_at_utc": "target_observation_at_utc",
            "num_bikes_available": "num_bikes_available_t_plus_30m",
            "num_docks_available": "num_docks_available_t_plus_30m",
        }
    )

    features = pd.merge_asof(
        features.sort_values(["target_timestamp_utc", "station_id"]),
        future.sort_values(["target_observation_at_utc", "station_id"]),
        left_on="target_timestamp_utc",
        right_on="target_observation_at_utc",
        by="station_id",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=target_tolerance_seconds),
    )

    features["collected_at_utc"] = features["collected_at_utc"].dt.strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    features["target_timestamp_utc"] = features["target_timestamp_utc"].dt.strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    features["target_observation_at_utc"] = pd.to_datetime(
        features["target_observation_at_utc"], utc=True, errors="coerce"
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    for column in [
        "num_bikes_available_t_plus_30m",
        "num_docks_available_t_plus_30m",
    ]:
        features[column] = pd.to_numeric(features[column], errors="coerce").astype(
            "Int64"
        )

    return features[FEATURE_COLUMNS].reset_index(drop=True)


def preprocess(
    raw_dir: Path,
    interim_dir: Path,
    processed_dir: Path,
    target_tolerance_seconds: int = 150,
) -> dict[str, Any]:
    """Process semua complete snapshot pairs tanpa mengubah raw JSON files."""
    snapshots: dict[str, dict[str, Path]] = {}
    for path in sorted(raw_dir.glob("*.json")):
        match = SNAPSHOT_PATTERN.match(path.stem)
        if match:
            feed_name, snapshot_id = match.groups()
            snapshots.setdefault(snapshot_id, {})[feed_name] = path

    complete_ids = sorted(
        snapshot_id
        for snapshot_id, files in snapshots.items()
        if set(files) == {"station_information", "station_status"}
    )
    incomplete_ids = sorted(set(snapshots) - set(complete_ids))
    if not complete_ids:
        raise RuntimeError(f"Tidak ada pasangan snapshot lengkap di {raw_dir}")

    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    observations_by_snapshot = []
    all_rejected = []
    snapshot_summaries = []
    invalid_snapshots = []

    for snapshot_id in complete_ids:
        try:
            information = read_stations(snapshots[snapshot_id]["station_information"])
            status = read_stations(snapshots[snapshot_id]["station_status"])
            clean, rejected = clean_snapshot(information, status, snapshot_id)
        except (OSError, ValueError, TypeError) as error:
            invalid_snapshots.append({"snapshot_id": snapshot_id, "error": str(error)})
            continue

        rejected_frame = pd.DataFrame(rejected, columns=REJECTION_COLUMNS)
        clean.to_csv(
            interim_dir / f"station_observations_{snapshot_id}.csv", index=False
        )
        rejected_frame.to_csv(
            interim_dir / f"station_rejections_{snapshot_id}.csv", index=False
        )

        observations_by_snapshot.append(clean)
        all_rejected.extend(rejected)
        snapshot_summaries.append(
            {
                "snapshot_id": snapshot_id,
                "rows_kept": len(clean),
                "rows_rejected": len(rejected),
            }
        )

    if not observations_by_snapshot:
        raise RuntimeError("Semua pasangan snapshot tidak bisa diproses")

    observations = pd.concat(observations_by_snapshot, ignore_index=True)
    observations = observations.drop_duplicates(
        ["station_id", "collected_at_utc"], keep="last"
    )
    observations = observations.sort_values(["station_id", "collected_at_utc"])

    pd.DataFrame(all_rejected, columns=REJECTION_COLUMNS).to_csv(
        interim_dir / "station_rejections.csv", index=False
    )
    features = make_features(observations, target_tolerance_seconds)
    features_path = processed_dir / "station_features.csv"
    features.to_csv(features_path, index=False)

    manifest = {
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshots": snapshot_summaries,
        "incomplete_snapshots": incomplete_ids,
        "invalid_snapshots": invalid_snapshots,
        "rows_kept": len(observations),
        "rows_rejected": len(all_rejected),
        "rows_with_30_minute_target": int(
            features["num_bikes_available_t_plus_30m"].notna().sum()
        ),
        "target_tolerance_seconds": target_tolerance_seconds,
    }
    (interim_dir / "preprocess_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--interim-dir", default="data/interim")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--target-tolerance-seconds", type=int, default=150)
    args = parser.parse_args()

    if args.target_tolerance_seconds < 0:
        parser.error("toleransi target tidak boleh negatif")

    result = preprocess(
        project_path(args.raw_dir),
        project_path(args.interim_dir),
        project_path(args.processed_dir),
        args.target_tolerance_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
