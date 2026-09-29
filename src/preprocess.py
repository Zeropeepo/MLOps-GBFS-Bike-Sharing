"""Validate raw GBFS snapshots and produce clean station feature tables."""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATTERN = re.compile(
    r"^(station_information|station_status)_(\d{8}T\d{6}(?:\d{6})?Z)$"
)
COUNT_FIELDS = ("num_bikes_available", "num_docks_available")
OUTPUT_COLUMNS = [
    "source_capture_id",
    "collected_at_utc",
    "station_id",
    "name",
    "lat",
    "lon",
    "capacity",
    "num_bikes_available",
    "num_docks_available",
    "is_installed",
    "is_renting",
    "is_returning",
    "last_reported_utc",
]
LOGGER = logging.getLogger("gbfs_preprocessing")


def _project_path(path_value: str) -> Path:
    """Resolve relative CLI paths from the repository root."""
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _capture_datetime(stamp: str) -> datetime:
    """Parse both the original second-based and newer microsecond stamps."""
    fmt = "%Y%m%dT%H%M%S%fZ" if len(stamp) == 22 else "%Y%m%dT%H%M%SZ"
    return datetime.strptime(stamp, fmt).replace(tzinfo=timezone.utc)


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def _append_reject(
    rejects: list[dict[str, str]],
    stamp: str,
    feed_name: str,
    station_id: Any,
    reason: str,
    raw_record: Any,
) -> None:
    rejects.append(
        {
            "source_capture_id": stamp,
            "feed": feed_name,
            "station_id": "" if pd.isna(station_id) else str(station_id),
            "reason": reason,
            "raw_record": _safe_json(raw_record),
        }
    )


def _read_stations(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} JSON root must be an object")
    feed_data = payload.get("data")
    if not isinstance(feed_data, dict):
        raise TypeError(f"{path.name} does not contain a data object")
    stations = feed_data.get("stations")
    if not isinstance(stations, list):
        raise TypeError(f"{path.name} does not contain a data.stations list")
    if any(not isinstance(record, dict) for record in stations):
        raise TypeError(f"{path.name} contains a station record that is not an object")
    return stations


def _normalise_feed(
    records: list[dict[str, Any]],
    feed_name: str,
    stamp: str,
    rejects: list[dict[str, str]],
) -> pd.DataFrame:
    """Keep one non-empty station ID per feed and retain reasons for discarded rows."""
    frame = pd.DataFrame(records)
    if "station_id" not in frame:
        frame["station_id"] = pd.NA
    frame["_raw_record"] = [_safe_json(record) for record in records]
    frame["station_id"] = frame["station_id"].astype("string").str.strip()
    missing_id = frame["station_id"].isna() | frame["station_id"].eq("")
    for _, row in frame.loc[missing_id].iterrows():
        _append_reject(
            rejects,
            stamp,
            feed_name,
            row.get("station_id"),
            "missing_station_id",
            row["_raw_record"],
        )
    frame = frame.loc[~missing_id].copy()

    duplicate = frame["station_id"].duplicated(keep="last")
    for _, row in frame.loc[duplicate].iterrows():
        _append_reject(
            rejects,
            stamp,
            feed_name,
            row["station_id"],
            "duplicate_station_id; kept_last_record",
            row["_raw_record"],
        )
    return frame.loc[~duplicate].copy()


def _normalise_flag(value: Any) -> int | None:
    """Return GBFS boolean flags as nullable 0/1 values."""
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"true", "1"}:
            return 1
        if normalised in {"false", "0"}:
            return 0
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number in (0, 1) else None


def _clean_snapshot(
    information_records: list[dict[str, Any]],
    status_records: list[dict[str, Any]],
    stamp: str,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    rejects: list[dict[str, str]] = []
    information = _normalise_feed(
        information_records, "station_information", stamp, rejects
    )
    status = _normalise_feed(status_records, "station_status", stamp, rejects)

    information_fields = [
        "station_id",
        "name",
        "lat",
        "lon",
        "capacity",
        "_raw_record",
    ]
    status_fields = [
        "station_id",
        *COUNT_FIELDS,
        "is_installed",
        "is_renting",
        "is_returning",
        "last_reported",
        "_raw_record",
    ]
    for field in information_fields:
        if field not in information:
            information[field] = pd.NA
    for field in status_fields:
        if field not in status:
            status[field] = pd.NA

    joined = information[information_fields].merge(
        status[status_fields],
        on="station_id",
        how="outer",
        indicator=True,
        suffixes=("_info", "_status"),
    )
    for _, row in joined.loc[joined["_merge"] == "left_only"].iterrows():
        _append_reject(
            rejects,
            stamp,
            "station_information",
            row["station_id"],
            "no_matching_station_status",
            row.get("_raw_record_info"),
        )
    for _, row in joined.loc[joined["_merge"] == "right_only"].iterrows():
        _append_reject(
            rejects,
            stamp,
            "station_status",
            row["station_id"],
            "no_matching_station_information",
            row.get("_raw_record_status"),
        )

    clean = joined.loc[joined["_merge"] == "both"].copy()
    for field in ("capacity", *COUNT_FIELDS, "lat", "lon", "last_reported"):
        clean[field] = pd.to_numeric(clean[field], errors="coerce")

    valid_mask = pd.Series(True, index=clean.index)
    for index, row in clean.iterrows():
        reasons: list[str] = []
        capacity = row["capacity"]
        if pd.isna(capacity):
            reasons.append("missing_or_non_numeric_capacity")
        elif capacity <= 0:
            reasons.append("capacity_not_positive")
        for field in COUNT_FIELDS:
            value = row[field]
            if pd.isna(value):
                reasons.append(f"missing_or_non_numeric_{field}")
            elif value < 0:
                reasons.append(f"negative_{field}")
        if reasons:
            valid_mask.at[index] = False
            _append_reject(
                rejects,
                stamp,
                "joined_snapshot",
                row["station_id"],
                ";".join(reasons),
                {
                    "station_information": row.get("_raw_record_info"),
                    "station_status": row.get("_raw_record_status"),
                },
            )

    clean = clean.loc[valid_mask].copy()
    collected_at = _capture_datetime(stamp).isoformat()
    clean["source_capture_id"] = stamp
    clean["collected_at_utc"] = collected_at
    clean["last_reported_utc"] = pd.to_datetime(
        clean["last_reported"], unit="s", utc=True, errors="coerce"
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    for field in ("is_installed", "is_renting", "is_returning"):
        clean[field] = clean[field].map(_normalise_flag).astype("Int64")

    clean = clean[OUTPUT_COLUMNS].copy()
    clean["lat"] = pd.to_numeric(clean["lat"], errors="coerce")
    clean["lon"] = pd.to_numeric(clean["lon"], errors="coerce")
    clean["capacity"] = pd.to_numeric(clean["capacity"], errors="coerce")
    for field in COUNT_FIELDS:
        clean[field] = pd.to_numeric(clean[field], errors="coerce").astype("Int64")
    clean = clean.sort_values("station_id", kind="stable").reset_index(drop=True)
    return clean, rejects


def _add_features(
    observations: pd.DataFrame, target_tolerance_seconds: int
) -> pd.DataFrame:
    """Add time, ratio, previous-snapshot, and optional 30-minute target fields."""
    feature_columns = [
        *OUTPUT_COLUMNS,
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
    if observations.empty:
        return pd.DataFrame(columns=feature_columns)

    features = observations.copy()
    features["collected_at_utc"] = pd.to_datetime(
        features["collected_at_utc"], utc=True, format="mixed"
    )
    features = features.sort_values(
        ["station_id", "collected_at_utc"], kind="stable"
    ).reset_index(drop=True)
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
    grouped = features.groupby("station_id", sort=False)
    features["previous_num_bikes_available"] = grouped["num_bikes_available"].shift(1)
    features["previous_num_docks_available"] = grouped["num_docks_available"].shift(1)

    features["target_timestamp_utc"] = features["collected_at_utc"] + pd.Timedelta(
        minutes=30
    )
    target_source = features[["station_id", "collected_at_utc", *COUNT_FIELDS]].rename(
        columns={
            "collected_at_utc": "target_observation_at_utc",
            "num_bikes_available": "num_bikes_available_t_plus_30m",
            "num_docks_available": "num_docks_available_t_plus_30m",
        }
    )
    left = features.sort_values(["target_timestamp_utc", "station_id"], kind="stable")
    right = target_source.sort_values(
        ["target_observation_at_utc", "station_id"], kind="stable"
    )
    result = pd.merge_asof(
        left,
        right,
        left_on="target_timestamp_utc",
        right_on="target_observation_at_utc",
        by="station_id",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=target_tolerance_seconds),
    )
    result["collected_at_utc"] = result["collected_at_utc"].dt.strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    result["target_timestamp_utc"] = result["target_timestamp_utc"].dt.strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )
    result["target_observation_at_utc"] = pd.to_datetime(
        result["target_observation_at_utc"], utc=True, errors="coerce"
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    result["num_bikes_available_t_plus_30m"] = pd.to_numeric(
        result["num_bikes_available_t_plus_30m"], errors="coerce"
    ).astype("Int64")
    result["num_docks_available_t_plus_30m"] = pd.to_numeric(
        result["num_docks_available_t_plus_30m"], errors="coerce"
    ).astype("Int64")
    return result[feature_columns].reset_index(drop=True)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp_path, index=False)
    temp_path.replace(path)


def preprocess(
    raw_dir: Path,
    interim_dir: Path,
    processed_dir: Path,
    target_tolerance_seconds: int = 150,
) -> dict[str, Any]:
    """Process every complete raw snapshot pair without changing raw files."""
    discovered: dict[str, dict[str, Path]] = {}
    for path in sorted(raw_dir.glob("*.json")):
        match = SNAPSHOT_PATTERN.match(path.stem)
        if match:
            feed_name, stamp = match.groups()
            discovered.setdefault(stamp, {})[feed_name] = path

    complete_stamps = sorted(
        stamp
        for stamp, feeds in discovered.items()
        if set(feeds) == {"station_information", "station_status"}
    )
    incomplete_stamps = sorted(set(discovered) - set(complete_stamps))
    if not complete_stamps:
        raise RuntimeError(f"No complete GBFS snapshot pairs found in {raw_dir}")

    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    snapshot_results: list[tuple[str, pd.DataFrame, list[dict[str, str]]]] = []
    snapshot_errors: list[dict[str, str]] = []

    for stamp in complete_stamps:
        try:
            information_records = _read_stations(
                discovered[stamp]["station_information"]
            )
            status_records = _read_stations(discovered[stamp]["station_status"])
            cleaned, rejects = _clean_snapshot(
                information_records, status_records, stamp
            )
            snapshot_results.append((stamp, cleaned, rejects))
        except (OSError, json.JSONDecodeError, ValueError, TypeError) as error:
            LOGGER.error("Skipping invalid snapshot %s: %s", stamp, error)
            snapshot_errors.append({"source_capture_id": stamp, "error": str(error)})

    if not snapshot_results:
        raise RuntimeError("All complete raw snapshot pairs were invalid")

    accepted_frames: list[pd.DataFrame] = []
    all_rejects: list[dict[str, str]] = []
    snapshot_summaries: list[dict[str, Any]] = []
    for stamp, cleaned, rejects in snapshot_results:
        _write_csv(cleaned, interim_dir / f"station_observations_{stamp}.csv")
        reject_frame = pd.DataFrame(
            rejects,
            columns=[
                "source_capture_id",
                "feed",
                "station_id",
                "reason",
                "raw_record",
            ],
        )
        _write_csv(reject_frame, interim_dir / f"station_rejections_{stamp}.csv")
        accepted_frames.append(cleaned)
        all_rejects.extend(rejects)
        snapshot_summaries.append(
            {
                "source_capture_id": stamp,
                "station_rows_accepted": len(cleaned),
                "records_rejected": len(rejects),
            }
        )

    observations = pd.concat(accepted_frames, ignore_index=True)
    observations = observations.drop_duplicates(
        ["station_id", "collected_at_utc"], keep="last"
    ).sort_values(["station_id", "collected_at_utc"], kind="stable")
    _write_csv(
        pd.DataFrame(
            all_rejects,
            columns=[
                "source_capture_id",
                "feed",
                "station_id",
                "reason",
                "raw_record",
            ],
        ),
        interim_dir / "station_rejections.csv",
    )

    features = _add_features(observations, target_tolerance_seconds)
    features_path = processed_dir / "station_features.csv"
    _write_csv(features, features_path)
    manifest: dict[str, Any] = {
        "processed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_capture_ids": [
            summary["source_capture_id"] for summary in snapshot_summaries
        ],
        "snapshots": snapshot_summaries,
        "incomplete_capture_ids": incomplete_stamps,
        "invalid_snapshots": snapshot_errors,
        "station_rows_accepted": len(observations),
        "records_rejected": len(all_rejects),
        "feature_rows": len(features),
        "rows_with_30_minute_target": int(
            features["num_bikes_available_t_plus_30m"].notna().sum()
        ),
        "target_tolerance_seconds": target_tolerance_seconds,
        "interim_directory": str(interim_dir),
        "processed_file": str(features_path),
    }
    manifest_path = interim_dir / "preprocess_manifest.json"
    temp_manifest = manifest_path.with_suffix(".json.tmp")
    temp_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_manifest.replace(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--interim-dir", default="data/interim")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--target-tolerance-seconds", type=int, default=150)
    args = parser.parse_args()
    if args.target_tolerance_seconds < 0:
        parser.error("target tolerance must be non-negative")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    manifest = preprocess(
        raw_dir=_project_path(args.raw_dir),
        interim_dir=_project_path(args.interim_dir),
        processed_dir=_project_path(args.processed_dir),
        target_tolerance_seconds=args.target_tolerance_seconds,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
