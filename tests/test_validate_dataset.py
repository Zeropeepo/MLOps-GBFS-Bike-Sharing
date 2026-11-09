import pandas as pd

from src.modeling.train import NUMERIC_FEATURES, TARGETS
from src.modeling.validate_dataset import validate_dataset


def make_dataset(path, spacing_minutes=60):
    rows = []
    for group in range(5):
        timestamp = pd.Timestamp("2026-10-22T12:00:00Z") + pd.Timedelta(
            minutes=group * spacing_minutes
        )
        for station in ["a", "b"]:
            row = {column: 1.0 for column in NUMERIC_FEATURES}
            row.update(
                station_id=station,
                capacity=10,
                num_bikes_available=3,
                num_docks_available=7,
                collected_at_utc=timestamp.isoformat(),
                target_timestamp_utc=(timestamp + pd.Timedelta(minutes=30)).isoformat(),
                target_observation_at_utc=(
                    timestamp + pd.Timedelta(minutes=30)
                ).isoformat(),
            )
            row[TARGETS[0]] = 4
            row[TARGETS[1]] = 6
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_valid_dataset_passes_with_separated_label_times(tmp_path):
    dataset = tmp_path / "features.csv"
    make_dataset(dataset)

    result = validate_dataset(dataset)

    assert result["errors"] == []
    assert result["train_rows"] == 8
    assert result["validation_rows"] == 2
    assert result["train_last_label_timestamp_utc"] < result["validation_first_timestamp_utc"]


def test_validator_rejects_missing_columns(tmp_path):
    dataset = tmp_path / "incomplete.csv"
    pd.DataFrame({"station_id": ["a"]}).to_csv(dataset, index=False)

    result = validate_dataset(dataset)

    assert any("Kolom wajib" in error for error in result["errors"])


def test_validator_rejects_negative_counts(tmp_path):
    dataset = tmp_path / "negative.csv"
    make_dataset(dataset)
    frame = pd.read_csv(dataset)
    frame.loc[0, "num_bikes_available"] = -1
    frame.to_csv(dataset, index=False)

    result = validate_dataset(dataset)

    assert any("num_bikes_available" in error for error in result["errors"])


def test_validator_reports_capacity_anomaly_without_changing_data(tmp_path):
    dataset = tmp_path / "over-capacity.csv"
    make_dataset(dataset)
    frame = pd.read_csv(dataset)
    frame.loc[0, "num_bikes_available"] = 11
    frame.to_csv(dataset, index=False)

    result = validate_dataset(dataset)

    assert result["errors"] == []
    assert any("melebihi kapasitas" in warning for warning in result["warnings"])
    assert pd.read_csv(dataset).loc[0, "num_bikes_available"] == 11


def test_validator_rejects_data_without_leakage_free_training(tmp_path):
    dataset = tmp_path / "overlapping.csv"
    make_dataset(dataset, spacing_minutes=5)

    result = validate_dataset(dataset)

    assert any("label masa depan" in error for error in result["errors"])
