import numpy as np
import pandas as pd
import pytest

from src.modeling.train import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGETS,
    build_model,
    calculate_metrics,
    chronological_split,
    load_data,
)


def test_load_data_rejects_missing_required_columns(tmp_path):
    dataset = tmp_path / "incomplete.csv"
    pd.DataFrame({"station_id": ["station-a"]}).to_csv(dataset, index=False)

    with pytest.raises(SystemExit, match="Kolom dataset tidak lengkap"):
        load_data(dataset)


def test_load_data_keeps_only_rows_with_targets(tmp_path):
    dataset = tmp_path / "features.csv"
    frame = pd.DataFrame({column: [1, 2, 3] for column in NUMERIC_FEATURES})
    frame["station_id"] = ["a", "a", "b"]
    frame["collected_at_utc"] = [
        "2026-10-22T12:00:00Z",
        "2026-10-22T12:05:00Z",
        "2026-10-22T12:10:00Z",
    ]
    frame[TARGETS[0]] = [2, 3, None]
    frame[TARGETS[1]] = [4, 5, None]
    frame.to_csv(dataset, index=False)

    features, targets, timestamps = load_data(dataset)

    assert len(features) == len(targets) == len(timestamps) == 2
    assert features.columns.tolist() == [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    assert targets[TARGETS[0]].tolist() == [2.0, 3.0]


def test_chronological_split_keeps_timestamp_groups_together():
    features = pd.DataFrame({"example": range(6)})
    targets = pd.DataFrame({"target": range(6)})
    timestamps = pd.to_datetime(
        [
            "2026-10-22T12:00:00Z",
            "2026-10-22T12:00:00Z",
            "2026-10-22T12:05:00Z",
            "2026-10-22T12:05:00Z",
            "2026-10-22T12:10:00Z",
            "2026-10-22T12:10:00Z",
        ],
        utc=True,
    ).to_series(index=features.index)

    x_train, x_test, y_train, y_test, train_times, test_times = chronological_split(
        features, targets, timestamps
    )

    assert len(x_train) == len(y_train) == 4
    assert len(x_test) == len(y_test) == 2
    assert train_times.max() < test_times.min()
    assert set(train_times).isdisjoint(set(test_times))


def test_chronological_split_excludes_labels_from_validation_future():
    features = pd.DataFrame({"example": range(6)})
    targets = pd.DataFrame({"target": range(6)})
    timestamps = pd.Series(pd.to_datetime([
        "2026-10-22T12:00:00Z", "2026-10-22T12:00:00Z",
        "2026-10-22T12:05:00Z", "2026-10-22T12:05:00Z",
        "2026-10-22T12:10:00Z", "2026-10-22T12:10:00Z",
    ], utc=True))
    label_times = pd.Series(pd.to_datetime([
        "2026-10-22T12:04:00Z", "2026-10-22T12:04:00Z",
        "2026-10-22T12:20:00Z", "2026-10-22T12:20:00Z",
        "2026-10-22T12:40:00Z", "2026-10-22T12:40:00Z",
    ], utc=True))

    x_train, x_test, _, _, train_times, test_times = chronological_split(
        features, targets, timestamps, label_times
    )

    assert x_train.index.tolist() == [0, 1]
    assert x_test.index.tolist() == [4, 5]
    assert label_times.loc[train_times.index].max() < test_times.min()


def test_chronological_split_rejects_all_training_labels_in_future():
    features = pd.DataFrame({"example": range(4)})
    targets = pd.DataFrame({"target": range(4)})
    timestamps = pd.Series(pd.to_datetime([
        "2026-10-22T12:00:00Z", "2026-10-22T12:00:00Z",
        "2026-10-22T12:05:00Z", "2026-10-22T12:05:00Z",
    ], utc=True))
    label_times = timestamps + pd.Timedelta(minutes=30)

    with pytest.raises(SystemExit, match="label masa depan"):
        chronological_split(features, targets, timestamps, label_times)


def test_split_excludes_label_recorded_in_validation_start_second():
    features = pd.DataFrame({"example": range(3)})
    targets = pd.DataFrame({"target": range(3)})
    timestamps = pd.Series(pd.to_datetime([
        "2026-10-22T12:00:00Z",
        "2026-10-22T12:05:00Z",
        "2026-10-22T12:10:00.500Z",
    ], utc=True, format="mixed"))
    label_times = pd.Series(pd.to_datetime([
        "2026-10-22T12:04:00Z",
        "2026-10-22T12:10:00Z",
        "2026-10-22T12:40:00Z",
    ], utc=True))

    x_train, _, _, _, _, _ = chronological_split(features, targets, timestamps, label_times)

    assert x_train.index.tolist() == [0]


def test_calculate_metrics_averages_rmse_for_both_targets():
    targets = pd.DataFrame({TARGETS[0]: [1, 3], TARGETS[1]: [2, 6]})
    predictions = pd.DataFrame({TARGETS[0]: [2, 4], TARGETS[1]: [4, 8]})

    metrics = calculate_metrics(targets, predictions)

    assert metrics["rmse_bikes"] == pytest.approx(1.0)
    assert metrics["rmse_docks"] == pytest.approx(2.0)
    assert metrics["mean_rmse"] == pytest.approx(1.5)


def test_model_predicts_for_unseen_station_and_missing_numeric_value():
    features = pd.DataFrame({column: [1.0, 2.0, 3.0] for column in NUMERIC_FEATURES})
    features["station_id"] = ["a", "a", "b"]
    targets = pd.DataFrame({TARGETS[0]: [1, 2, 3], TARGETS[1]: [3, 2, 1]})
    model = build_model({"n_estimators": 3, "max_depth": 2, "min_samples_leaf": 1})
    model.fit(features, targets)

    new_station = features.iloc[[0]].copy()
    new_station["station_id"] = "unseen-station"
    new_station["capacity"] = np.nan
    prediction = model.predict(new_station)

    assert prediction.shape == (1, 2)
    assert np.isfinite(prediction).all()
