import pandas as pd
import pytest

from src.preprocess import make_features


def test_features_use_previous_observation_and_match_future_target():
    observations = pd.DataFrame(
        {
            "source_capture_id": ["first", "second", "future"],
            "collected_at_utc": [
                "2026-10-22T12:00:00Z",
                "2026-10-22T12:05:00Z",
                "2026-10-22T12:30:00Z",
            ],
            "station_id": ["station-a"] * 3,
            "name": ["Station A"] * 3,
            "lat": [40.7] * 3,
            "lon": [-74.0] * 3,
            "capacity": [20] * 3,
            "num_bikes_available": [3, 5, 9],
            "num_docks_available": [17, 15, 11],
            "is_installed": [1] * 3,
            "is_renting": [1] * 3,
            "is_returning": [1] * 3,
            "last_reported_utc": ["2026-10-22T12:00:00Z"] * 3,
        }
    )

    result = make_features(observations, target_tolerance_seconds=150).set_index(
        "source_capture_id"
    )

    assert result.loc["first", "bike_availability_ratio"] == pytest.approx(0.15)
    assert pd.isna(result.loc["first", "previous_num_bikes_available"])
    assert result.loc["second", "previous_num_bikes_available"] == 3
    assert result.loc["first", "num_bikes_available_t_plus_30m"] == 9
    assert result.loc["first", "num_docks_available_t_plus_30m"] == 11
    assert pd.isna(result.loc["second", "num_bikes_available_t_plus_30m"])
