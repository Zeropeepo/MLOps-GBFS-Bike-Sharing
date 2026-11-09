from types import SimpleNamespace

import pytest

from src.modeling.validate_candidate import validate


class FakeClient:
    def __init__(self, bikes=2.0, docks=3.0, source_run_id="run1"):
        self.metrics = {"rmse_bikes": bikes, "rmse_docks": docks}
        self.source_run_id = source_run_id

    def get_run(self, _run_id):
        return SimpleNamespace(
            info=SimpleNamespace(status="FINISHED"),
            data=SimpleNamespace(metrics=self.metrics, tags={"feature_table_sha256": "abc"}),
        )

    def get_logged_model(self, _model_id):
        return SimpleNamespace(source_run_id=self.source_run_id)


CANDIDATE = {"run_id": "run1", "model_uri": "models:/m-123", "data_sha256": "abc"}
LIMITS = {"max_rmse_bikes": 5.0, "max_rmse_docks": 5.0}


def test_candidate_passes_only_when_both_metrics_meet_limits():
    assert validate(CANDIDATE, LIMITS, FakeClient())["passed"]
    assert not validate(CANDIDATE, LIMITS, FakeClient(docks=5.1))["passed"]


def test_candidate_rejects_nonfinite_metrics():
    with pytest.raises(ValueError, match="tidak valid"):
        validate(CANDIDATE, LIMITS, FakeClient(bikes=float("nan")))


def test_candidate_rejects_model_from_another_run():
    with pytest.raises(ValueError, match="tidak berasal"):
        validate(CANDIDATE, LIMITS, FakeClient(source_run_id="another"))
