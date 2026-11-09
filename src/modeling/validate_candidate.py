"""Validate a trained candidate and optionally register it to Staging."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient


def validate(candidate: dict, limits: dict, client: MlflowClient) -> dict:
    run = client.get_run(candidate["run_id"])
    if run.info.status != "FINISHED":
        raise ValueError("Run kandidat belum selesai dengan sukses.")
    model_uri = candidate["model_uri"]
    if not model_uri.startswith("models:/m-"):
        raise ValueError("URI kandidat bukan logged model MLflow yang didukung.")
    model = client.get_logged_model(model_uri.removeprefix("models:/"))
    if model.source_run_id != candidate["run_id"]:
        raise ValueError("Model tidak berasal dari run kandidat.")
    if run.data.tags.get("feature_table_sha256") != candidate["data_sha256"]:
        raise ValueError("Identitas dataset kandidat tidak cocok dengan run MLflow.")
    checks = {}
    for metric in ("rmse_bikes", "rmse_docks"):
        value = float(run.data.metrics[metric])
        limit = float(limits[f"max_{metric}"])
        if not math.isfinite(value) or not math.isfinite(limit) or limit <= 0:
            raise ValueError(f"Nilai {metric} atau ambangnya tidak valid.")
        checks[metric] = {"value": value, "maximum": limit, "passed": value <= limit}
    return {"run_id": candidate["run_id"], "model_uri": model_uri,
            "checks": checks, "passed": all(item["passed"] for item in checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--limits", type=Path, default=Path("config/model_validation.yaml"))
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    limits = yaml.safe_load(args.limits.read_text(encoding="utf-8"))
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    decision = validate(candidate, limits, client)
    args.decision.parent.mkdir(parents=True, exist_ok=True)
    args.decision.write_text(json.dumps(decision, indent=2, allow_nan=False) + "\n")
    print(json.dumps(decision, indent=2))
    if not decision["passed"]:
        raise SystemExit("Model tidak lolos ambang validasi. Registrasi dibatalkan.")
    if args.register:
        version = mlflow.register_model(candidate["model_uri"], candidate["model_name"])
        client.update_model_version(
            candidate["model_name"], version.version,
            description=(f"Kandidat CI dari run {candidate['run_id']}. "
                         f"Data SHA-256: {candidate['data_sha256']}. "
                         f"Validasi: {json.dumps(decision['checks'])}"),
        )
        client.set_model_version_tag(candidate["model_name"], version.version,
                                     "validation", "passed")
        client.transition_model_version_stage(candidate["model_name"], version.version, "Staging")
        print(f"Terdaftar: {candidate['model_name']} versi {version.version}, stage Staging")


if __name__ == "__main__":
    main()
