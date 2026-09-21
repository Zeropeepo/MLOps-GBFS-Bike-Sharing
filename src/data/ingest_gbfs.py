"""Capture a reproducible snapshot of the Citi Bike GBFS feeds.

The script discovers the current feed URLs from ``gbfs.json``, downloads the
station information and station status feeds, stores the raw JSON in
``data/raw`` and writes a compact capture manifest for the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


DEFAULT_DISCOVERY_URL = "https://gbfs.citibikenyc.com/gbfs/2.3/gbfs.json"
FEED_NAMES = ("station_information", "station_status")


def fetch_json(url: str, retries: int = 3, timeout: int = 20) -> tuple[dict[str, Any], int]:
    """Fetch one JSON endpoint with bounded retries and return payload/status."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json(), response.status_code
        except (requests.RequestException, ValueError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def sha256_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def capture(discovery_url: str, raw_dir: Path, evidence_path: Path) -> Path:
    captured_at = datetime.now(timezone.utc)
    stamp = captured_at.strftime("%Y%m%dT%H%M%SZ")
    raw_dir.mkdir(parents=True, exist_ok=True)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)

    discovery, discovery_status = fetch_json(discovery_url)
    language_data = discovery.get("data", {}).get("en", {})
    discovered = {
        item["name"]: item["url"]
        for item in language_data.get("feeds", [])
        if "name" in item and "url" in item
    }
    missing = [name for name in FEED_NAMES if name not in discovered]
    if missing:
        raise RuntimeError(f"Discovery feed is missing: {', '.join(missing)}")

    manifest: dict[str, Any] = {
        "captured_at_utc": captured_at.isoformat(),
        "discovery": {
            "url": discovery_url,
            "http_status": discovery_status,
            "version": discovery.get("version"),
            "last_updated": discovery.get("last_updated"),
            "ttl": discovery.get("ttl"),
        },
        "feeds": {},
    }

    for feed_name in FEED_NAMES:
        payload, status = fetch_json(discovered[feed_name])
        serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        output_path = raw_dir / f"{feed_name}_{stamp}.json"
        output_path.write_text(serialized, encoding="utf-8")
        feed_data = payload.get("data", {})
        stations = feed_data.get("stations", [])
        first_station = stations[0] if stations else {}
        manifest["feeds"][feed_name] = {
            "url": discovered[feed_name],
            "http_status": status,
            "version": payload.get("version"),
            "last_updated": payload.get("last_updated"),
            "ttl": payload.get("ttl"),
            "record_count": len(stations),
            "sample_fields": sorted(first_station.keys()),
            "raw_file": str(output_path),
            "raw_bytes": output_path.stat().st_size,
            "sha256": sha256_text(serialized),
        }

    evidence_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return evidence_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-url", default=DEFAULT_DISCOVERY_URL)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--evidence-path",
        type=Path,
        default=Path("reports/evidence/gbfs_capture.json"),
    )
    args = parser.parse_args()
    print(capture(args.discovery_url, args.raw_dir, args.evidence_path))


if __name__ == "__main__":
    main()
