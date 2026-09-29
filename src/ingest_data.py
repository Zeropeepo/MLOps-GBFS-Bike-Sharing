"""Download timestamped snapshots of the public Citi Bike GBFS feeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DISCOVERY_URL = "https://gbfs.citibikenyc.com/gbfs/2.3/gbfs.json"
FEED_NAMES = ("station_information", "station_status")
LOGGER = logging.getLogger("gbfs_ingestion")


def fetch_json(
    url: str, retries: int = 3, timeout: int = 20
) -> tuple[dict[str, Any], int, bytes]:
    """Fetch a JSON response, retrying transient errors with bounded backoff."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "MLOps-GBFS-Bike-Sharing/1.0"},
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise TypeError("The JSON response root must be an object")
            return payload, response.status_code, response.content
        except (requests.RequestException, ValueError) as error:
            last_error = error
            if attempt + 1 < retries:
                delay = min(2**attempt, 30)
                LOGGER.warning(
                    "Request failed (%s/%s): %s; retrying in %s seconds",
                    attempt + 1,
                    retries,
                    error,
                    delay,
                )
                time.sleep(delay)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def _discovered_feeds(discovery: dict[str, Any]) -> dict[str, str]:
    """Read feed URLs from the GBFS discovery document across language keys."""
    language_data = discovery.get("data", {})
    if isinstance(language_data, dict):
        language_entries = language_data.values()
    elif isinstance(language_data, list):
        language_entries = language_data
    else:
        language_entries = []

    discovered: dict[str, str] = {}
    for language_entry in language_entries:
        if not isinstance(language_entry, dict):
            continue
        for feed in language_entry.get("feeds", []):
            if isinstance(feed, dict) and feed.get("name") and feed.get("url"):
                discovered[str(feed["name"])] = str(feed["url"])
    return discovered


def _capture_stamp(captured_at: datetime) -> str:
    """Return a UTC filename stamp with microseconds to avoid snapshot clobbering."""
    return captured_at.strftime("%Y%m%dT%H%M%S%fZ")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def capture(
    discovery_url: str,
    raw_dir: Path,
    evidence_dir: Path,
    retries: int = 3,
    timeout: int = 20,
) -> Path:
    """Download both station feeds and save a manifest for this capture."""
    captured_at = datetime.now(timezone.utc)
    raw_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    while True:
        stamp = _capture_stamp(captured_at)
        raw_paths = [raw_dir / f"{feed_name}_{stamp}.json" for feed_name in FEED_NAMES]
        manifest_path = evidence_dir / f"gbfs_capture_{stamp}.json"
        if not any(path.exists() for path in raw_paths) and not manifest_path.exists():
            break
        captured_at += timedelta(microseconds=1)

    discovery, discovery_status, _ = fetch_json(
        discovery_url, retries=retries, timeout=timeout
    )
    discovered = _discovered_feeds(discovery)
    missing = [name for name in FEED_NAMES if name not in discovered]
    if missing:
        raise RuntimeError("Discovery feed is missing: " + ", ".join(missing))

    downloaded: dict[str, tuple[dict[str, Any], int, bytes]] = {}
    for feed_name in FEED_NAMES:
        payload, status, content = fetch_json(
            discovered[feed_name], retries=retries, timeout=timeout
        )
        feed_data = payload.get("data")
        if not isinstance(feed_data, dict):
            raise TypeError(f"Feed {feed_name} does not contain a data object")
        stations = feed_data.get("stations")
        if not isinstance(stations, list):
            raise TypeError(f"Feed {feed_name} does not contain a data.stations list")
        downloaded[feed_name] = (payload, status, content)

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
    staged_paths: list[Path] = []
    final_paths: list[Path] = []
    try:
        for feed_name, (payload, status, content) in downloaded.items():
            stations = payload["data"]["stations"]
            output_path = raw_dir / f"{feed_name}_{stamp}.json"
            temp_path = output_path.with_suffix(".json.tmp")
            temp_path.write_bytes(content)
            staged_paths.append(temp_path)
            final_paths.append(output_path)
            try:
                raw_file_reference = output_path.resolve().relative_to(PROJECT_ROOT)
            except ValueError:
                raw_file_reference = output_path
            manifest["feeds"][feed_name] = {
                "url": discovered[feed_name],
                "http_status": status,
                "version": payload.get("version"),
                "last_updated": payload.get("last_updated"),
                "ttl": payload.get("ttl"),
                "record_count": len(stations),
                "sample_fields": sorted(stations[0].keys()) if stations else [],
                "raw_file": str(raw_file_reference),
                "raw_bytes": len(content),
                "sha256": _sha256(content),
            }

        manifest_path = evidence_dir / f"gbfs_capture_{stamp}.json"
        manifest_temp_path = manifest_path.with_suffix(".json.tmp")
        manifest_temp_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staged_paths.append(manifest_temp_path)

        for temp_path, output_path in zip(staged_paths[:2], final_paths):
            temp_path.replace(output_path)
        staged_paths[2].replace(manifest_path)
    except Exception:
        for temp_path in staged_paths:
            temp_path.unlink(missing_ok=True)
        for output_path in final_paths:
            output_path.unlink(missing_ok=True)
        raise

    return manifest_path


def _project_path(path_value: str) -> Path:
    """Resolve relative CLI paths from the repository root."""
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _run_once(args: argparse.Namespace) -> Path:
    manifest_path = capture(
        discovery_url=args.discovery_url,
        raw_dir=_project_path(args.raw_dir),
        evidence_dir=_project_path(args.evidence_dir),
        retries=args.retries,
        timeout=args.timeout,
    )
    LOGGER.info("Snapshot saved; manifest: %s", manifest_path)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-url", default=DEFAULT_DISCOVERY_URL)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--evidence-dir", default="reports/evidence")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=300,
        help="Delay between scheduled captures when --watch is enabled.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep collecting snapshots at the configured interval.",
    )
    args = parser.parse_args()
    if args.retries < 1 or args.timeout < 1 or args.interval_seconds < 1:
        parser.error("retries, timeout, and interval must be positive")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if not args.watch:
        _run_once(args)
        return

    LOGGER.info("Polling GBFS every %s seconds", args.interval_seconds)
    try:
        while True:
            cycle_started = time.monotonic()
            try:
                _run_once(args)
            except (RuntimeError, OSError, TypeError) as error:
                LOGGER.error("Capture failed; the next cycle will still run: %s", error)
            remaining = args.interval_seconds - (time.monotonic() - cycle_started)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        LOGGER.info("Polling stopped by user")


if __name__ == "__main__":
    main()
