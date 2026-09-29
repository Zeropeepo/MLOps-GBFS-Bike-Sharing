"""Ambil snapshot feed GBFS Citi Bike dan simpan sebagai file JSON."""

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

ROOT_DIR = Path(__file__).resolve().parents[1]
DISCOVERY_URL = "https://gbfs.citibikenyc.com/gbfs/2.3/gbfs.json"
FEED_NAMES = ("station_information", "station_status")
LOGGER = logging.getLogger(__name__)


def get_json(url: str, retries: int, timeout: int) -> tuple[dict[str, Any], int, bytes]:
    """Ambil JSON lewat HTTP dan coba ulang jika koneksi atau respons gagal."""
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()

            data = response.json()
            if not isinstance(data, dict):
                raise TypeError("Isi JSON harus berupa object")

            return data, response.status_code, response.content
        except (requests.RequestException, ValueError) as error:
            if attempt == retries:
                raise RuntimeError(f"Gagal mengambil {url}: {error}") from error

            delay = attempt * 2
            LOGGER.warning("Request gagal; mencoba lagi dalam %s detik", delay)
            time.sleep(delay)

    raise RuntimeError(f"Gagal mengambil {url}")


def get_feed_urls(discovery: dict[str, Any]) -> dict[str, str]:
    """Cari URL station_information dan station_status di discovery feed."""
    data = discovery.get("data", {})
    if not isinstance(data, dict):
        raise TypeError("Discovery feed tidak memiliki object data")

    # GBFS v2.3 menaruh daftar feed di bawah kode bahasa, biasanya 'en'.
    language_data = data.get("en", {})
    if not isinstance(language_data, dict):
        raise TypeError("Discovery feed tidak memiliki data bahasa 'en'")

    feeds = language_data.get("feeds", [])
    urls = {
        feed["name"]: feed["url"]
        for feed in feeds
        if isinstance(feed, dict) and "name" in feed and "url" in feed
    }

    missing = [name for name in FEED_NAMES if name not in urls]
    if missing:
        raise ValueError(f"Feed tidak ditemukan: {', '.join(missing)}")
    return urls


def next_timestamp(raw_dir: Path, evidence_dir: Path) -> tuple[datetime, str]:
    """Buat timestamp UTC yang belum dipakai file snapshot sebelumnya."""
    captured_at = datetime.now(timezone.utc)

    while True:
        stamp = captured_at.strftime("%Y%m%dT%H%M%S%fZ")
        raw_files = [raw_dir / f"{name}_{stamp}.json" for name in FEED_NAMES]
        manifest = evidence_dir / f"gbfs_capture_{stamp}.json"
        if not any(path.exists() for path in raw_files) and not manifest.exists():
            return captured_at, stamp
        captured_at += timedelta(microseconds=1)


def save_snapshot(
    discovery_url: str,
    raw_dir: Path,
    evidence_dir: Path,
    retries: int = 3,
    timeout: int = 20,
) -> Path:
    """Unduh kedua feed, simpan file mentah, lalu tulis manifest pengambilan."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    captured_at, stamp = next_timestamp(raw_dir, evidence_dir)

    discovery, discovery_status, _ = get_json(discovery_url, retries, timeout)
    feed_urls = get_feed_urls(discovery)

    # Ambil kedua feed lebih dulu agar error koneksi tidak meninggalkan pasangan
    # snapshot yang hanya berisi satu file.
    downloaded: dict[str, tuple[dict[str, Any], int, bytes]] = {}
    for name in FEED_NAMES:
        payload, status, content = get_json(feed_urls[name], retries, timeout)
        feed_data = payload.get("data", {})
        stations = feed_data.get("stations") if isinstance(feed_data, dict) else None
        if not isinstance(stations, list):
            raise TypeError(f"Feed {name} tidak memiliki daftar data.stations")
        downloaded[name] = payload, status, content

    manifest: dict[str, Any] = {
        "captured_at_utc": captured_at.isoformat(),
        "discovery": {
            "url": discovery_url,
            "http_status": discovery_status,
            "version": discovery.get("version"),
        },
        "feeds": {},
    }

    for name, (payload, status, content) in downloaded.items():
        stations = payload["data"]["stations"]
        raw_path = raw_dir / f"{name}_{stamp}.json"
        raw_path.write_bytes(content)

        try:
            display_path = raw_path.resolve().relative_to(ROOT_DIR)
        except ValueError:
            display_path = raw_path

        manifest["feeds"][name] = {
            "url": feed_urls[name],
            "http_status": status,
            "version": payload.get("version"),
            "last_updated": payload.get("last_updated"),
            "record_count": len(stations),
            "raw_file": str(display_path),
            "raw_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    manifest_path = evidence_dir / f"gbfs_capture_{stamp}.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def project_path(path: str) -> Path:
    """Jadikan path relatif mengarah dari folder utama repository."""
    result = Path(path).expanduser()
    return result if result.is_absolute() else ROOT_DIR / result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-url", default=DISCOVERY_URL)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--evidence-dir", default="reports/evidence")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument(
        "--watch", action="store_true", help="Ulangi capture sesuai interval."
    )
    args = parser.parse_args()

    if args.retries < 1 or args.timeout < 1 or args.interval_seconds < 1:
        parser.error("retries, timeout, dan interval harus lebih dari nol")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    while True:
        started = time.monotonic()
        try:
            manifest = save_snapshot(
                args.discovery_url,
                project_path(args.raw_dir),
                project_path(args.evidence_dir),
                args.retries,
                args.timeout,
            )
            LOGGER.info("Snapshot tersimpan: %s", manifest)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            LOGGER.error("Capture gagal: %s", error)
            if not args.watch:
                raise SystemExit(1) from error

        if not args.watch:
            break

        wait_seconds = args.interval_seconds - (time.monotonic() - started)
        if wait_seconds > 0:
            try:
                time.sleep(wait_seconds)
            except KeyboardInterrupt:
                LOGGER.info("Polling dihentikan")
                break


if __name__ == "__main__":
    main()
