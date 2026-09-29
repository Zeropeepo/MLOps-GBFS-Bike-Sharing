"""Backward-compatible module path for :mod:`src.ingest_data`."""

from src.ingest_data import capture, fetch_json, main

__all__ = ["capture", "fetch_json", "main"]


if __name__ == "__main__":
    main()
