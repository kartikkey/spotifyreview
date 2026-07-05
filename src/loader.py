"""Data ingestion utilities for Spotify Review Discovery Engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

SCHEMA = [
    "source",
    "country",
    "rating",
    "date",
    "review_text",
    "review_title",
    "author",
]

# Maps filename prefix → (source_label, normalizer_key)
_PREFIX_MAP = {
    "playstore": ("google_play", "google_play"),
    "reddit": ("reddit", "reddit"),
    "appstore_india": ("app_store", "app_store"),
    "appstore_us": ("app_store", "app_store"),
    "spotify_community": ("spotify_community", "spotify_community"),
}


def _detect_source(file_name: str) -> tuple[str, str] | None:
    """Return (source_label, normalizer_key) for a filename, or None if unknown."""
    stem = file_name.lower()
    # Longest-prefix-first to avoid 'appstore' matching before 'appstore_india'
    for prefix in sorted(_PREFIX_MAP, key=len, reverse=True):
        if stem.startswith(prefix):
            return _PREFIX_MAP[prefix]
    return None


def load_reviews(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Discover, load, and normalize all review datasets under *data_dir*."""
    data_path = Path(data_dir)

    json_files = sorted(data_path.glob("*.json"))
    print(f"\nFiles discovered ({len(json_files)}):")
    for f in json_files:
        print(f"  {f.name}")
    print()

    frames: list[pd.DataFrame] = []
    records_per_file: dict[str, int] = {}

    for file_path in json_files:
        mapping = _detect_source(file_path.name)
        if mapping is None:
            print(f"  [SKIP] {file_path.name} — unknown prefix, skipping")
            continue

        source_label, normalizer_key = mapping
        records = _read_json_records(file_path)
        normalized = _normalize(records, source_label, normalizer_key)
        df = pd.DataFrame(normalized, columns=SCHEMA)
        frames.append(df)
        records_per_file[file_path.name] = len(df)

    print("Records loaded per file:")
    for fname, count in records_per_file.items():
        print(f"  {fname}: {count}")
    print()

    reviews = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SCHEMA)
    _print_load_summary(reviews)
    return reviews


def _read_json_records(file_path: Path) -> list[dict[str, Any]]:
    if not file_path.exists():
        raise FileNotFoundError(f"Required data file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    records = _extract_records(payload)
    return [r for r in records if isinstance(r, dict)]


def _extract_records(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("reviews", "data", "posts", "items", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return value

    raise ValueError("JSON payload must be a list or contain a supported records key.")


def _normalize(
    records: list[dict[str, Any]],
    source_label: str,
    normalizer_key: str,
) -> list[dict[str, Any]]:
    if normalizer_key == "google_play":
        return _normalize_google_play(records, source_label)
    if normalizer_key == "reddit":
        return _normalize_reddit(records, source_label)
    if normalizer_key == "app_store":
        return _normalize_app_store(records, source_label)
    if normalizer_key == "spotify_community":
        return _normalize_spotify_community(records, source_label)
    raise ValueError(f"Unsupported normalizer key: {normalizer_key}")


def _normalize_google_play(
    records: list[dict[str, Any]], source: str
) -> list[dict[str, Any]]:
    return [
        {
            "source": source,
            "country": r.get("country"),
            "rating": r.get("rating"),
            "date": r.get("date"),
            "review_text": r.get("body"),
            "review_title": r.get("title"),
            "author": r.get("reviewer"),
        }
        for r in records
    ]


def _normalize_reddit(
    records: list[dict[str, Any]], source: str
) -> list[dict[str, Any]]:
    return [
        {
            "source": source,
            "country": r.get("country"),
            "rating": r.get("rating"),
            "date": r.get("createdAt"),
            "review_text": r.get("body"),
            "review_title": r.get("title"),
            "author": r.get("authorName"),
        }
        for r in records
    ]


def _normalize_app_store(
    records: list[dict[str, Any]], source: str
) -> list[dict[str, Any]]:
    return [
        {
            "source": source,
            "country": r.get("country"),
            "rating": r.get("score"),
            "date": r.get("date"),
            "review_text": r.get("text"),
            "review_title": r.get("title"),
            "author": r.get("userName"),
        }
        for r in records
    ]


def _normalize_spotify_community(
    records: list[dict[str, Any]], source: str
) -> list[dict[str, Any]]:
    return [
        {
            "source": source,
            "country": r.get("country"),
            "rating": r.get("rating"),
            "date": r.get("date") or r.get("createdAt"),
            "review_text": r.get("original_post") or r.get("body") or r.get("text") or r.get("review_text"),
            "review_title": r.get("title") or r.get("review_title"),
            "author": r.get("author") or r.get("userName") or r.get("authorName"),
        }
        for r in records
    ]


def _print_load_summary(reviews: pd.DataFrame) -> None:
    print(f"Total raw records loaded: {len(reviews)}")
    print("Records per source:")
    source_counts = reviews["source"].value_counts().sort_index()
    for source, count in source_counts.items():
        print(f"  {source}: {count}")
    print()
