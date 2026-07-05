"""Batch-splitting utilities for the Spotify Review Discovery Engine."""

from __future__ import annotations

from typing import Iterator

import pandas as pd


DEFAULT_BATCH_SIZE = 25


def chunk_reviews(
    reviews: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Iterator[pd.DataFrame]:
    """Yield successive *batch_size* slices of *reviews*, preserving order."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    total = len(reviews)
    for start in range(0, total, batch_size):
        yield reviews.iloc[start : start + batch_size]
