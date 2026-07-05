"""Preprocessing utilities for normalized Spotify review data."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect


DetectorFactory.seed = 0

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class PreprocessingSummary:
    """Counts produced while preparing review text."""

    original_record_count: int
    duplicate_reviews_removed: int
    empty_reviews_removed: int
    non_english_reviews_removed: int
    final_cleaned_record_count: int


def preprocess_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Prepare normalized reviews for later analysis."""
    original_count = len(reviews)
    cleaned = reviews.copy()

    duplicate_mask = cleaned.duplicated(subset=["review_text"], keep="first")
    duplicate_count = int(duplicate_mask.sum())
    cleaned = cleaned.loc[~duplicate_mask].copy()

    non_empty_mask = cleaned["review_text"].apply(_has_text)
    initial_empty_count = int((~non_empty_mask).sum())
    cleaned = cleaned.loc[non_empty_mask].copy()

    cleaned["review_text"] = cleaned["review_text"].apply(_clean_text)

    cleaned_non_empty_mask = cleaned["review_text"].apply(_has_text)
    cleaned_empty_count = int((~cleaned_non_empty_mask).sum())
    cleaned = cleaned.loc[cleaned_non_empty_mask].copy()

    english_mask = cleaned["review_text"].apply(_is_english)
    non_english_count = int((~english_mask).sum())
    cleaned = cleaned.loc[english_mask].reset_index(drop=True)

    summary = PreprocessingSummary(
        original_record_count=original_count,
        duplicate_reviews_removed=duplicate_count,
        empty_reviews_removed=initial_empty_count + cleaned_empty_count,
        non_english_reviews_removed=non_english_count,
        final_cleaned_record_count=len(cleaned),
    )
    cleaned.attrs["preprocessing_summary"] = summary
    print_preprocessing_summary(summary)
    return cleaned


def print_preprocessing_summary(summary: PreprocessingSummary) -> None:
    """Print preprocessing counts."""
    print("Preprocessing summary:")
    print(f"Original record count: {summary.original_record_count}")
    print(f"Duplicate reviews removed: {summary.duplicate_reviews_removed}")
    print(f"Empty reviews removed: {summary.empty_reviews_removed}")
    print(f"Non-English reviews removed: {summary.non_english_reviews_removed}")
    print(f"Final cleaned record count: {summary.final_cleaned_record_count}")


def _clean_text(value: object) -> str:
    text = html.unescape(str(value))
    text = URL_PATTERN.sub(" ", text)
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def _has_text(value: object) -> bool:
    if pd.isna(value):
        return False

    return bool(str(value).strip())


def _is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False
