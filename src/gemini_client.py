"""Gemini API client for the Spotify Review Discovery Engine."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.prompts import SYSTEM_PROMPT


load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "ai_analysis"

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0   # seconds; doubles on each retry
BATCH_DELAY = 1.0        # seconds between successful batch calls


class GeminiClient:
    """Wraps the Google GenAI SDK for structured review analysis."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    ) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file and try again."
            )

        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._gen_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
        )

        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyse_batch(self, batch: pd.DataFrame, batch_index: int) -> list[dict]:
        """Send one *batch* to Gemini, persist the result, and return parsed JSON."""
        payload = _dataframe_to_payload(batch)
        raw_json = self._call_with_retry(payload, batch_index)
        results = _parse_response(raw_json, batch_index)
        self._save_batch(results, batch_index)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_with_retry(self, payload: str, batch_index: int) -> str:
        delay = RETRY_BASE_DELAY
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(
                    "Batch %03d — attempt %d/%d", batch_index, attempt, MAX_RETRIES
                )
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=payload,
                    config=self._gen_config,
                )
                return response.text
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Batch %03d — attempt %d failed: %s", batch_index, attempt, exc
                )
                if attempt == MAX_RETRIES:
                    raise
                logger.info("Retrying in %.1f s …", delay)
                time.sleep(delay)
                delay *= 2

        raise RuntimeError("Unreachable")  # pragma: no cover

    def _save_batch(self, results: list[dict], batch_index: int) -> Path:
        file_path = self._output_dir / f"batch_{batch_index:03d}.json"
        with file_path.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        logger.info(
            "Batch %03d — saved %d records → %s", batch_index, len(results), file_path
        )
        return file_path


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _dataframe_to_payload(batch: pd.DataFrame) -> str:
    """Serialise a DataFrame batch into the JSON string sent to Gemini."""
    records = []
    for _, row in batch.iterrows():
        records.append(
            {
                "review_id": str(row.name),
                "source": str(row.get("source", "")),
                "review_title": str(row.get("review_title", "") or ""),
                "review_text": str(row.get("review_text", "") or ""),
                "rating": row.get("rating"),
                "country": str(row.get("country", "") or ""),
                "date": str(row.get("date", "") or ""),
            }
        )
    return json.dumps(records, ensure_ascii=False)


def _parse_response(raw: str, batch_index: int) -> list[dict]:
    """Extract and validate the JSON array from Gemini's response text."""
    text = raw.strip()

    # Strip accidental markdown fences
    if text.startswith("```"):
        lines = text.splitlines()
        inner = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(inner).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Batch {batch_index:03d}: Gemini returned invalid JSON — {exc}\n"
            f"Raw response (first 500 chars):\n{raw[:500]}"
        ) from exc

    if not isinstance(parsed, list):
        raise ValueError(
            f"Batch {batch_index:03d}: expected a JSON array, "
            f"got {type(parsed).__name__}"
        )

    return parsed


# ---------------------------------------------------------------------------
# Orchestration entry point
# ---------------------------------------------------------------------------

def run_analysis(
    reviews: pd.DataFrame,
    *,
    batch_size: int = 25,
    test_mode: bool = False,
    model_name: str = DEFAULT_MODEL,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> list[dict]:
    """
    Orchestrate chunking → Gemini analysis → persistence for *reviews*.

    Parameters
    ----------
    reviews:     Cleaned DataFrame from the preprocessing pipeline.
    batch_size:  Number of reviews per Gemini call.
    test_mode:   When True, process only the first 5 reviews (1 batch).
    model_name:  Gemini model identifier.
    output_dir:  Directory where batch JSON files are written.
    """
    from src.chunker import chunk_reviews  # local import avoids circular deps at module load

    client = GeminiClient(model_name=model_name, output_dir=output_dir)

    if test_mode:
        reviews = reviews.head(5)
        print("TEST MODE — processing first 5 reviews only.\n")

    batches = list(chunk_reviews(reviews, batch_size=batch_size))
    total_batches = len(batches)
    print(f"Reviews to analyse : {len(reviews)}")
    print(f"Batch size         : {batch_size}")
    print(f"Total batches      : {total_batches}\n")

    all_results: list[dict] = []

    for idx, batch in enumerate(batches, start=1):
        print(
            f"Processing batch {idx:03d}/{total_batches:03d} "
            f"({len(batch)} reviews) …",
            end=" ",
            flush=True,
        )
        results = client.analyse_batch(batch, batch_index=idx)
        all_results.extend(results)
        print(f"done  ({len(results)} records)")

        if idx < total_batches:
            time.sleep(BATCH_DELAY)

    print(
        f"\nAnalysis complete. {len(all_results)} records written "
        f"to {Path(output_dir)}."
    )
    return all_results
