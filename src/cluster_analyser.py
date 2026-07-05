"""Cluster-level AI analysis for the Spotify Review Discovery Engine.

Sends each review cluster to Gemini as a single request and extracts
strategic Product Management insights. One API call per cluster.

Input  : output/review_clusters.json
Output : output/cluster_insights.json
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLUSTERS_PATH   = PROJECT_ROOT / "output" / "review_clusters.json"
INSIGHTS_PATH   = PROJECT_ROOT / "output" / "cluster_insights.json"

DEFAULT_MODEL      = "gemini-2.5-flash"
MAX_RETRIES        = 3
RETRY_BASE_DELAY   = 2.0   # seconds; doubles on each attempt
INTER_CLUSTER_DELAY = 1.5  # seconds between successful calls

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a Senior Product Manager at Spotify with deep expertise in music discovery,
personalization algorithms, and user experience research.

You will receive a summary of a cluster of user reviews. The cluster was identified
by grouping semantically similar reviews from multiple sources (App Store, Google Play,
Reddit, Spotify Community).

Your task is to analyze this cluster as a whole and extract high-quality, actionable
Product Management insights that can feed directly into a product roadmap.

STRICT OUTPUT RULES
- Return ONLY a single valid JSON object. No markdown. No code fences. No commentary.
- Never hallucinate. Base every field strictly on the evidence in the cluster.
- If a field cannot be confidently inferred, return an empty string "" or empty array [].

REQUIRED OUTPUT SCHEMA:
{
  "cluster_id": <integer — copy from input>,
  "theme": "<short noun-phrase naming the dominant theme of this cluster>",
  "problem_statement": "<one or two sentences stating the core user problem this cluster represents>",
  "root_cause_hypothesis": "<one or two sentences on the likely product or system-level cause>",
  "affected_users": "<description of the user segment most affected: free, premium, power users, etc.>",
  "evidence": ["<direct quote or paraphrase from a representative review>", ...],
  "product_opportunities": ["<specific product improvement opportunity>", ...],
  "recommended_features": ["<concrete feature or change that would address the problem>", ...],
  "priority": "<critical | high | medium | low>",
  "impact": "<high | medium | low>",
  "effort": "<high | medium | low>",
  "success_metrics": ["<measurable metric that would confirm the problem is solved>", ...],
  "confidence": <float 0.0–1.0 — your confidence in this analysis given the evidence>
}

FIELD GUIDANCE
theme               : 3–6 words, e.g. "Repetitive Recommendations", "AI DJ Satisfaction", "Premium Value Concerns"
problem_statement   : state the problem from the user's perspective, not a list of complaints
root_cause_hypothesis : focus on system/product behavior, not user frustration
affected_users      : be specific — mention subscription tier, use-case, or listening behavior when inferable
evidence            : include 3–5 of the most illustrative quotes or paraphrases from the reviews
product_opportunities : list 2–5 concrete opportunities, ordered by potential impact
recommended_features  : list 2–5 specific features or product changes
priority            : based on cluster size, severity of complaints, and breadth of user impact
impact              : expected improvement in user satisfaction, retention, or engagement
effort              : engineering and design effort estimate relative to Spotify's scale
success_metrics     : specific, measurable outcomes (e.g. "increase Discover Weekly save rate by 15%")
confidence          : 0.0 if very few or ambiguous reviews, 1.0 if large consistent cluster
"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_cluster_analysis(
    *,
    clusters_path: Path = CLUSTERS_PATH,
    insights_path: Path = INSIGHTS_PATH,
    model_name: str = DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    """
    Analyse every cluster in *clusters_path* with Gemini and write
    the combined insights to *insights_path*.

    Skips clusters whose cluster_id already appears in *insights_path*
    so interrupted runs can be resumed safely.
    """
    clusters = _load_clusters(clusters_path)
    existing = _load_existing_insights(insights_path)
    already_done = {r["cluster_id"] for r in existing}

    pending = [c for c in clusters if c["cluster_id"] not in already_done]

    print(f"\nCluster Analysis")
    print("=" * 40)
    print(f"Total clusters    : {len(clusters)}")
    print(f"Already analysed  : {len(already_done)}")
    print(f"To process        : {len(pending)}\n")

    if not pending:
        print("Nothing to do — all clusters already analysed.")
        return existing

    client = _make_client(model_name)
    results = list(existing)  # start with whatever already exists

    for i, cluster in enumerate(pending, start=1):
        cid  = cluster["cluster_id"]
        size = cluster["size"]
        print(
            f"[{i:>2}/{len(pending)}]  Cluster {cid:>3}  "
            f"({size:>4} reviews)  …",
            end=" ",
            flush=True,
        )

        payload = _build_payload(cluster)
        raw     = _call_with_retry(client, model_name, payload, cluster_id=cid)
        insight = _parse_response(raw, cluster_id=cid)

        # Guarantee cluster_id is always an int copied from input
        insight["cluster_id"] = cid

        results.append(insight)
        _save_insights(results, insights_path)

        print(f"done  [{insight.get('theme', '—')}]")
        logger.info("Cluster %d analysed: theme=%r", cid, insight.get("theme"))

        if i < len(pending):
            time.sleep(INTER_CLUSTER_DELAY)

    print(f"\nCluster analysis complete. {len(results)} insights -> {insights_path}")
    _print_summary(results)
    return results


# ---------------------------------------------------------------------------
# Gemini interaction
# ---------------------------------------------------------------------------

def _make_client(model_name: str) -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Add it to your .env file and try again."
        )
    return genai.Client(api_key=api_key)


def _call_with_retry(
    client: genai.Client,
    model_name: str,
    payload: str,
    cluster_id: int,
) -> str:
    config = types.GenerateContentConfig(
        system_instruction=_SYSTEM_PROMPT,
        temperature=0.2,
    )
    delay = RETRY_BASE_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=payload,
                config=config,
            )
            return response.text
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Cluster %d — attempt %d/%d failed: %s",
                cluster_id, attempt, MAX_RETRIES, exc,
            )
            if attempt == MAX_RETRIES:
                raise
            logger.info("Retrying in %.1f s …", delay)
            time.sleep(delay)
            delay *= 2

    raise RuntimeError("Unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

def _build_payload(cluster: dict[str, Any]) -> str:
    """Serialise a cluster dict into the prompt payload sent to Gemini."""
    payload = {
        "cluster_id": cluster["cluster_id"],
        "cluster_size": cluster["size"],
        "dominant_source": cluster["dominant_source"],
        "avg_review_length_words": cluster["avg_review_length_words"],
        "top_keywords": cluster["top_keywords"],
        "representative_reviews": cluster["representative_reviews"],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_response(raw: str, cluster_id: int) -> dict[str, Any]:
    text = raw.strip()

    # Strip accidental markdown fences
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Cluster {cluster_id}: Gemini returned invalid JSON — {exc}\n"
            f"Raw response (first 500 chars):\n{raw[:500]}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            f"Cluster {cluster_id}: expected a JSON object, got {type(parsed).__name__}"
        )

    return parsed


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _load_clusters(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Cluster file not found: {path}\n"
            "Run --analytics first to generate review_clusters.json."
        )
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_existing_insights(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []


def _save_insights(insights: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(insights, fh, ensure_ascii=False, indent=2)
    logger.info("Saved %d insights -> %s", len(insights), path)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _print_summary(insights: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 40)
    print("Cluster Insights Summary")
    print("=" * 40)
    for ins in sorted(insights, key=lambda x: x.get("cluster_id", 0)):
        cid      = ins.get("cluster_id", "?")
        theme    = ins.get("theme", "—")
        priority = ins.get("priority", "—")
        impact   = ins.get("impact", "—")
        effort   = ins.get("effort", "—")
        conf     = ins.get("confidence", 0.0)
        print(
            f"  Cluster {cid:>3}  priority={priority:<8}  "
            f"impact={impact:<6}  effort={effort:<6}  "
            f"conf={conf:.2f}  [{theme}]"
        )
