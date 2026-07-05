"""Analytics engine for the Spotify Review Discovery Engine.

Responsibilities
----------------
- Pre-clustering filter: removes off-topic / template / non-product-feedback posts
- Descriptive statistics (source counts, rating distribution, review length)
- N-gram frequency analysis (unigrams, bigrams, trigrams)
- Recommendation-keyword spotting
- Sentence-embedding + HDBSCAN clustering with per-cluster summaries
- Persists:
    output/master_reviews.csv
    output/review_clusters.json
    output/analytics_summary.json
"""

from __future__ import annotations

import json
import logging
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sentence_transformers import SentenceTransformer
import hdbscan
import umap

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# all-MiniLM-L6-v2 has a 256-token limit; truncate long posts before embedding
EMBEDDING_MAX_WORDS = 200

TOP_N_NGRAMS = 30
TOP_N_CLUSTER_KEYWORDS = 10
TOP_N_REPRESENTATIVE = 5

RECOMMENDATION_KEYWORDS = {
    "discover weekly", "daily mix", "smart shuffle", "ai dj", "radio",
    "release radar", "autoplay", "blend", "recommendation", "recommendations",
    "recommended", "suggest", "suggestion", "suggestions", "algorithm",
    "personalized", "personalised", "playlist", "playlists", "made for you",
    "discover", "discovery", "explore", "new music", "similar artists",
}

# ---------------------------------------------------------------------------
# Off-topic / template filter patterns
# ---------------------------------------------------------------------------
# These are applied only to reddit posts because app_store / google_play /
# spotify_community are already guaranteed to be product feedback.
_REPOST_PATTERNS = re.compile(
    r"i am not (?:the )?original op"
    r"|i(?:'m| am) not oop"
    r"|this is a repost sub"
    r"|\boop\b.{0,40}\bposting in\b"
    r"|published on: r/"
    r"|story timeline"
    # Sports game-thread templates (Nuggets daily schedule posts)
    r"|\bnuggets next \d"
    r"|\bnba\.com/game\b"
    # Playlist-dump templates with no prose
    r"|\[spotify\]\s*\(https"
    r"|curated by u/",
    re.IGNORECASE,
)

_NO_SPOTIFY_SOURCES = {"reddit"}  # sources where we require a Spotify mention

_MAX_WORDS_FOR_CLUSTERING = 1500  # posts longer than this are almost never product reviews
_MIN_WORDS_FOR_CLUSTERING = 5     # posts shorter than this carry too little signal

STOPWORDS: set[str] = set()       # populated lazily in _ensure_nltk_resources()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_analytics(reviews: pd.DataFrame) -> dict[str, Any]:
    """
    Compute all analytics for *reviews* and persist results to output/.

    Returns the analytics summary dict.
    """
    _ensure_nltk_resources()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nAnalytics Engine")
    print("=" * 40)

    # 1. Descriptive statistics (on full cleaned set)
    print("Computing descriptive statistics …")
    desc_stats = _descriptive_stats(reviews)

    # 2. N-gram frequencies (on full cleaned set)
    print("Extracting n-gram frequencies …")
    ngram_stats = _ngram_frequencies(reviews)

    # 3. Recommendation keyword counts (on full cleaned set)
    print("Identifying recommendation keywords …")
    rec_keywords = _recommendation_keyword_counts(reviews)

    # 4. Filter to product-feedback reviews before clustering
    print("Filtering reviews for clustering …")
    cluster_input, outlier_mask = _filter_for_clustering(reviews)
    n_outliers_pre = int(outlier_mask.sum())
    print(f"  {len(cluster_input):,} reviews pass filter  "
          f"({n_outliers_pre:,} excluded as off-topic / template)")

    # 5. Embed, reduce, cluster
    print("Generating sentence embeddings …")
    embeddings = _embed_reviews(cluster_input)

    print("Reducing dimensions …")
    reduced = _reduce_dimensions(embeddings)

    print("Clustering reviews …")
    hdb_labels = _cluster(reduced)

    # 6. Build full label array aligned to original index
    # Rows excluded by the filter get label -1 (outlier)
    all_labels = np.full(len(reviews), -1, dtype=int)
    all_labels[~outlier_mask.values] = hdb_labels

    # Remap HDBSCAN's -1 noise to continue using -1 as "outlier"
    # (they are already -1, so nothing to do)

    # 7. Cluster summaries
    print("Summarising clusters …")
    clusters = _summarise_clusters(cluster_input, embeddings, hdb_labels)

    # 8. Persist
    print("Saving outputs …")
    _save_master_csv(reviews, all_labels)
    _save_clusters_json(clusters)

    n_real_clusters = int(max(hdb_labels) + 1) if any(l >= 0 for l in hdb_labels) else 0
    n_hdb_noise = int(sum(1 for l in hdb_labels if l == -1))
    total_outliers = n_outliers_pre + n_hdb_noise

    summary = {
        "total_reviews": int(len(reviews)),
        "reviews_used_for_clustering": int(len(cluster_input)),
        "descriptive_statistics": desc_stats,
        "ngram_frequencies": ngram_stats,
        "recommendation_keywords": rec_keywords,
        "clustering": {
            "num_clusters": n_real_clusters,
            "outlier_count": total_outliers,
            "hdbscan_noise_points": n_hdb_noise,
            "prefilter_excluded": n_outliers_pre,
            "clusters": [
                {
                    "cluster_id": c["cluster_id"],
                    "size": c["size"],
                    "top_keywords": c["top_keywords"],
                    "dominant_source": c["dominant_source"],
                    "avg_review_length_words": c["avg_review_length_words"],
                }
                for c in clusters
            ],
        },
    }

    _save_analytics_summary(summary)
    _print_analytics_summary(summary, clusters)
    return summary


# ---------------------------------------------------------------------------
# Pre-clustering filter
# ---------------------------------------------------------------------------

def _filter_for_clustering(reviews: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return (kept_df, outlier_bool_mask_aligned_to_reviews_index).

    A review is excluded (marked outlier) when ANY of:
    - Fewer than _MIN_WORDS_FOR_CLUSTERING words
    - More than _MAX_WORDS_FOR_CLUSTERING words
    - Matches a known repost / template pattern
    - Comes from a variable-quality source AND contains no Spotify mention
    """
    lens = reviews["review_text"].str.split().str.len()

    too_short = lens < _MIN_WORDS_FOR_CLUSTERING
    too_long  = lens > _MAX_WORDS_FOR_CLUSTERING

    is_repost = reviews["review_text"].str.contains(_REPOST_PATTERNS, regex=True)

    from_variable_source = reviews["source"].isin(_NO_SPOTIFY_SOURCES)
    no_spotify_mention = ~reviews["review_text"].str.contains(
        r"\bspotify\b", flags=re.IGNORECASE, regex=True
    )
    off_topic = from_variable_source & no_spotify_mention

    outlier_mask = too_short | too_long | is_repost | off_topic

    return reviews[~outlier_mask].copy(), outlier_mask


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def _descriptive_stats(reviews: pd.DataFrame) -> dict[str, Any]:
    df = reviews.copy()
    df["review_length"] = df["review_text"].str.split().str.len()

    source_counts = df["source"].value_counts().to_dict()

    rating_col = df["rating"].dropna()
    if not rating_col.empty:
        rating_dist = {
            str(k): int(v)
            for k, v in df["rating"].value_counts().sort_index().items()
        }
        rating_stats: dict[str, Any] = {
            "mean": round(float(rating_col.mean()), 2),
            "median": round(float(rating_col.median()), 2),
            "std": round(float(rating_col.std()), 2),
            "distribution": rating_dist,
        }
    else:
        rating_stats = {"mean": None, "median": None, "std": None, "distribution": {}}

    lc = df["review_length"]
    length_stats: dict[str, Any] = {
        "mean": round(float(lc.mean()), 1),
        "median": round(float(lc.median()), 1),
        "min": int(lc.min()),
        "max": int(lc.max()),
        "percentile_25": round(float(lc.quantile(0.25)), 1),
        "percentile_75": round(float(lc.quantile(0.75)), 1),
    }

    return {
        "reviews_per_source": {str(k): int(v) for k, v in source_counts.items()},
        "rating": rating_stats,
        "review_length_words": length_stats,
    }


# ---------------------------------------------------------------------------
# N-gram frequencies
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[" + re.escape(string.punctuation) + r"]", " ", text)
    tokens = word_tokenize(text)
    return [t for t in tokens if t.isalpha() and t not in STOPWORDS and len(t) > 1]


def _ngram_frequencies(reviews: pd.DataFrame) -> dict[str, list[dict]]:
    all_tokens: list[list[str]] = reviews["review_text"].apply(_tokenize).tolist()

    flat = [tok for tokens in all_tokens for tok in tokens]
    bigram_list  = [bg for tokens in all_tokens for bg in ngrams(tokens, 2)]
    trigram_list = [tg for tokens in all_tokens for tg in ngrams(tokens, 3)]

    def _top(counter: Counter, n: int) -> list[dict]:
        return [
            {"term": " ".join(term) if isinstance(term, tuple) else term, "count": cnt}
            for term, cnt in counter.most_common(n)
        ]

    return {
        "unigrams": _top(Counter(flat), TOP_N_NGRAMS),
        "bigrams":  _top(Counter(bigram_list),  TOP_N_NGRAMS),
        "trigrams": _top(Counter(trigram_list), TOP_N_NGRAMS),
    }


# ---------------------------------------------------------------------------
# Recommendation keyword counts
# ---------------------------------------------------------------------------

def _recommendation_keyword_counts(reviews: pd.DataFrame) -> list[dict]:
    texts = reviews["review_text"].str.lower()
    counts: dict[str, int] = {}
    for kw in sorted(RECOMMENDATION_KEYWORDS):
        pattern = re.compile(r"\b" + re.escape(kw) + r"\b")
        count = int(texts.apply(lambda t: bool(pattern.search(t))).sum())
        if count > 0:
            counts[kw] = count

    return [
        {"keyword": kw, "reviews_containing": cnt}
        for kw, cnt in sorted(counts.items(), key=lambda x: -x[1])
    ]


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def _truncate_for_embedding(text: str) -> str:
    """Truncate to EMBEDDING_MAX_WORDS words to stay within model token limits."""
    words = text.split()
    if len(words) > EMBEDDING_MAX_WORDS:
        return " ".join(words[:EMBEDDING_MAX_WORDS])
    return text


def _embed_reviews(reviews: pd.DataFrame) -> np.ndarray:
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = reviews["review_text"].apply(_truncate_for_embedding).tolist()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    return np.array(embeddings)


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def _reduce_dimensions(embeddings: np.ndarray) -> np.ndarray:
    n = len(embeddings)
    # n_neighbors: larger = more global structure preserved, helps separate themes
    n_neighbors = min(30, n - 1)
    reducer = umap.UMAP(
        n_components=15,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _cluster(reduced: np.ndarray) -> np.ndarray:
    # min_cluster_size=8  → allows more granular clusters
    # min_samples=3       → reduces noise sensitivity, keeps small but coherent clusters
    # cluster_selection_method="leaf" → produces more clusters than "eom"
    # cluster_selection_epsilon=0.3   → merges only very close leaf clusters
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=8,
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="leaf",
        cluster_selection_epsilon=0.3,
    )
    return clusterer.fit_predict(reduced)


# ---------------------------------------------------------------------------
# Cluster summaries
# ---------------------------------------------------------------------------

def _summarise_clusters(
    reviews: pd.DataFrame,
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> list[dict]:
    unique_labels = sorted(set(labels))
    clusters = []

    for label in unique_labels:
        if label == -1:
            continue

        mask = labels == label
        cluster_reviews    = reviews[mask].copy()
        cluster_embeddings = embeddings[mask]

        # Representative reviews: closest to centroid
        centroid  = cluster_embeddings.mean(axis=0)
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closest   = np.argsort(distances)[:TOP_N_REPRESENTATIVE]
        representative_reviews = cluster_reviews.iloc[closest]["review_text"].tolist()

        # Keywords
        tokens_all: list[str] = []
        for text in cluster_reviews["review_text"]:
            tokens_all.extend(_tokenize(text))
        top_kw = [term for term, _ in Counter(tokens_all).most_common(TOP_N_CLUSTER_KEYWORDS)]

        # Average review length
        avg_len = round(float(cluster_reviews["review_text"].str.split().str.len().mean()), 1)

        # Dominant source
        dominant_source = cluster_reviews["source"].value_counts().idxmax()

        clusters.append({
            "cluster_id": int(label),
            "size": int(mask.sum()),
            "top_keywords": top_kw,
            "representative_reviews": representative_reviews,
            "avg_review_length_words": avg_len,
            "dominant_source": str(dominant_source),
        })

    clusters.sort(key=lambda c: -c["size"])
    return clusters


# ---------------------------------------------------------------------------
# Output persistence
# ---------------------------------------------------------------------------

def _save_master_csv(reviews: pd.DataFrame, labels: np.ndarray) -> None:
    out = reviews.copy()
    out["cluster_id"] = labels
    path = OUTPUT_DIR / "master_reviews.csv"
    out.to_csv(path, index=True, encoding="utf-8")
    logger.info("Saved %s", path)
    print(f"  Saved {path.name} ({len(out):,} rows)")


def _save_clusters_json(clusters: list[dict]) -> None:
    path = OUTPUT_DIR / "review_clusters.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(clusters, fh, ensure_ascii=False, indent=2)
    logger.info("Saved %s", path)
    print(f"  Saved {path.name} ({len(clusters)} clusters)")


def _save_analytics_summary(summary: dict) -> None:
    path = OUTPUT_DIR / "analytics_summary.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    logger.info("Saved %s", path)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def _print_analytics_summary(summary: dict, clusters: list[dict]) -> None:
    print("\n" + "=" * 40)
    print("Analytics Summary")
    print("=" * 40)

    print(f"Total reviews          : {summary['total_reviews']:,}")
    print(f"Used for clustering    : {summary['reviews_used_for_clustering']:,}")

    print("\nReviews per source:")
    for src, cnt in summary["descriptive_statistics"]["reviews_per_source"].items():
        print(f"  {src}: {cnt:,}")

    rating = summary["descriptive_statistics"]["rating"]
    if rating["mean"] is not None:
        print(f"\nRating  mean={rating['mean']}  median={rating['median']}  std={rating['std']}")

    length = summary["descriptive_statistics"]["review_length_words"]
    print(
        f"Length (words)  mean={length['mean']}  "
        f"median={length['median']}  min={length['min']}  max={length['max']}"
    )

    print("\nTop 10 unigrams:")
    for item in summary["ngram_frequencies"]["unigrams"][:10]:
        print(f"  {item['term']}: {item['count']}")

    print("\nTop 10 bigrams:")
    for item in summary["ngram_frequencies"]["bigrams"][:10]:
        print(f"  {item['term']}: {item['count']}")

    cl = summary["clustering"]
    sizes = [c["size"] for c in clusters]
    print(f"\nClusters found         : {cl['num_clusters']}")
    print(f"Outlier count          : {cl['outlier_count']:,}  "
          f"(pre-filter: {cl['prefilter_excluded']}, "
          f"HDBSCAN noise: {cl['hdbscan_noise_points']})")
    if sizes:
        print(f"Largest cluster        : {max(sizes):,}")
        print(f"Average cluster size   : {sum(sizes)/len(sizes):.1f}")
        median_size = float(pd.Series(sizes).median())
        print(f"Median cluster size    : {median_size:.1f}")

    print(f"\nAll {len(clusters)} clusters (sorted by size):")
    for c in clusters:
        kw  = ", ".join(c["top_keywords"][:6])
        src = c["dominant_source"]
        avg = c["avg_review_length_words"]
        print(
            f"  Cluster {c['cluster_id']:>3}  "
            f"({c['size']:>4} reviews)  "
            f"avg={avg:>6} words  "
            f"src={src:<15}  [{kw}]"
        )


# ---------------------------------------------------------------------------
# NLTK bootstrap
# ---------------------------------------------------------------------------

def _ensure_nltk_resources() -> None:
    import nltk
    global STOPWORDS

    for resource in ("punkt", "punkt_tab", "stopwords"):
        try:
            nltk.data.find(
                "corpora/stopwords" if resource == "stopwords"
                else f"tokenizers/{resource}"
            )
        except LookupError:
            nltk.download(resource, quiet=True)

    STOPWORDS = set(stopwords.words("english"))
