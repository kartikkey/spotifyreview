"""Application entry point for Spotify Review Discovery Engine."""

from __future__ import annotations

import argparse
import logging

from src.loader import load_reviews
from src.preprocessor import preprocess_reviews


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spotify Review Discovery Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--analyse",
        action="store_true",
        help="Run the Gemini AI analysis layer after preprocessing.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: analyse only the first 5 reviews (implies --analyse).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        metavar="N",
        help="Number of reviews per Gemini API call.",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        metavar="MODEL",
        help="Gemini model identifier.",
    )
    parser.add_argument(
        "--analytics",
        action="store_true",
        help="Run the analytics engine (clustering, n-grams, statistics).",
    )
    parser.add_argument(
        "--cluster-analysis",
        action="store_true",
        help="Run the Gemini cluster analyser (reads output/review_clusters.json).",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    print("Spotify Review Discovery Engine")
    print("=" * 40)

    reviews = load_reviews()
    cleaned_reviews = preprocess_reviews(reviews)

    print(f"\nRaw DataFrame    : {reviews.shape[0]} rows × {reviews.shape[1]} columns")
    print(f"Cleaned DataFrame: {cleaned_reviews.shape[0]} rows × {cleaned_reviews.shape[1]} columns")

    if args.analytics:
        print("\n" + "=" * 40)
        print("Analytics Engine")
        print("=" * 40)
        from src.analytics import run_analytics
        run_analytics(cleaned_reviews)

    if args.cluster_analysis:
        print("\n" + "=" * 40)
        print("Cluster Analysis")
        print("=" * 40)
        from src.cluster_analyser import run_cluster_analysis
        run_cluster_analysis(model_name=args.model)

    if args.analyse or args.test:
        print("\n" + "=" * 40)
        print("AI Analysis")
        print("=" * 40)
        from src.gemini_client import run_analysis  # deferred — avoids dotenv load on import

        run_analysis(
            cleaned_reviews,
            batch_size=args.batch_size,
            test_mode=args.test,
            model_name=args.model,
        )


if __name__ == "__main__":
    main()
