"""Scrape Spotify Community discussions about discovery and recommendations."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urljoin

from bs4 import BeautifulSoup
from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "data" / "spotify_community.json"
BASE_URL = "https://community.spotify.com"
SEARCH_API_URL = f"{BASE_URL}/restapi/vc/search/messages"

SEARCH_KEYWORDS = [
    "Discover Weekly",
    "Discover Weekly recommendations",
    "Discover Weekly algorithm",
    "Smart Shuffle",
    "Smart Shuffle recommendations",
    "AI DJ",
    "AI DJ recommendations",
    "Daily Mix",
    "Spotify recommendations",
    "Recommendation algorithm",
    "Music discovery",
    "Music recommendations",
    "Recommended songs",
    "Song recommendations",
    "Playlist recommendations",
    "Recommendation quality",
    "Recommendation accuracy",
    "New music discovery",
    "Music exploration",
    "Enhance Playlist",
    "Spotify Radio",
    "Radio recommendations",
    "Suggested songs",
    "Recommendation system",
]

AVOID_TERMS = {
    "billing",
    "premium payment",
    "payment",
    "invoice",
    "login",
    "log in",
    "password",
    "account recovery",
    "family plan",
    "family plans",
    "subscription",
    "subscription management",
}

FOCUS_TERMS = {
    "discover weekly",
    "smart shuffle",
    "ai dj",
    "music discovery",
    "recommendation",
    "recommendations",
    "recommended songs",
    "daily mix",
    "enhance playlist",
    "playlist generation",
    "algorithm",
}

THREADS_PER_KEYWORD = 15
CANDIDATE_MULTIPLIER = 4   # fetch this many × limit from the API before filtering
REQUEST_DELAY_SECONDS = 1.5
RETRY_ATTEMPTS = 3
NAVIGATION_TIMEOUT_MS = 45_000

LOGGER = logging.getLogger(__name__)


@dataclass
class ScrapeResult:
    """Collected discussions and per-keyword counts."""

    discussions: list[dict[str, Any]]
    counts_by_keyword: dict[str, int]
    duplicates_skipped: int
    filtered_out: int


async def scrape_spotify_community(
    output_file: Path | str = OUTPUT_FILE,
    threads_per_keyword: int = THREADS_PER_KEYWORD,
) -> ScrapeResult:
    """Scrape Spotify Community search results and write discussions to JSON."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_urls: set[str] = set()
    discussions: list[dict[str, Any]] = []
    counts_by_keyword = {keyword: 0 for keyword in SEARCH_KEYWORDS}
    duplicates_skipped = 0
    filtered_out = 0

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/126.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1366, "height": 768},
            locale="en-US",
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )
        page = await context.new_page()
        page.set_default_timeout(NAVIGATION_TIMEOUT_MS)

        try:
            for keyword in SEARCH_KEYWORDS:
                LOGGER.info("Searching Spotify Community for %s", keyword)
                thread_urls = await collect_thread_urls(page, keyword, threads_per_keyword)

                for thread_url in thread_urls:
                    canonical_url = normalize_url(thread_url)
                    if canonical_url in seen_urls:
                        duplicates_skipped += 1
                        LOGGER.debug("Duplicate skipped: %s", canonical_url)
                        continue

                    seen_urls.add(canonical_url)
                    discussion = await scrape_thread(page, canonical_url, keyword)
                    await polite_delay()

                    if discussion is None:
                        filtered_out += 1
                        continue

                    discussions.append(discussion)
                    counts_by_keyword[keyword] += 1
        finally:
            await browser.close()

    write_json(output_path, discussions)
    print_scrape_summary(discussions, counts_by_keyword, duplicates_skipped, filtered_out, output_path)
    return ScrapeResult(
        discussions=discussions,
        counts_by_keyword=counts_by_keyword,
        duplicates_skipped=duplicates_skipped,
        filtered_out=filtered_out,
    )


async def collect_thread_urls(
    page: Page,
    keyword: str,
    limit: int,
) -> list[str]:
    """Collect candidate discussion URLs from Spotify Community search."""
    pool_size = limit * CANDIDATE_MULTIPLIER
    xml = await fetch_url_text(page, build_search_api_url(keyword, limit))
    await polite_delay()

    if not xml:
        return []

    candidates = parse_search_api_urls(xml)
    if candidates:
        LOGGER.info("API returned %s candidate URLs for %s (pool target %s)", len(candidates), keyword, pool_size)
        return candidates[:pool_size]

    search_url = build_search_url(keyword)
    html = await fetch_page_html(page, search_url)
    await polite_delay()

    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    candidates: list[str] = []

    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"])
        text = anchor.get_text(" ", strip=True)
        absolute_url = normalize_url(urljoin(BASE_URL, href))

        if not is_candidate_thread_url(absolute_url):
            continue
        if is_avoided_content(f"{text} {absolute_url}"):
            continue
        if absolute_url not in candidates:
            candidates.append(absolute_url)

        if len(candidates) >= pool_size:
            break

    LOGGER.info("HTML fallback found %s candidate URLs for %s", len(candidates), keyword)
    return candidates


async def scrape_thread(
    page: Page,
    url: str,
    keyword: str,
) -> dict[str, Any] | None:
    """Scrape one Spotify Community discussion thread."""
    try:
        html = await fetch_page_html(page, url)
        if not html:
            return None

        soup = BeautifulSoup(html, "html.parser")
        title = first_text(
            soup,
            [
                "h1",
                ".lia-message-subject",
                ".lia-thread-topic h2",
                "[data-testid='thread-title']",
                "title",
            ],
        )
        original_post = extract_original_post(soup)
        author = first_text(
            soup,
            [
                ".lia-user-name-link",
                ".UserName a",
                ".lia-message-author-username",
                "[class*='author'] a",
            ],
        )
        date = extract_date(soup)
        reply_count = extract_reply_count(soup)
        if reply_count is None:
            reply_count = await fetch_reply_count(page, url)
        labels = extract_labels(soup)

        combined_text = " ".join(
            value for value in [title, original_post, " ".join(labels)] if value
        )
        if is_avoided_content(combined_text) or not is_focused_content(keyword, combined_text):
            LOGGER.info("Skipping off-topic thread: %s", url)
            return None

        return {
            "source": "spotify_community",
            "keyword_used": keyword,
            "title": title,
            "url": url,
            "author": author,
            "date": date,
            "original_post": original_post,
            "reply_count": reply_count,
            "labels": labels,
        }
    except Exception:
        LOGGER.exception("Failed to scrape thread: %s", url)
        return None


async def fetch_page_html(page: Page, url: str) -> str | None:
    """Load a page with retries and return the rendered HTML."""
    text = await fetch_url_text(page, url)
    if text is None:
        return None

    if text.lstrip().startswith("<html"):
        return text

    return await page.content()


async def fetch_url_text(page: Page, url: str) -> str | None:
    """Load a URL with retries and return the response body text."""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=NAVIGATION_TIMEOUT_MS,
            )
            if response and response.status >= 400:
                LOGGER.warning("HTTP %s for %s", response.status, url)

            if response:
                return await response.text()

            await page.wait_for_load_state("networkidle", timeout=10_000)
            return await page.content()
        except PlaywrightTimeoutError:
            LOGGER.warning("Timeout loading %s on attempt %s", url, attempt)
        except Exception:
            LOGGER.exception("Error loading %s on attempt %s", url, attempt)

        await asyncio.sleep(REQUEST_DELAY_SECONDS * attempt)

    LOGGER.error("Giving up on %s after %s attempts", url, RETRY_ATTEMPTS)
    return None


def build_search_api_url(keyword: str, limit: int) -> str:
    query = quote_plus(keyword)
    page_size = max(limit * CANDIDATE_MULTIPLIER, 40)
    return f"{SEARCH_API_URL}?q={query}&page_size={page_size}"


def build_search_url(keyword: str) -> str:
    query = quote_plus(keyword)
    return (
        f"{BASE_URL}/t5/forums/searchpage/tab/message"
        f"?advanced=false&allow_punctuation=false&q={query}"
    )


def parse_search_api_urls(xml: str) -> list[str]:
    soup = BeautifulSoup(xml, "xml")
    messages = soup.select("messages > message")
    urls: list[str] = []

    for message in messages:
        canonical_url = text_from_xml(message, "canonical_url")
        subject = text_from_xml(message, "subject") or ""

        if not canonical_url or not is_candidate_thread_url(canonical_url):
            continue
        if is_avoided_content(f"{subject} {canonical_url}"):
            continue

        normalized = normalize_url(canonical_url)
        if normalized not in urls:
            urls.append(normalized)

    return urls


def text_from_xml(element: Any, selector: str) -> str | None:
    found = element.select_one(selector)
    if not found:
        return None

    text = normalize_text(found.get_text(" ", strip=True))
    return text or None


def extract_original_post(soup: BeautifulSoup) -> str | None:
    selectors = [
        ".lia-message-body-content",
        ".lia-message-body",
        "[class*='message-body']",
        "article",
    ]
    for selector in selectors:
        element = soup.select_one(selector)
        if element:
            return normalize_text(element.get_text(" ", strip=True))

    return None


def extract_date(soup: BeautifulSoup) -> str | None:
    time_element = soup.find("time")
    if time_element:
        return time_element.get("datetime") or normalize_text(time_element.get_text(" ", strip=True))

    for selector in [".local-date", ".lia-message-post-date", "[class*='date']"]:
        text = first_text(soup, [selector])
        if text:
            return text

    return None


def extract_reply_count(soup: BeautifulSoup) -> int | None:
    text = soup.get_text(" ", strip=True)
    patterns = [
        r"(\d+)\s+repl(?:y|ies)",
        r"Replies\s*[:)]?\s*(\d+)",
        r"(\d+)\s+comments?",
        r"Comments\s+(?:Previous\s+)?(?:\d+\s+)*(\d+)\s+Next",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

    return None


async def fetch_reply_count(page: Page, thread_url: str) -> int | None:
    thread_id = extract_thread_id(thread_url)
    if not thread_id:
        return None

    xml = await fetch_url_text(
        page,
        f"{BASE_URL}/restapi/vc/threads/id/{thread_id}/replies/count",
    )
    if not xml:
        return None

    soup = BeautifulSoup(xml, "xml")
    value = text_from_xml(soup, "value")
    return int(value) if value and value.isdigit() else None


def extract_thread_id(url: str) -> str | None:
    match = re.search(r"/(?:td|idi)-p/(\d+)", url)
    return match.group(1) if match else None


def extract_labels(soup: BeautifulSoup) -> list[str]:
    labels: list[str] = []
    selectors = [
        ".lia-list-standard-inline .lia-link-navigation",
        ".lia-message-labels a",
        ".label a",
        "[class*='label'] a",
        "[class*='tag'] a",
    ]
    for selector in selectors:
        for element in soup.select(selector):
            label = normalize_text(element.get_text(" ", strip=True))
            if label and label not in labels:
                labels.append(label)

    return labels


def first_text(soup: BeautifulSoup, selectors: list[str]) -> str | None:
    for selector in selectors:
        element = soup.select_one(selector)
        if element:
            text = normalize_text(element.get_text(" ", strip=True))
            if text:
                return text

    return None


def is_candidate_thread_url(url: str) -> bool:
    if not url.startswith(BASE_URL):
        return False

    excluded_parts = [
        "/searchpage/",
        "/user/",
        "/users/",
        "/plugins/",
        "/settings/",
        "/custom/",
        "/login",
        "#",
    ]
    if any(part in url for part in excluded_parts):
        return False

    return "/t5/" in url and ("/td-p/" in url or "/idi-p/" in url)


def normalize_url(url: str) -> str:
    cleaned = url.split("#", 1)[0].split("?", 1)[0]
    return cleaned.rstrip("/")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_avoided_content(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in AVOID_TERMS)


def is_focused_content(keyword: str, text: str) -> bool:
    lowered = f"{keyword} {text}".lower()
    return any(term in lowered for term in FOCUS_TERMS)


async def polite_delay() -> None:
    await asyncio.sleep(REQUEST_DELAY_SECONDS)


def write_json(output_file: Path, discussions: list[dict[str, Any]]) -> None:
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(discussions, file, ensure_ascii=False, indent=2)


def print_scrape_summary(
    discussions: list[dict[str, Any]],
    counts_by_keyword: dict[str, int],
    duplicates_skipped: int,
    filtered_out: int,
    output_file: Path,
) -> None:
    print("\n=== Spotify Community Scraping Complete ===")
    print(f"Total discussions collected:    {len(discussions)}")
    print(f"Duplicate URLs skipped:         {duplicates_skipped}")
    print(f"Discussions filtered (off-topic): {filtered_out}")
    print("\nCollected per keyword:")
    for keyword, count in counts_by_keyword.items():
        print(f"  {count:>3}  {keyword}")
    print(f"\nJSON output: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Spotify Community discussions about music discovery."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help="Path to write the JSON dataset.",
    )
    parser.add_argument(
        "--threads-per-keyword",
        type=int,
        default=THREADS_PER_KEYWORD,
        help="Maximum candidate discussion threads to collect per keyword.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(
        scrape_spotify_community(
            output_file=args.output,
            threads_per_keyword=args.threads_per_keyword,
        )
    )


if __name__ == "__main__":
    main()
