#!/usr/bin/env python3
"""Summarize subjects across subreddit/HN titles."""

from __future__ import annotations

import os
import sys
import html
import asyncio
import logging
import traceback
import json
import re
import time
import random
from typing import List

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from reddit_browser.api import get_first_two_pages, RedditAPI  # noqa: E402
from reddit_browser.hn_api import HackerNewsAPI  # noqa: E402
from reddit_browser.media import extract_article_text, is_image_url  # noqa: E402
from reddit_browser.text_ai import TextAIClient  # noqa: E402


SUBREDDITS_PATH = os.path.join(ROOT, "subreddits.txt")
TITLES_OUT = os.path.join(ROOT, "titles.txt")
SUBJECTS_OUT = os.path.join(ROOT, "subjects.txt")
ARTICLES_DIR = os.path.join(ROOT, "articles")

MAX_REDDIT_COMMENTS = 50
MAX_HN_COMMENTS = 50
ARTICLE_TIMEOUT_SECONDS = 15
REDDIT_RETRY_LIMIT = 3
REDDIT_RETRY_SLEEP_MIN = 2.0
REDDIT_RETRY_SLEEP_MAX = 6.0
REDDIT_SLEEP_BETWEEN_SOURCES = 2.0
LOCAL_LLAMA_CPP_BASE_URL = os.getenv("LOCAL_LLAMA_CPP_BASE_URL", "http://192.168.1.90:8080/v1")
LOCAL_LLAMA_CPP_TIMEOUT = float(os.getenv("LOCAL_LLAMA_CPP_TIMEOUT", "1.5"))
SUMMARY_POST_LIMIT = int(os.getenv("SUMMARY_POST_LIMIT", "50"))
SUMMARY_SOURCE_LIMIT = int(os.getenv("SUMMARY_SOURCE_LIMIT", "0"))
SUBJECT_TITLE_CHUNK_SIZE = int(os.getenv("SUBJECT_TITLE_CHUNK_SIZE", "75"))
MEDIA_URL_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".webm",
    ".gifv",
}


LOGGER = logging.getLogger("summarize")
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_formatter)
LOGGER.addHandler(_stream_handler)

_file_handler = logging.FileHandler(os.path.join(ROOT, "summarize.log"), encoding="utf-8")
_file_handler.setFormatter(_formatter)
LOGGER.addHandler(_file_handler)


def _format_duration(seconds: float) -> str:
    minutes, remainder = divmod(max(seconds, 0.0), 60.0)
    if minutes >= 1:
        return f"{int(minutes)}m {remainder:.2f}s"
    return f"{remainder:.2f}s"


def _load_subreddits(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    items = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            items.append(line)
    return items


def _is_hn(source: str) -> bool:
    return source.strip().lower() == "news.ycombinator.com"

def _should_skip_source(source: str) -> bool:
    return source.strip().lower() == "twitter"


def _env_truthy(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off"}


def _use_firefox_reddit_session() -> bool:
    return _env_truthy("USE_FIREFOX_REDDIT_SESSION", "1")


def _enabled_sources(sources: List[str]) -> List[str]:
    enabled = []
    for source in sources:
        if _should_skip_source(source):
            LOGGER.info("Skipping source %s", source)
            continue
        enabled.append(source)
        if SUMMARY_SOURCE_LIMIT > 0 and len(enabled) >= SUMMARY_SOURCE_LIMIT:
            break
    return enabled


def _limit_posts(posts: List[dict]) -> List[dict]:
    if SUMMARY_POST_LIMIT <= 0:
        return posts
    return posts[:SUMMARY_POST_LIMIT]


def _should_extract_article_url(url: str) -> bool:
    if not url or not url.startswith("http"):
        return False
    url_lower = url.lower()
    if is_image_url(url_lower):
        return False
    if "reddit.com" in url_lower or "redd.it" in url_lower:
        return False
    if "v.redd.it" in url_lower or "i.redd.it" in url_lower:
        return False
    return not any(url_lower.split("?", 1)[0].endswith(ext) for ext in MEDIA_URL_EXTENSIONS)


def _chunks(items: List[str], size: int) -> List[List[str]]:
    if size <= 0:
        return [items]
    return [items[index:index + size] for index in range(0, len(items), size)]


def _generate_subjects_summary(titles_text: str, text_ai: TextAIClient) -> str:
    title_lines = [line for line in titles_text.splitlines() if line.strip()]
    if not title_lines:
        return ""

    chunk_summaries: List[str] = []
    title_chunks = _chunks(title_lines, SUBJECT_TITLE_CHUNK_SIZE)
    for index, title_chunk in enumerate(title_chunks, start=1):
        chunk_start = time.perf_counter()
        LOGGER.info("Calling LLM to extract subjects for title chunk %d/%d", index, len(title_chunks))
        prompt = (
            "Read these post titles and extract the common subject matter/themes.\n"
            "Write only concise bullet points.\n\n"
            "Titles:\n"
            + "\n".join(title_chunk)
            + "\n\nCommon themes:\n-"
        )
        response = asyncio.run(text_ai.generate_response(prompt, max_tokens=1000))
        summary = (response or "").strip()
        LOGGER.info(
            "Subject chunk %d/%d summary time: %s (%d titles, %d chars output)",
            index,
            len(title_chunks),
            _format_duration(time.perf_counter() - chunk_start),
            len(title_chunk),
            len(summary),
        )
        if summary:
            if not summary.startswith("-"):
                summary = "- " + summary
            chunk_summaries.append(summary)

    if not chunk_summaries:
        return ""

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    merge_start = time.perf_counter()
    LOGGER.info("Calling LLM to merge %d title subject chunk(s)", len(chunk_summaries))
    merge_prompt = (
        "Merge these topic lists into one deduplicated set of common themes across all sources.\n"
        "Write 5-15 concise bullet points. Write only the bullets.\n\n"
        "Topic lists:\n"
        + "\n\n".join(chunk_summaries)
        + "\n\nMerged common themes:\n-"
    )
    merged = asyncio.run(text_ai.generate_response(merge_prompt, max_tokens=1200))
    merged = (merged or "").strip()
    LOGGER.info(
        "Subject merge summary time: %s (%d chars output)",
        _format_duration(time.perf_counter() - merge_start),
        len(merged),
    )
    if merged and not merged.startswith("-"):
        merged = "- " + merged
    return merged


def _fetch_reddit_posts_with_retry(subreddit: str):
    last_exc = None
    last_tb = None
    for attempt in range(1, REDDIT_RETRY_LIMIT + 1):
        try:
            return get_first_two_pages(
                subreddit,
                user_agent=os.getenv("REDDIT_USER_AGENT"),
                use_firefox_session=_use_firefox_reddit_session(),
                firefox_profile=os.getenv("FIREFOX_PROFILE"),
            )
        except Exception as exc:  # httpx.HTTPStatusError is in api.py call stack
            last_exc = exc
            last_tb = traceback.format_exc()
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status == 429:
                LOGGER.warning("Rate limited for r/%s; skipping Reddit fetch.", subreddit)
                return []
            LOGGER.error("Failed fetching r/%s (attempt %d/%d): %s", subreddit, attempt, REDDIT_RETRY_LIMIT, exc)
            if attempt < REDDIT_RETRY_LIMIT:
                sleep_for = random.uniform(REDDIT_RETRY_SLEEP_MIN, REDDIT_RETRY_SLEEP_MAX)
                LOGGER.info("Retrying r/%s after %.1fs", subreddit, sleep_for)
                time.sleep(sleep_for)
    LOGGER.error("Giving up on r/%s after %d attempts", subreddit, REDDIT_RETRY_LIMIT)
    if last_exc and last_tb:
        LOGGER.error("Last error:\n%s", last_tb)
    return []


def _fetch_hn_titles(limit: int = 50) -> List[str]:
    LOGGER.info("Fetching Hacker News top %d titles", limit)
    hn = HackerNewsAPI()
    try:
        stories = hn.get_top_stories(limit=limit)
    finally:
        hn.close()
    titles: List[str] = []
    for story in stories:
        title = story.get("title") or ""
        title = html.unescape(title).strip()
        if title:
            titles.append(title)
    LOGGER.info("Collected %d Hacker News titles", len(titles))
    return titles


def _sanitize_source(source: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", source.strip()).strip("_").lower()


def _extract_reddit_comments(permalink: str) -> List[str]:
    reddit = RedditAPI(
        use_firefox_session=_use_firefox_reddit_session(),
        firefox_profile=os.getenv("FIREFOX_PROFILE"),
    )
    try:
        data = reddit.get_json(f"{permalink}.json")
    finally:
        reddit.close()
    comments = []
    if isinstance(data, list) and len(data) > 1:
        children = data[1].get("data", {}).get("children", [])
        for item in children:
            if item.get("kind") != "t1":
                continue
            body = item.get("data", {}).get("body") or ""
            body = html.unescape(body).strip()
            if body:
                comments.append(body)
            if len(comments) >= MAX_REDDIT_COMMENTS:
                break
    return comments


def _extract_hn_comments(story_id: int) -> List[str]:
    hn = HackerNewsAPI()
    try:
        comments = asyncio.run(hn.get_comments_tree_async(story_id, max_comments=MAX_HN_COMMENTS))
    finally:
        asyncio.run(hn.aclose())
    texts = []
    for comment in comments:
        body = comment.get("data", {}).get("body") or ""
        body = html.unescape(body).strip()
        if body:
            texts.append(body)
        if len(texts) >= MAX_HN_COMMENTS:
            break
    return texts


async def _extract_article_text_with_timeout(url: str) -> str:
    try:
        return await asyncio.wait_for(extract_article_text(url), timeout=ARTICLE_TIMEOUT_SECONDS)
    except Exception:
        LOGGER.error("Article extraction failed for %s\n%s", url, traceback.format_exc())
        return ""


def _summarize_text_content(text: str, text_ai: TextAIClient) -> str:
    if not text or text.startswith("Error:"):
        return ""
    if text_ai.backend is None:
        return ""
    try:
        summary = asyncio.run(text_ai.generate_text_summary(text[:12000])).strip()
        if summary.startswith("Error"):
            return ""
        return summary
    except Exception:
        LOGGER.error("Failed summarizing text\n%s", traceback.format_exc())
        return ""


def _summarize_parent_comments(
    comments: List[str],
    text_ai: TextAIClient,
    *,
    post_summary: str = "",
    article_summary: str = "",
) -> str:
    if not comments:
        return ""
    if text_ai.backend is None:
        return ""
    top_comments = comments[:7]
    comments_text = "\n".join(f"- {comment}" for comment in top_comments)
    try:
        context_parts = []
        if post_summary:
            context_parts.append(f"Reddit/HN post summary:\n{post_summary}")
        if article_summary:
            context_parts.append(f"Linked article summary:\n{article_summary}")
        if context_parts:
            context_text = "\n\n".join(context_parts)
            prompt = (
                "Summarize the key points, sentiments, and disagreements in these top comments. "
                "Use the post/article context, but focus the output on what commenters are saying. "
                "Keep it concise and readable in 5-8 short bullets.\n\n"
                "Context:\n"
                f"{context_text}\n\n"
                "Top comments:\n"
                f"{comments_text}\n\n"
                "Comments summary:"
            )
            return asyncio.run(text_ai.generate_response(prompt, max_tokens=2000, use_ai_response_model=False))
        return asyncio.run(text_ai.generate_comments_summary(comments_text))
    except Exception:
        LOGGER.error("Failed summarizing comments\n%s", traceback.format_exc())
        return ""


def _write_article_jsonl(source: str, index: int, payload: dict) -> None:
    os.makedirs(ARTICLES_DIR, exist_ok=True)
    filename = f"{_sanitize_source(source)}_{index}.jsonl"
    out_path = os.path.join(ARTICLES_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    LOGGER.info("Wrote %s", out_path)


def main() -> int:
    run_start = time.perf_counter()
    text_ai = TextAIClient(
        prefer_local=_env_truthy("USE_LOCAL_LLAMA_CPP", "1"),
        local_base_url=LOCAL_LLAMA_CPP_BASE_URL,
        local_timeout=LOCAL_LLAMA_CPP_TIMEOUT,
    )
    text_ai.initialize()
    LOGGER.info(text_ai.describe_backend())

    sources = _load_subreddits(SUBREDDITS_PATH)
    if not sources:
        LOGGER.error("No subreddits found in %s", SUBREDDITS_PATH)
        return 1

    LOGGER.info("Loaded %d sources from %s", len(sources), SUBREDDITS_PATH)
    sources = _enabled_sources(sources)
    if not sources:
        LOGGER.error("No enabled sources found in %s", SUBREDDITS_PATH)
        return 1
    LOGGER.info("Using %d enabled source(s): %s", len(sources), ", ".join(sources))
    lines: List[str] = []
    reddit_posts_cache = {}
    titles_start = time.perf_counter()
    for source in sources:
        if _should_skip_source(source):
            LOGGER.info("Skipping source %s", source)
            continue
        if _is_hn(source):
            titles = _fetch_hn_titles()
            for title in titles:
                lines.append(f"[hn] {title}")
        else:
            LOGGER.info("Fetching Reddit titles for r/%s", source)
            posts = _limit_posts(_fetch_reddit_posts_with_retry(source))
            reddit_posts_cache[source] = posts
            titles = []
            for post in posts:
                data = post.get("data") or {}
                title = data.get("title") or ""
                title = html.unescape(title).strip()
                if title:
                    titles.append(title)
            LOGGER.info("Collected %d titles for r/%s", len(titles), source)
            for title in titles:
                lines.append(f"[r/{source}] {title}")
            if REDDIT_SLEEP_BETWEEN_SOURCES > 0:
                time.sleep(REDDIT_SLEEP_BETWEEN_SOURCES)
    LOGGER.info("Title collection time: %s", _format_duration(time.perf_counter() - titles_start))

    with open(TITLES_OUT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")
    LOGGER.info("Wrote %d titles to %s", len(lines), TITLES_OUT)

    LOGGER.info("Extracting article content and comments into %s", ARTICLES_DIR)
    extraction_start = time.perf_counter()
    for source in sources:
        if _should_skip_source(source):
            continue
        source_start = time.perf_counter()
        source_record_count = 0
        article_index = 1
        if _is_hn(source):
            hn = HackerNewsAPI()
            try:
                stories = _limit_posts(hn.get_top_stories(limit=50))
            finally:
                hn.close()
            for story in stories:
                record_start = time.perf_counter()
                article_extract_seconds = 0.0
                post_summary_seconds = 0.0
                article_summary_seconds = 0.0
                comment_extract_seconds = 0.0
                comment_summary_seconds = 0.0
                title = html.unescape(story.get("title") or "").strip()
                url = story.get("url") or ""
                post_text = html.unescape(story.get("text") or "").strip()
                article_text = ""
                if _should_extract_article_url(url):
                    try:
                        step_start = time.perf_counter()
                        article_text = asyncio.run(_extract_article_text_with_timeout(url))
                        article_extract_seconds = time.perf_counter() - step_start
                    except Exception:
                        article_text = ""
                step_start = time.perf_counter()
                post_summary = _summarize_text_content(post_text, text_ai)
                post_summary_seconds = time.perf_counter() - step_start
                step_start = time.perf_counter()
                article_summary = _summarize_text_content(article_text, text_ai)
                article_summary_seconds = time.perf_counter() - step_start
                comments = []
                story_id = story.get("id")
                if story_id is not None:
                    try:
                        step_start = time.perf_counter()
                        comments = _extract_hn_comments(int(story_id))
                        comment_extract_seconds = time.perf_counter() - step_start
                    except Exception:
                        comments = []
                step_start = time.perf_counter()
                comments_summary = _summarize_parent_comments(
                    comments,
                    text_ai,
                    post_summary=post_summary,
                    article_summary=article_summary,
                )
                comment_summary_seconds = time.perf_counter() - step_start
                payload = {
                    "source": "hn",
                    "title": title,
                    "url": url,
                    "post_text": post_text,
                    "post_summary": post_summary,
                    "article_text": article_text,
                    "article_summary": article_summary,
                    "comments": comments,
                    "comments_summary": comments_summary,
                }
                _write_article_jsonl(source, article_index, payload)
                source_record_count += 1
                LOGGER.info(
                    "Article timing source=%s index=%d total=%s article_extract=%s post_summary=%s article_summary=%s comments_extract=%s comments_summary=%s comments=%d",
                    source,
                    article_index,
                    _format_duration(time.perf_counter() - record_start),
                    _format_duration(article_extract_seconds),
                    _format_duration(post_summary_seconds),
                    _format_duration(article_summary_seconds),
                    _format_duration(comment_extract_seconds),
                    _format_duration(comment_summary_seconds),
                    len(comments),
                )
                article_index += 1
        else:
            posts = reddit_posts_cache.get(source) or _limit_posts(_fetch_reddit_posts_with_retry(source))
            for post in posts:
                record_start = time.perf_counter()
                article_extract_seconds = 0.0
                post_summary_seconds = 0.0
                article_summary_seconds = 0.0
                comment_extract_seconds = 0.0
                comment_summary_seconds = 0.0
                data = post.get("data") or {}
                title = html.unescape(data.get("title") or "").strip()
                url = data.get("url") or ""
                selftext = html.unescape(data.get("selftext") or "").strip()
                permalink = data.get("permalink") or ""
                if permalink and not permalink.startswith("http"):
                    permalink = f"https://www.reddit.com{permalink}"
                article_text = ""
                if _should_extract_article_url(url):
                    try:
                        step_start = time.perf_counter()
                        article_text = asyncio.run(_extract_article_text_with_timeout(url))
                        article_extract_seconds = time.perf_counter() - step_start
                    except Exception:
                        article_text = ""
                step_start = time.perf_counter()
                post_summary = _summarize_text_content(selftext, text_ai)
                post_summary_seconds = time.perf_counter() - step_start
                step_start = time.perf_counter()
                article_summary = _summarize_text_content(article_text, text_ai)
                article_summary_seconds = time.perf_counter() - step_start
                comments = []
                if permalink:
                    try:
                        step_start = time.perf_counter()
                        comments = _extract_reddit_comments(permalink)
                        comment_extract_seconds = time.perf_counter() - step_start
                    except Exception:
                        comments = []
                step_start = time.perf_counter()
                comments_summary = _summarize_parent_comments(
                    comments,
                    text_ai,
                    post_summary=post_summary,
                    article_summary=article_summary,
                )
                comment_summary_seconds = time.perf_counter() - step_start
                payload = {
                    "source": f"r/{source}",
                    "title": title,
                    "url": url,
                    "post_text": selftext,
                    "post_summary": post_summary,
                    "article_text": article_text,
                    "article_summary": article_summary,
                    "comments": comments,
                    "comments_summary": comments_summary,
                }
                _write_article_jsonl(source, article_index, payload)
                source_record_count += 1
                LOGGER.info(
                    "Article timing source=%s index=%d total=%s article_extract=%s post_summary=%s article_summary=%s comments_extract=%s comments_summary=%s comments=%d",
                    source,
                    article_index,
                    _format_duration(time.perf_counter() - record_start),
                    _format_duration(article_extract_seconds),
                    _format_duration(post_summary_seconds),
                    _format_duration(article_summary_seconds),
                    _format_duration(comment_extract_seconds),
                    _format_duration(comment_summary_seconds),
                    len(comments),
                )
                article_index += 1
        LOGGER.info(
            "Source timing source=%s records=%d total=%s avg_per_article=%s",
            source,
            source_record_count,
            _format_duration(time.perf_counter() - source_start),
            _format_duration((time.perf_counter() - source_start) / source_record_count if source_record_count else 0),
        )
    LOGGER.info("Article/comment extraction total time: %s", _format_duration(time.perf_counter() - extraction_start))

    if text_ai.backend is None:
        LOGGER.warning("No text AI backend available; skipping subjects.txt")
        return 0

    LOGGER.info("Reading titles from %s before LLM step", TITLES_OUT)
    with open(TITLES_OUT, "r", encoding="utf-8") as handle:
        titles_text = handle.read().strip()

    subject_start = time.perf_counter()
    subjects_text = _generate_subjects_summary(titles_text, text_ai).strip()
    LOGGER.info("Subject summary total time: %s", _format_duration(time.perf_counter() - subject_start))
    if not subjects_text:
        subjects_text = "No subjects generated."

    with open(SUBJECTS_OUT, "w", encoding="utf-8") as handle:
        handle.write(subjects_text)
        handle.write("\n")

    LOGGER.info("Wrote subjects to %s", SUBJECTS_OUT)
    LOGGER.info("End-to-end time: %s", _format_duration(time.perf_counter() - run_start))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
