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
import requests
from typing import List

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from reddit_browser.api import NonJSONResponseError, get_first_two_pages, RedditAPI  # noqa: E402
from reddit_browser.hn_api import HackerNewsAPI  # noqa: E402
from reddit_browser.media import extract_article_text, is_image_url  # noqa: E402
from reddit_browser.text_ai import TextAIClient  # noqa: E402


SUBREDDITS_PATH = os.path.join(ROOT, "subreddits.txt")
TITLES_OUT = os.path.join(ROOT, "titles.txt")
SUBJECTS_OUT = os.path.join(ROOT, "subjects.txt")
SUMMARY_REPORT_OUT = os.path.join(ROOT, "summary_report.html")
SUMMARY_REPORT_URL_OUT = os.path.join(ROOT, "summary_report_url.txt")
SUMMARY_REPORT_UPLOAD_ERROR_OUT = os.path.join(ROOT, "summary_report_upload_error.html")
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
SUMMARY_USE_ARTICLE_CACHE = os.getenv("SUMMARY_USE_ARTICLE_CACHE", "0").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
SUMMARY_UPLOAD_REPORT = os.getenv("SUMMARY_UPLOAD_REPORT", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
PAGEBOWL_API_URL = os.getenv("PAGEBOWL_API_URL", "https://api.pagebowl.com/v1/pages/upload")
PAGEBOWL_USER_AGENT = os.getenv(
    "PAGEBOWL_USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36 ReddBrowserSummary/0.1",
)
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


def _source_label(source: str) -> str:
    if _is_hn(source):
        return "hn"
    return f"r/{source}"


def _normalize_source_label(source: str) -> str:
    source = source.strip()
    if _is_hn(source) or source.lower() == "hn":
        return "hn"
    if source.lower().startswith("r/"):
        return f"r/{source[2:]}"
    return f"r/{source}"


def _compact_text(text: str, limit: int = 900) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].strip() + "..."


def _subject_input_line(record: dict) -> str:
    source = record.get("source") or ""
    title = _compact_text(record.get("title") or "", 220)
    parts = [f"[{source}] {title}"]
    post_summary = _compact_text(record.get("post_summary") or "", 500)
    article_summary = _compact_text(record.get("article_summary") or "", 500)
    comments_summary = _compact_text(record.get("comments_summary") or "", 500)
    if post_summary:
        parts.append(f"Post summary: {post_summary}")
    if article_summary:
        parts.append(f"Linked article summary: {article_summary}")
    if comments_summary:
        parts.append(f"Comments summary: {comments_summary}")
    return " | ".join(parts)


def _load_cached_article_records(sources: List[str]) -> List[dict]:
    if not os.path.isdir(ARTICLES_DIR):
        return []

    enabled_labels = {_normalize_source_label(source).lower() for source in sources}
    counts = {label: 0 for label in enabled_labels}
    records = []
    for filename in sorted(os.listdir(ARTICLES_DIR)):
        if not filename.endswith(".jsonl"):
            continue
        path = os.path.join(ARTICLES_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                line = handle.readline().strip()
            if not line:
                continue
            record = json.loads(line)
        except Exception:
            LOGGER.warning("Skipping unreadable cached article file %s", path)
            continue

        label = _normalize_source_label(record.get("source") or "").lower()
        if label not in enabled_labels:
            continue
        if SUMMARY_POST_LIMIT > 0 and counts[label] >= SUMMARY_POST_LIMIT:
            continue
        counts[label] += 1
        records.append(record)

    return records


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
        LOGGER.info("Calling LLM to extract subjects for input chunk %d/%d", index, len(title_chunks))
        prompt = (
            "Read these post records and extract the common subject matter/themes. "
            "Records may include titles, post summaries, linked article summaries, and comment summaries.\n"
            "Write each subject as its own clearly separated section using this exact format:\n\n"
            "Heading\n"
            "Sources: source1, source2\n"
            "Summary: 3-4 sentences explaining what is specifically happening in this record set.\n"
            "Representative titles:\n"
            "- [source] title\n"
            "- [source] title\n\n"
            "Use only sources that appear in the records for that subject. Include 2-4 representative titles per subject. "
            "Keep subjects specific enough to be useful; split unrelated items instead of creating a catch-all category. "
            "Separate subjects with one blank line.\n\n"
            "Records:\n"
            + "\n".join(title_chunk)
            + "\n\nCommon themes:\n"
        )
        response = asyncio.run(text_ai.generate_response(prompt, max_tokens=3500))
        summary = (response or "").strip()
        LOGGER.info(
            "Subject chunk %d/%d summary time: %s (%d input lines, %d chars output)",
            index,
            len(title_chunks),
            _format_duration(time.perf_counter() - chunk_start),
            len(title_chunk),
            len(summary),
        )
        if summary:
            chunk_summaries.append(summary)

    if not chunk_summaries:
        return ""

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    merge_start = time.perf_counter()
    LOGGER.info("Calling LLM to merge %d title subject chunk(s)", len(chunk_summaries))
    merge_prompt = (
        "Merge these topic lists into one deduplicated set of common themes across all sources.\n"
        "Write 5-15 subject sections using this exact format:\n\n"
        "Heading\n"
        "Sources: source1, source2\n"
        "Summary: 3-4 sentences explaining what is specifically happening across the included titles.\n"
        "Representative titles:\n"
        "- [source] title\n"
        "- [source] title\n\n"
        "Use only sources that appear in the representative titles for that subject. Include 2-4 representative titles per subject. "
        "Keep subjects specific enough to be useful; split unrelated items instead of creating a catch-all category. "
        "Drop isolated low-signal topics unless they represent a clear recurring subject. "
        "Separate sections with one blank line.\n\n"
        "Topic lists:\n"
        + "\n\n".join(chunk_summaries)
        + "\n\nMerged common themes:\n"
    )
    merged = asyncio.run(text_ai.generate_response(merge_prompt, max_tokens=5000))
    merged = (merged or "").strip()
    LOGGER.info(
        "Subject merge summary time: %s (%d chars output)",
        _format_duration(time.perf_counter() - merge_start),
        len(merged),
    )
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
            if isinstance(exc, NonJSONResponseError):
                LOGGER.warning("Non-JSON response for r/%s; skipping Reddit fetch. %s", subreddit, exc)
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


def _title_lookup_key(source: str, title: str) -> str:
    normalized_title = re.sub(r"\s+", " ", title or "").strip().lower()
    return f"{source.strip().lower()}\n{normalized_title}"


def _build_title_url_lookup(records: List[dict]) -> dict:
    lookup = {}
    for record in records:
        source = record.get("source") or ""
        title = record.get("title") or ""
        url = record.get("url") or ""
        if source and title and url:
            lookup[_title_lookup_key(source, title)] = url
    return lookup


def _render_representative_title(raw_line: str, title_urls: dict) -> str:
    line = raw_line.strip()
    if line.startswith("-"):
        line = line[1:].strip()
    match = re.match(r"\[([^\]]+)\]\s*(.+)", line)
    if not match:
        return f"<li>{html.escape(line)}</li>"
    source = match.group(1).strip()
    title = match.group(2).strip()
    url = title_urls.get(_title_lookup_key(source, title))
    label = f"[{source}] {title}"
    if url:
        return f'<li><a href="{html.escape(url)}">{html.escape(label)}</a></li>'
    return f"<li>{html.escape(label)}</li>"


def _parse_subject_sections(subjects_text: str) -> List[dict]:
    sections = []
    for raw_section in re.split(r"\n\s*\n", subjects_text.strip()):
        lines = [line.strip() for line in raw_section.splitlines() if line.strip()]
        if not lines:
            continue

        heading = lines[0]
        sources = ""
        summary_lines = []
        representative_lines = []
        mode = ""
        for line in lines[1:]:
            if line.startswith("Sources:"):
                sources = line[len("Sources:"):].strip()
                mode = ""
            elif line.startswith("Summary:"):
                summary_lines.append(line[len("Summary:"):].strip())
                mode = "summary"
            elif line.startswith("Representative titles:"):
                mode = "representative_titles"
            elif mode == "summary":
                summary_lines.append(line)
            elif mode == "representative_titles":
                representative_lines.append(line)
            else:
                summary_lines.append(line)

        sections.append(
            {
                "heading": heading,
                "sources": sources,
                "summary": " ".join(summary_lines),
                "representative_titles": representative_lines,
            }
        )
    return sections


def _render_subjects_html(subject_sections: List[dict], records: List[dict]) -> str:
    title_urls = _build_title_url_lookup(records)
    sections = []
    for subject in subject_sections:
        heading = subject["heading"]
        sources = subject["sources"]
        summary = subject["summary"]
        representative_lines = subject["representative_titles"]
        representative_html = ""
        if representative_lines:
            representative_html = (
                "<h3>Representative Titles</h3>"
                "<ul>"
                + "".join(_render_representative_title(line, title_urls) for line in representative_lines)
                + "</ul>"
            )

        sources_html = f'<div class="subject-sources">Sources: {html.escape(sources)}</div>' if sources else ""
        search_text = html.escape(f"{heading} {sources} {summary} {' '.join(representative_lines)}".lower())
        sections.append(
            f"<article class=\"subject\" data-search-text=\"{search_text}\">"
            f"<h2>{html.escape(heading)}</h2>"
            f"{sources_html}"
            f"<p>{html.escape(summary)}</p>"
            f"{representative_html}"
            "</article>"
        )

    if sections:
        return "\n".join(sections)
    return ""


def _wordcloud_weight(subject: dict) -> int:
    title_count = len(subject["representative_titles"])
    source_count = len([source for source in subject["sources"].split(",") if source.strip()])
    return max(1, title_count + source_count)


def _render_wordcloud_html(subject_sections: List[dict], records: List[dict]) -> str:
    if not subject_sections:
        return ""

    title_urls = _build_title_url_lookup(records)
    buttons = []
    details = []
    weights = [_wordcloud_weight(subject) for subject in subject_sections]
    min_weight = min(weights)
    max_weight = max(weights)
    for index, subject in enumerate(subject_sections):
        weight = _wordcloud_weight(subject)
        if max_weight == min_weight:
            size = 1.1
        else:
            size = 0.95 + ((weight - min_weight) / (max_weight - min_weight)) * 0.75
        heading = subject["heading"]
        active_class = " active" if index == 0 else ""
        search_text = html.escape(
            f"{heading} {subject['sources']} {subject['summary']} {' '.join(subject['representative_titles'])}".lower()
        )
        buttons.append(
            f'<button type="button" class="cloud-term{active_class}" '
            f'data-subject-index="{index}" data-search-text="{search_text}" style="font-size:{size:.2f}rem">'
            f"{html.escape(heading)}</button>"
        )

        representative_lines = subject["representative_titles"]
        representative_html = ""
        if representative_lines:
            representative_html = (
                "<h3>Related Topics</h3>"
                "<ul>"
                + "".join(_render_representative_title(line, title_urls) for line in representative_lines)
                + "</ul>"
            )
        hidden_attr = "" if index == 0 else " hidden"
        details.append(
            f'<article class="cloud-detail" data-subject-detail="{index}" data-search-text="{search_text}"{hidden_attr}>'
            f"<h2>{html.escape(heading)}</h2>"
            f"<div class=\"subject-sources\">Sources: {html.escape(subject['sources'])}</div>"
            f"<p>{html.escape(subject['summary'])}</p>"
            f"{representative_html}"
            "</article>"
        )

    return (
        '<section class="wordcloud-section">'
        '<h2>Topic Cloud</h2>'
        '<div class="wordcloud" aria-label="Clickable topic cloud">'
        + "".join(buttons)
        + "</div>"
        '<div class="cloud-details">'
        + "".join(details)
        + "</div>"
        "</section>"
    )


def _render_summary_report(subjects_text: str, records: List[dict]) -> str:
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    source_counts = {}
    for record in records:
        source = record.get("source") or "unknown"
        source_counts[source] = source_counts.get(source, 0) + 1

    source_summary = ", ".join(f"{html.escape(source)}: {count}" for source, count in sorted(source_counts.items()))
    subject_sections = _parse_subject_sections(subjects_text)
    wordcloud_html = _render_wordcloud_html(subject_sections, records)
    subjects_html = _render_subjects_html(subject_sections, records) or f"<pre>{html.escape(subjects_text)}</pre>"
    rows = []
    for record in records:
        source = html.escape(record.get("source") or "")
        title = html.escape(record.get("title") or "")
        url = record.get("url") or ""
        title_html = f'<a href="{html.escape(url)}">{title}</a>' if url else title
        url_html = f'<a href="{html.escape(url)}">{html.escape(url)}</a>' if url else ""
        post_summary = html.escape(record.get("post_summary") or "")
        article_summary = html.escape(record.get("article_summary") or "")
        comments_summary = html.escape(record.get("comments_summary") or "")
        summary_parts = []
        if post_summary:
            summary_parts.append(f"<h3>Post Summary</h3><p>{post_summary}</p>")
        if article_summary:
            summary_parts.append(f"<h3>Linked Article Summary</h3><p>{article_summary}</p>")
        if comments_summary:
            summary_parts.append(f"<h3>Comments Summary</h3><pre>{comments_summary}</pre>")
        search_text = html.escape(
            " ".join(
                [
                    record.get("source") or "",
                    record.get("title") or "",
                    record.get("url") or "",
                    record.get("post_summary") or "",
                    record.get("article_summary") or "",
                    record.get("comments_summary") or "",
                ]
            ).lower()
        )
        rows.append(
            f"<article class=\"record\" data-search-text=\"{search_text}\">"
            f"<div class=\"source\">{source}</div>"
            f"<h2>{title_html}</h2>"
            f"<p class=\"url\">{url_html}</p>"
            f"{''.join(summary_parts) or '<p class=\"muted\">No saved summaries.</p>'}"
            "</article>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Reddit Browser Summary Report</title>
  <style>
    body {{
      margin: 0;
      color: #17202a;
      background: #f6f7f9;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }}
    main {{
      max-width: 1040px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    header {{
      margin-bottom: 28px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 20px;
    }}
    h3 {{
      margin: 18px 0 6px;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: .04em;
      color: #5a6673;
    }}
    .meta, .muted, .url {{
      color: #5a6673;
    }}
    .controls {{
      position: sticky;
      top: 0;
      z-index: 10;
      display: grid;
      gap: 12px;
      background: rgba(246, 247, 249, .96);
      border-bottom: 1px solid #dde2e8;
      padding: 12px 0;
      margin-bottom: 18px;
    }}
    .tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .tab-button {{
      border: 1px solid #b9c5d1;
      border-radius: 6px;
      background: #fff;
      color: #24313c;
      cursor: pointer;
      font: inherit;
      font-weight: 700;
      padding: 8px 12px;
    }}
    .tab-button.active {{
      background: #155e75;
      border-color: #155e75;
      color: #fff;
    }}
    .search-box {{
      width: min(520px, 100%);
      box-sizing: border-box;
      border: 1px solid #b9c5d1;
      border-radius: 6px;
      font: inherit;
      padding: 10px 12px;
    }}
    .tab-panel[hidden],
    [data-filter-hidden="true"] {{
      display: none;
    }}
    .filter-empty {{
      background: #fff;
      border: 1px dashed #b9c5d1;
      border-radius: 8px;
      color: #5a6673;
      padding: 20px;
      margin: 16px 0;
    }}
    .filter-empty[hidden] {{
      display: none;
    }}
    .subjects, .subject, .record {{
      background: #fff;
      border: 1px solid #dde2e8;
      border-radius: 8px;
      padding: 20px;
      margin: 16px 0;
    }}
    .subjects {{
      background: transparent;
      border: 0;
      padding: 0;
    }}
    .subject-sources {{
      color: #155e75;
      font-weight: 700;
      margin-bottom: 10px;
      font-size: 14px;
    }}
    .wordcloud-section {{
      background: #fff;
      border: 1px solid #dde2e8;
      border-radius: 8px;
      padding: 20px;
      margin: 16px 0 24px;
    }}
    .wordcloud {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 10px 12px;
      padding: 8px 0 18px;
    }}
    .cloud-term {{
      border: 1px solid #b9c5d1;
      border-radius: 999px;
      background: #eef4f8;
      color: #174356;
      cursor: pointer;
      font: inherit;
      font-weight: 700;
      line-height: 1.2;
      padding: 8px 12px;
      transition: background .12s ease, border-color .12s ease, transform .12s ease;
    }}
    .cloud-term:hover,
    .cloud-term:focus {{
      background: #dcecf3;
      border-color: #6f98aa;
      outline: none;
      transform: translateY(-1px);
    }}
    .cloud-term.active {{
      background: #155e75;
      border-color: #155e75;
      color: #fff;
    }}
    .cloud-detail {{
      border-top: 1px solid #dde2e8;
      padding-top: 16px;
    }}
    .cloud-detail[hidden] {{
      display: none;
    }}
    ul {{
      margin: 6px 0 0 20px;
      padding: 0;
    }}
    li {{
      margin: 6px 0;
    }}
    .subjects pre, .record pre {{
      white-space: pre-wrap;
      word-wrap: break-word;
      font-family: inherit;
      margin: 0;
    }}
    .source {{
      display: inline-block;
      margin-bottom: 8px;
      color: #155e75;
      font-weight: 700;
      font-size: 13px;
    }}
    a {{
      color: #0f5f8f;
    }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>Reddit Browser Summary Report</h1>
    <div class="meta">Generated {html.escape(generated_at)} · {len(records)} records · {source_summary}</div>
  </header>
  <nav class="controls" aria-label="Report controls">
    <div class="tabs" role="tablist" aria-label="Report views">
      <button type="button" class="tab-button active" data-tab-target="subjects-panel" role="tab" aria-selected="true">Subjects</button>
      <button type="button" class="tab-button" data-tab-target="posts-panel" role="tab" aria-selected="false">Post Summaries</button>
      <button type="button" class="tab-button" data-tab-target="cloud-panel" role="tab" aria-selected="false">Topic Cloud</button>
    </div>
    <input class="search-box" type="search" placeholder="Search report" aria-label="Search report">
  </nav>
  <section id="subjects-panel" class="subjects tab-panel" data-panel-name="Subjects">
    <h2>Subjects</h2>
    {subjects_html}
    <p class="filter-empty" hidden>No matching subjects.</p>
  </section>
  <section id="posts-panel" class="tab-panel" data-panel-name="Post Summaries" hidden>
    <h2>Post Summaries</h2>
    {''.join(rows)}
    <p class="filter-empty" hidden>No matching posts.</p>
  </section>
  <div id="cloud-panel" class="tab-panel" data-panel-name="Topic Cloud" hidden>
    {wordcloud_html}
    <p class="filter-empty" hidden>No matching topics.</p>
  </div>
</main>
<script>
  const searchBox = document.querySelector('.search-box');
  const tabButtons = Array.from(document.querySelectorAll('.tab-button'));
  const panels = Array.from(document.querySelectorAll('.tab-panel'));

  function activePanel() {{
    return panels.find((panel) => !panel.hidden);
  }}

  function applyFilter() {{
    const query = (searchBox.value || '').trim().toLowerCase();
    const panel = activePanel();
    if (!panel) return;
    const items = Array.from(panel.querySelectorAll('[data-search-text]'));
    let visibleCount = 0;
    items.forEach((item) => {{
      const matches = !query || item.dataset.searchText.includes(query);
      item.dataset.filterHidden = matches ? 'false' : 'true';
      if (matches) visibleCount += 1;
    }});
    const empty = panel.querySelector('.filter-empty');
    if (empty) {{
      empty.hidden = visibleCount > 0;
    }}
    if (panel.id === 'cloud-panel') {{
      const visibleButtons = Array.from(panel.querySelectorAll('.cloud-term')).filter(
        (button) => button.dataset.filterHidden !== 'true'
      );
      const activeButton = visibleButtons.find((button) => button.classList.contains('active')) || visibleButtons[0];
      if (activeButton) {{
        setActiveCloud(activeButton);
      }} else {{
        panel.querySelectorAll('.cloud-detail').forEach((detail) => {{
          detail.hidden = true;
        }});
      }}
    }}
  }}

  function setActiveTab(targetId) {{
    tabButtons.forEach((button) => {{
      const active = button.dataset.tabTarget === targetId;
      button.classList.toggle('active', active);
      button.setAttribute('aria-selected', active ? 'true' : 'false');
    }});
    panels.forEach((panel) => {{
      panel.hidden = panel.id !== targetId;
    }});
    applyFilter();
  }}

  function setActiveCloud(button) {{
    const index = button.dataset.subjectIndex;
    document.querySelectorAll('.cloud-term').forEach((item) => {{
      item.classList.toggle('active', item === button);
    }});
    document.querySelectorAll('.cloud-detail').forEach((detail) => {{
      const matchesSelected = detail.dataset.subjectDetail === index;
      const filteredOut = detail.dataset.filterHidden === 'true';
      detail.hidden = !matchesSelected || filteredOut;
    }});
  }}

  tabButtons.forEach((button) => {{
    button.addEventListener('click', () => setActiveTab(button.dataset.tabTarget));
  }});

  document.querySelectorAll('.cloud-term').forEach((button) => {{
    button.addEventListener('click', () => setActiveCloud(button));
  }});

  searchBox.addEventListener('input', applyFilter);
  applyFilter();
</script>
</body>
</html>
"""


def _write_summary_report(subjects_text: str, records: List[dict]) -> None:
    with open(SUMMARY_REPORT_OUT, "w", encoding="utf-8") as handle:
        handle.write(_render_summary_report(subjects_text, records))
    LOGGER.info("Wrote HTML summary report to %s", SUMMARY_REPORT_OUT)


def _upload_summary_report() -> str:
    if not SUMMARY_UPLOAD_REPORT:
        LOGGER.info("Skipping summary report upload because SUMMARY_UPLOAD_REPORT=0")
        return ""
    with open(SUMMARY_REPORT_OUT, "rb") as handle:
        response = requests.post(
            PAGEBOWL_API_URL,
            headers={"User-Agent": PAGEBOWL_USER_AGENT},
            files={"file": (os.path.basename(SUMMARY_REPORT_OUT), handle, "text/html")},
            timeout=60,
        )
    if response.status_code >= 400:
        with open(SUMMARY_REPORT_UPLOAD_ERROR_OUT, "w", encoding="utf-8") as handle:
            handle.write(response.text)
        raise RuntimeError(
            f"Page Bowl upload failed with HTTP {response.status_code}; "
            f"response body saved to {SUMMARY_REPORT_UPLOAD_ERROR_OUT}"
        )
    try:
        payload = response.json()
    except ValueError:
        with open(SUMMARY_REPORT_UPLOAD_ERROR_OUT, "w", encoding="utf-8") as handle:
            handle.write(response.text)
        raise RuntimeError(f"Page Bowl upload returned non-JSON response saved to {SUMMARY_REPORT_UPLOAD_ERROR_OUT}")
    url = (payload.get("url") or "").strip()
    if not url:
        with open(SUMMARY_REPORT_UPLOAD_ERROR_OUT, "w", encoding="utf-8") as handle:
            handle.write(response.text)
        raise RuntimeError(f"Page Bowl upload response did not include a URL; response saved to {SUMMARY_REPORT_UPLOAD_ERROR_OUT}")
    with open(SUMMARY_REPORT_URL_OUT, "w", encoding="utf-8") as handle:
        handle.write(url)
        handle.write("\n")
    LOGGER.info("Uploaded HTML summary report to %s", url)
    return url


def _write_subject_input(lines: List[str]) -> None:
    with open(TITLES_OUT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")
    LOGGER.info("Wrote %d subject input lines to %s", len(lines), TITLES_OUT)


def _regenerate_subjects_from_input(text_ai: TextAIClient, run_start: float, records: List[dict]) -> int:
    if text_ai.backend is None:
        LOGGER.warning("No text AI backend available; skipping subjects.txt")
        return 0

    LOGGER.info("Reading subject input from %s before LLM step", TITLES_OUT)
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
    _write_summary_report(subjects_text, records)
    try:
        _upload_summary_report()
    except Exception:
        LOGGER.error("Failed uploading summary report\n%s", traceback.format_exc())
    LOGGER.info("End-to-end time: %s", _format_duration(time.perf_counter() - run_start))
    return 0


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

    if SUMMARY_USE_ARTICLE_CACHE:
        LOGGER.info("Using cached article records from %s; skipping fetch/extraction.", ARTICLES_DIR)
        cached_records = _load_cached_article_records(sources)
        if not cached_records:
            LOGGER.error("No cached article records found in %s for enabled sources.", ARTICLES_DIR)
            return 1
        _write_subject_input([_subject_input_line(record) for record in cached_records])
        return _regenerate_subjects_from_input(text_ai, run_start, cached_records)

    lines: List[str] = []
    article_records: List[dict] = []
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

    _write_subject_input(lines)

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
                article_records.append(payload)
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
                article_records.append(payload)
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

    if article_records:
        _write_subject_input([_subject_input_line(record) for record in article_records])

    return _regenerate_subjects_from_input(text_ai, run_start, article_records)


if __name__ == "__main__":
    raise SystemExit(main())
