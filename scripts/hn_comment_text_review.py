#!/usr/bin/env python3
"""Fetch HN story comments and show raw vs converted text for review."""

from __future__ import annotations

import argparse
import json
import textwrap
from typing import Any, Dict, List, Optional

import httpx

from reddit_browser.text_utils import html_to_text


HN_BASE = "https://hacker-news.firebaseio.com/v0"


def _get_json(path: str) -> Any:
    url = f"{HN_BASE}/{path}.json"
    response = httpx.get(url, timeout=10.0)
    response.raise_for_status()
    return response.json()


def _get_item(item_id: int) -> Optional[Dict[str, Any]]:
    data = _get_json(f"item/{item_id}")
    return data if isinstance(data, dict) else None


def _collect_comment_ids(story: Dict[str, Any], limit: int) -> List[int]:
    kids = story.get("kids") or []
    return [int(kid) for kid in kids[:limit]]


def _format_block(title: str, body: str) -> str:
    sep = "-" * 80
    return f"{sep}\n{title}\n{sep}\n{body}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("story_id", type=int, help="HN story id (e.g. 8863)")
    parser.add_argument("--limit", type=int, default=10, help="Number of top-level comments to inspect")
    args = parser.parse_args()

    story = _get_item(args.story_id)
    if not story:
        raise SystemExit("Story not found.")

    print(_format_block("STORY", f"{story.get('title', '')}\n{story.get('url', '')}"))

    comment_ids = _collect_comment_ids(story, args.limit)
    if not comment_ids:
        print("No comments.")
        return

    for idx, comment_id in enumerate(comment_ids, start=1):
        item = _get_item(comment_id)
        if not item or item.get("type") != "comment":
            continue
        raw = item.get("text") or ""
        converted = html_to_text(raw)
        author = item.get("by") or "[deleted]"

        raw_preview = textwrap.fill(raw, width=100)
        converted_preview = textwrap.fill(converted, width=100)

        print(_format_block(
            f"COMMENT {idx} by {author} (id {comment_id})",
            f"RAW:\n{raw_preview}\n\nCONVERTED:\n{converted_preview}",
        ))


if __name__ == "__main__":
    main()
