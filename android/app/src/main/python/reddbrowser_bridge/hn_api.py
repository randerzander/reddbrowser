"""Hacker News API helpers for Android bridge."""

from __future__ import annotations

from typing import Any

import requests

from .text_utils import html_to_text

BASE_URL = "https://hacker-news.firebaseio.com/v0"


def _item_url(item_id: int | str) -> str:
    return f"{BASE_URL}/item/{item_id}.json"


def get_top_stories(limit: int = 50) -> list[dict[str, Any]]:
    ids_resp = requests.get(f"{BASE_URL}/topstories.json", timeout=10)
    ids_resp.raise_for_status()
    story_ids = ids_resp.json()[:limit]
    out: list[dict[str, Any]] = []
    for item_id in story_ids:
        item_resp = requests.get(_item_url(item_id), timeout=10)
        item_resp.raise_for_status()
        item = item_resp.json()
        if item and item.get("type") == "story":
            out.append(item)
    return out


def story_to_post(story: dict[str, Any]) -> dict[str, Any]:
    story_id = story.get("id")
    hn_comments_url = f"https://news.ycombinator.com/item?id={story_id}" if story_id else ""
    url = story.get("url") or hn_comments_url
    return {
        "source": "hn",
        "data": {
            "id": str(story_id) if story_id is not None else "",
            "title": story.get("title", ""),
            "author": story.get("by", "[deleted]"),
            "score": int(story.get("score", 0) or 0),
            "num_comments": int(story.get("descendants", 0) or 0),
            "url": url,
            "permalink": hn_comments_url,
            "selftext": html_to_text(story.get("text", "")),
            "hn_comments_url": hn_comments_url,
            "hn_id": str(story_id) if story_id is not None else "",
            "created_at": str(story.get("time", "")),
        },
    }


def get_hn_post_detail(story_id: str, max_comments: int = 250, max_depth: int = 12) -> dict[str, Any]:
    root_resp = requests.get(_item_url(story_id), timeout=10)
    root_resp.raise_for_status()
    root = root_resp.json() or {}

    remaining = {"count": max_comments}

    def build_comment(item_id: int | str, depth: int) -> dict[str, Any] | None:
        if remaining["count"] <= 0 or depth > max_depth:
            return None
        resp = requests.get(_item_url(item_id), timeout=10)
        resp.raise_for_status()
        item = resp.json() or {}
        if item.get("type") != "comment":
            return None
        remaining["count"] -= 1
        author = item.get("by") or "[deleted]"
        body = html_to_text(item.get("text") or "")
        if item.get("deleted") or item.get("dead"):
            author = "[deleted]"
            body = ""
        children: list[dict[str, Any]] = []
        for child_id in item.get("kids") or []:
            child = build_comment(child_id, depth + 1)
            if child:
                children.append(child)
        return {
            "data": {
                "id": str(item.get("id")),
                "author": author,
                "body": body,
                "score": int(item.get("score", 0) or 0),
            },
            "replies": children,
            "level": 0,
        }

    comments = []
    for kid in root.get("kids") or []:
        comment = build_comment(kid, 0)
        if comment:
            comments.append(comment)

    return {
        "post": story_to_post(root),
        "comments_tree": comments,
    }
