"""Reddit API helpers for Android bridge."""

from __future__ import annotations

from typing import Any

import requests

from .comments import build_reddit_comment_tree
from .http_headers import get_default_headers


def get_first_two_pages(subreddit: str, limit: int = 25) -> list[dict[str, Any]]:
    url = f"https://www.reddit.com/r/{subreddit}/.json"
    headers = get_default_headers()
    first = requests.get(url, params={"limit": limit}, headers=headers, timeout=10)
    first.raise_for_status()
    first_json = first.json()
    posts = list((first_json.get("data") or {}).get("children") or [])
    after = (first_json.get("data") or {}).get("after")
    if after:
        second = requests.get(url, params={"limit": limit, "after": after}, headers=headers, timeout=10)
        second.raise_for_status()
        second_json = second.json()
        posts.extend((second_json.get("data") or {}).get("children") or [])
    return [post for post in posts if not (post.get("data") or {}).get("stickied", False)]


def get_reddit_post_detail(permalink: str) -> dict[str, Any]:
    url = f"https://www.reddit.com{permalink}.json"
    response = requests.get(url, headers=get_default_headers(), timeout=10)
    response.raise_for_status()
    payload = response.json()
    listing = (payload[0].get("data") or {}).get("children") if len(payload) > 0 else []
    comments_data = (payload[1].get("data") or {}).get("children") if len(payload) > 1 else []
    post = listing[0] if listing else {"source": "reddit", "data": {}}
    return {
        "post": {"source": "reddit", "data": post.get("data") or {}},
        "comments_tree": build_reddit_comment_tree(comments_data),
    }
