"""Chaquopy bridge entry points for Android Kotlin callers."""

from __future__ import annotations

import json
from typing import Any

from .api import get_first_two_pages, get_reddit_post_detail
from .hn_api import get_hn_post_detail, get_top_stories, story_to_post
from .media import ask_ai as ask_ai_impl
from .media import extract_article_text_sync, summarize_comments as summarize_comments_impl
from .media import summarize_text as summarize_text_impl


def _ok(result: Any) -> str:
    return json.dumps({"ok": True, "result": result}, ensure_ascii=False)


def _error(code: str, message: str, retryable: bool = False) -> str:
    return json.dumps(
        {"ok": False, "error": {"code": code, "message": message, "retryable": retryable}},
        ensure_ascii=False,
    )


def _reddit_post_to_item(post: dict[str, Any]) -> dict[str, Any]:
    data = (post or {}).get("data") or {}
    return {"source": "reddit", "data": data}


def list_posts(source: str, subreddit: str, limit: int = 50, page_token: str = "") -> str:
    try:
        if source == "hn":
            stories = get_top_stories(limit=limit)
            posts = [story_to_post(story) for story in stories]
            return _ok({"posts": posts, "next_page_token": ""})

        effective_subreddit = subreddit or "localllama"
        posts = [_reddit_post_to_item(post) for post in get_first_two_pages(effective_subreddit, limit=25)]
        if len(posts) > limit:
            posts = posts[:limit]
        return _ok({"posts": posts, "next_page_token": ""})
    except Exception as exc:  # pragma: no cover
        return _error("feed_load_failed", str(exc), retryable=True)


def get_post_detail(source: str, post_id_or_permalink: str) -> str:
    try:
        if source == "hn":
            detail = get_hn_post_detail(post_id_or_permalink)
            return _ok(detail)
        detail = get_reddit_post_detail(post_id_or_permalink)
        return _ok(detail)
    except Exception as exc:  # pragma: no cover
        return _error("post_detail_failed", str(exc), retryable=True)


def summarize_text(text: str, api_key: str, model: str) -> str:
    try:
        output = summarize_text_impl(text, api_key=api_key, model=model)
        return _ok({"summary": output})
    except Exception as exc:  # pragma: no cover
        return _error("summary_failed", str(exc), retryable=False)


def summarize_article(url: str, api_key: str, model: str) -> str:
    try:
        extracted = extract_article_text_sync(url)
        output = summarize_text_impl(extracted, api_key=api_key, model=model)
        return _ok({"summary": output})
    except Exception as exc:  # pragma: no cover
        return _error("article_summary_failed", str(exc), retryable=True)


def summarize_comments(comments_text: str, api_key: str, model: str) -> str:
    try:
        output = summarize_comments_impl(comments_text, api_key=api_key, model=model)
        return _ok({"summary": output})
    except Exception as exc:  # pragma: no cover
        return _error("comments_summary_failed", str(exc), retryable=False)


def ask_ai(context: str, user_prompt: str, api_key: str, model: str) -> str:
    try:
        output = ask_ai_impl(context, user_prompt, api_key=api_key, model=model)
        return _ok({"response": output})
    except Exception as exc:  # pragma: no cover
        return _error("ask_ai_failed", str(exc), retryable=False)
