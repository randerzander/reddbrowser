"""Comment tree helpers."""

from __future__ import annotations

from typing import Any


def build_reddit_comment_tree(comments_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def process(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in nodes:
            if item.get("kind") != "t1":
                continue
            data = item.get("data", {})
            replies = data.get("replies")
            children = []
            if isinstance(replies, dict):
                children = process((replies.get("data") or {}).get("children") or [])
            out.append({"data": data, "replies": children, "level": 0})
        out.sort(key=lambda c: c["data"].get("score", 0), reverse=True)
        return out

    return process(comments_data or [])


def flatten_top_comments(comments: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    return (comments or [])[:limit]
