"""Comment tree helpers shared across Reddit/HN views."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set


def build_comment_tree(comments_data: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build a tree structure from flat comments data."""

    def process_replies(replies_list: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        if not replies_list:
            return result

        for item in replies_list:
            if item.get("kind") == "t1":  # It's a comment
                comment_data = item.get("data", {})
                comment_obj = {"data": comment_data, "replies": [], "level": 0}

                replies = comment_data.get("replies")
                if isinstance(replies, dict) and "data" in replies:
                    nested_replies = replies["data"].get("children", [])
                    comment_obj["replies"] = process_replies(nested_replies)

                result.append(comment_obj)

        result.sort(key=lambda x: x["data"].get("score", 0), reverse=True)
        return result

    root_comments: List[Dict[str, Any]] = []
    for item in comments_data:
        if item.get("kind") == "t1":
            comment_data = item.get("data", {})
            comment_obj = {"data": comment_data, "replies": [], "level": 0}

            replies = comment_data.get("replies")
            if isinstance(replies, dict) and "data" in replies:
                nested_replies = replies["data"].get("children", [])
                comment_obj["replies"] = process_replies(nested_replies)

            root_comments.append(comment_obj)

    root_comments.sort(key=lambda x: x["data"].get("score", 0), reverse=True)
    return root_comments


def flatten_comments(
    comments: Iterable[Dict[str, Any]],
    expanded_ids: Set[str],
    level: int = 0,
) -> List[Dict[str, Any]]:
    """Flatten the comment tree for display, respecting expanded state."""
    result: List[Dict[str, Any]] = []
    for comment in comments:
        comment_copy = dict(comment)
        comment_copy["level"] = level
        result.append(comment_copy)

        comment_id = comment.get("data", {}).get("id")
        if comment_id in expanded_ids:
            result.extend(flatten_comments(comment.get("replies", []), expanded_ids, level + 1))

    return result
