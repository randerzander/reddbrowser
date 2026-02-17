"""Hacker News API helpers."""

import asyncio
import html
import logging
import re
from typing import Any, Dict, List, Optional

import httpx


def html_to_text(content_html: str) -> str:
    """Best-effort conversion of HTML content to plain text."""
    if not content_html:
        return ""
    text = re.sub(r"<[^>]+>", " ", content_html)
    return html.unescape(" ".join(text.split()))


class HackerNewsAPI:
    """A small client for the Hacker News Firebase API."""

    def __init__(self, base_url: str = "https://hacker-news.firebaseio.com/v0"):
        self.base_url = base_url.rstrip("/")
        self.logger = logging.getLogger(__name__)
        self.client = httpx.Client(timeout=10.0)
        self.async_client = httpx.AsyncClient(timeout=10.0)

    def _url(self, path: str) -> str:
        path = path.lstrip("/")
        if path.endswith(".json"):
            return f"{self.base_url}/{path}"
        return f"{self.base_url}/{path}.json"

    def get_item(self, item_id: int) -> Optional[Dict[str, Any]]:
        response = self.client.get(self._url(f"item/{item_id}"))
        response.raise_for_status()
        return response.json()

    def get_top_story_ids(self) -> List[int]:
        response = self.client.get(self._url("topstories"))
        response.raise_for_status()
        ids = response.json()
        return ids if isinstance(ids, list) else []

    def get_top_stories(self, limit: int = 50) -> List[Dict[str, Any]]:
        ids = self.get_top_story_ids()[:limit]
        stories: List[Dict[str, Any]] = []
        for item_id in ids:
            item = self.get_item(item_id)
            if item and item.get("type") == "story":
                stories.append(item)
        return stories

    async def get_item_async(self, item_id: int) -> Optional[Dict[str, Any]]:
        response = await self.async_client.get(self._url(f"item/{item_id}"))
        response.raise_for_status()
        return response.json()

    async def get_top_story_ids_async(self) -> List[int]:
        response = await self.async_client.get(self._url("topstories"))
        response.raise_for_status()
        ids = response.json()
        return ids if isinstance(ids, list) else []

    async def get_top_stories_async(self, limit: int = 50) -> List[Dict[str, Any]]:
        ids = (await self.get_top_story_ids_async())[:limit]
        semaphore = asyncio.Semaphore(12)

        async def fetch(item_id: int) -> Optional[Dict[str, Any]]:
            async with semaphore:
                return await self.get_item_async(item_id)

        tasks = [fetch(item_id) for item_id in ids]
        results = await asyncio.gather(*tasks)
        return [item for item in results if item and item.get("type") == "story"]

    async def get_comments_tree_async(
        self,
        story_id: int,
        max_comments: int = 300,
        max_depth: int = 12,
    ) -> List[Dict[str, Any]]:
        story = await self.get_item_async(story_id)
        if not story:
            return []
        kids = story.get("kids") or []
        remaining = [max_comments]
        semaphore = asyncio.Semaphore(12)

        async def fetch(item_id: int) -> Optional[Dict[str, Any]]:
            async with semaphore:
                return await self.get_item_async(item_id)

        async def build_comment(item_id: int, depth: int) -> Optional[Dict[str, Any]]:
            if remaining[0] <= 0 or depth > max_depth:
                return None
            item = await fetch(item_id)
            if not item or item.get("type") != "comment":
                return None
            remaining[0] -= 1

            author = item.get("by") or "[deleted]"
            body_html = item.get("text") or ""
            if item.get("deleted") or item.get("dead"):
                author = "[deleted]"
                body_html = ""

            comment = {
                "data": {
                    "id": str(item.get("id")),
                    "author": author,
                    "body": html_to_text(body_html),
                    "score": item.get("score", 0),
                },
                "replies": [],
                "level": 0,
            }

            child_ids = item.get("kids") or []
            if child_ids and remaining[0] > 0 and depth < max_depth:
                tasks = [build_comment(child_id, depth + 1) for child_id in child_ids]
                children = await asyncio.gather(*tasks)
                comment["replies"] = [child for child in children if child]

            return comment

        tasks = [build_comment(child_id, 0) for child_id in kids]
        results = await asyncio.gather(*tasks)
        return [comment for comment in results if comment]

    def close(self) -> None:
        self.client.close()

    async def aclose(self) -> None:
        await self.async_client.aclose()
