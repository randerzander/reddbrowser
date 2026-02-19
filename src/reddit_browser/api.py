"""Reddit Browser - A textual TUI for browsing Reddit"""

import httpx
import logging
from typing import Any, Dict, List, Optional
import html
from urllib.parse import urlparse, urlunparse
from .comments import build_comment_tree as _build_comment_tree, flatten_comments as _flatten_comments
from .http_headers import get_default_headers


class RedditAPI:
    """A simple client for interacting with the Reddit API."""

    def __init__(
        self,
        user_agent: Optional[str] = None,
        base_url: str = "https://www.reddit.com",
        fallback_base_url: str = "https://old.reddit.com",
    ):
        self.base_url = base_url
        self.fallback_base_url = fallback_base_url
        self.headers = get_default_headers(user_agent)
        self.logger = logging.getLogger(__name__)
        self.client = httpx.Client(
            headers=self.headers,
            timeout=10.0
        )
        self.async_client = httpx.AsyncClient(
            headers=self.headers,
            timeout=10.0
        )

    def _build_url(self, path_or_url: str) -> str:
        if path_or_url.startswith("http"):
            return path_or_url
        return f"{self.base_url}{path_or_url}"

    def _build_fallback_url(self, url: str) -> Optional[str]:
        if not self.fallback_base_url:
            return None
        if url.startswith(self.fallback_base_url):
            return None
        if url.startswith(self.base_url):
            return url.replace(self.base_url, self.fallback_base_url, 1)
        parsed = urlparse(url)
        if parsed.netloc.endswith("reddit.com"):
            fallback_parsed = urlparse(self.fallback_base_url)
            return urlunparse(
                (
                    fallback_parsed.scheme,
                    fallback_parsed.netloc,
                    parsed.path,
                    parsed.params,
                    parsed.query,
                    parsed.fragment,
                )
            )
        return None

    def _request_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        response = self.client.get(url, params=params)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code == 403:
                fallback_url = self._build_fallback_url(url)
                if fallback_url:
                    response = self.client.get(fallback_url, params=params)
                    response.raise_for_status()
                else:
                    raise
            else:
                raise
        return response.json()

    async def _request_json_async(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        response = await self.async_client.get(url, params=params)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code == 403:
                fallback_url = self._build_fallback_url(url)
                if fallback_url:
                    response = await self.async_client.get(fallback_url, params=params)
                    response.raise_for_status()
                else:
                    raise
            else:
                raise
        return response.json()
    
    def get_subreddit_posts(self, subreddit: str, limit: int = 25, after: Optional[str] = None) -> Dict:
        """Fetch posts from a subreddit (sync)."""
        url = self._build_url(f"/r/{subreddit}/.json")
        params = {"limit": limit}
        if after:
            params["after"] = after
        self.logger.debug("Requesting subreddit posts: url=%s params=%s headers=%s", url, params, self.headers)
        data = self._request_json(url, params=params)
        return data

    async def get_subreddit_posts_async(self, subreddit: str, limit: int = 25, after: Optional[str] = None) -> Dict:
        """Fetch posts from a subreddit (async)."""
        url = self._build_url(f"/r/{subreddit}/.json")
        params = {"limit": limit}
        if after:
            params["after"] = after
        self.logger.debug("Requesting subreddit posts (async): url=%s params=%s headers=%s", url, params, self.headers)
        data = await self._request_json_async(url, params=params)
        return data

    async def get_comments_async(self, permalink: str) -> List[Dict]:
        """Fetch comments for a post (async)."""
        url = self._build_url(f"{permalink}.json")
        self.logger.debug("Requesting comments: url=%s headers=%s", url, self.headers)
        data = await self._request_json_async(url)
        return data

    def get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Fetch JSON from a URL, retrying on 403 with a fallback base."""
        target_url = self._build_url(url)
        return self._request_json(target_url, params=params)

    async def get_json_async(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Fetch JSON from a URL (async), retrying on 403 with a fallback base."""
        target_url = self._build_url(url)
        return await self._request_json_async(target_url, params=params)

    def build_comment_tree(self, comments_data: List[Dict]) -> List[Dict]:
        """Build a tree structure from flat comments data."""
        return _build_comment_tree(comments_data)

    def flatten_comments(self, comments: List[Dict], expanded_ids: set, level: int = 0) -> List[Dict]:
        """Flatten the comment tree for display, respecting expanded state."""
        return _flatten_comments(comments, expanded_ids, level)

    def close(self):
        """Close the HTTP clients."""
        self.client.close()
        # Note: async_client.aclose() should be awaited, but we can't easily do it here
        # In a real app, we'd use a context manager or proper lifecycle management

    async def aclose(self):
        """Close the async HTTP client."""
        await self.async_client.aclose()


def get_first_two_pages(subreddit: str, user_agent: Optional[str] = None) -> List[Dict]:
    """Get the first two pages of posts from a subreddit (sync)."""
    reddit = RedditAPI(user_agent=user_agent)
    try:
        first_page = reddit.get_subreddit_posts(subreddit, limit=25)
        posts = first_page["data"]["children"]
        
        after_token = first_page["data"].get("after")
        if after_token:
            second_page = reddit.get_subreddit_posts(subreddit, limit=25, after=after_token)
            posts.extend(second_page["data"]["children"])
        
        return posts
    finally:
        reddit.close()


async def get_comments_tree(permalink: str) -> List[Dict]:
    """Fetch and build comment tree (async)."""
    reddit = RedditAPI()
    try:
        data = await reddit.get_comments_async(permalink)
        comments_data = data[1]["data"]["children"] if len(data) > 1 else []
        return reddit.build_comment_tree(comments_data)
    finally:
        await reddit.aclose()
