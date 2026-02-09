"""Reddit Browser - A textual TUI for browsing Reddit"""

import httpx
import logging
from typing import Dict, List, Optional, Any
import html
from .http_headers import get_default_headers


class RedditAPI:
    """A simple client for interacting with the Reddit API."""
    
    def __init__(self, user_agent: Optional[str] = None, base_url: str = "https://www.reddit.com"):
        self.base_url = base_url
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
    
    def get_subreddit_posts(self, subreddit: str, limit: int = 25, after: Optional[str] = None) -> Dict:
        """Fetch posts from a subreddit (sync)."""
        url = f"{self.base_url}/r/{subreddit}/.json"
        params = {"limit": limit}
        if after:
            params["after"] = after
        self.logger.debug("Requesting subreddit posts: url=%s params=%s headers=%s", url, params, self.headers)
        response = self.client.get(url, params=params)
        self.logger.debug("Response status: %s headers=%s", response.status_code, response.headers)
        response.raise_for_status()
        return response.json()

    async def get_subreddit_posts_async(self, subreddit: str, limit: int = 25, after: Optional[str] = None) -> Dict:
        """Fetch posts from a subreddit (async)."""
        url = f"{self.base_url}/r/{subreddit}/.json"
        params = {"limit": limit}
        if after:
            params["after"] = after
        self.logger.debug("Requesting subreddit posts (async): url=%s params=%s headers=%s", url, params, self.headers)
        response = await self.async_client.get(url, params=params)
        self.logger.debug("Response status (async): %s headers=%s", response.status_code, response.headers)
        response.raise_for_status()
        return response.json()

    async def get_comments_async(self, permalink: str) -> List[Dict]:
        """Fetch comments for a post (async)."""
        url = f"{self.base_url}{permalink}.json"
        self.logger.debug("Requesting comments: url=%s headers=%s", url, self.headers)
        response = await self.async_client.get(url)
        self.logger.debug("Comments response status: %s headers=%s", response.status_code, response.headers)
        response.raise_for_status()
        return response.json()

    def build_comment_tree(self, comments_data: List[Dict]) -> List[Dict]:
        """Build a tree structure from flat comments data."""
        def process_replies(replies_list):
            """Recursively process replies to build the tree."""
            result = []
            if not replies_list or replies_list == []:
                return result

            for item in replies_list:
                if item["kind"] == "t1":  # It's a comment
                    comment_data = item["data"]
                    comment_obj = {
                        "data": comment_data,
                        "replies": [],
                        "level": 0
                    }

                    if "replies" in comment_data and comment_data["replies"]:
                        if isinstance(comment_data["replies"], dict) and "data" in comment_data["replies"]:
                            nested_replies = comment_data["replies"]["data"].get("children", [])
                            comment_obj["replies"] = process_replies(nested_replies)

                    result.append(comment_obj)

            result.sort(key=lambda x: x["data"].get("score", 0), reverse=True)
            return result

        root_comments = []
        for item in comments_data:
            if item["kind"] == "t1":
                comment_data = item["data"]
                comment_obj = {
                    "data": comment_data,
                    "replies": [],
                    "level": 0
                }

                if "replies" in comment_data and comment_data["replies"]:
                    if isinstance(comment_data["replies"], dict) and "data" in comment_data["replies"]:
                        nested_replies = comment_data["replies"]["data"].get("children", [])
                        comment_obj["replies"] = process_replies(nested_replies)

                root_comments.append(comment_obj)

        root_comments.sort(key=lambda x: x["data"].get("score", 0), reverse=True)
        return root_comments

    def flatten_comments(self, comments: List[Dict], expanded_ids: set, level: int = 0) -> List[Dict]:
        """Flatten the comment tree for display, respecting expanded state."""
        result = []
        for comment in comments:
            comment_copy = dict(comment)
            comment_copy["level"] = level
            result.append(comment_copy)

            comment_id = comment["data"]["id"]
            if comment_id in expanded_ids:
                result.extend(self.flatten_comments(comment["replies"], expanded_ids, level + 1))

        return result

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
        try:
            first_page = reddit.get_subreddit_posts(subreddit, limit=25)
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code == 403:
                reddit.close()
                reddit = RedditAPI(user_agent=user_agent, base_url="https://old.reddit.com")
                first_page = reddit.get_subreddit_posts(subreddit, limit=25)
            else:
                raise
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
