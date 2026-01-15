"""Reddit Browser - A textual TUI for browsing Reddit"""

import httpx
from typing import Dict, List, Optional


class RedditAPI:
    """A simple client for interacting with the Reddit API."""
    
    def __init__(self):
        self.base_url = "https://www.reddit.com"
        self.client = httpx.Client(
            headers={"User-Agent": "RedditBrowser/0.1.0"},
            timeout=10.0
        )
    
    def get_subreddit_posts(self, subreddit: str, limit: int = 25, after: Optional[str] = None) -> Dict:
        """
        Fetch posts from a subreddit.
        
        Args:
            subreddit: The name of the subreddit (without r/)
            limit: Number of posts to fetch (max 100)
            after: Pagination token for next page
        
        Returns:
            Dictionary containing the API response
        """
        url = f"{self.base_url}/r/{subreddit}/.json"
        params = {"limit": limit}
        if after:
            params["after"] = after
            
        response = self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()


def get_first_two_pages(subreddit: str) -> List[Dict]:
    """
    Get the first two pages of posts from a subreddit.
    
    Args:
        subreddit: The name of the subreddit (without r/)
    
    Returns:
        List of post dictionaries from both pages
    """
    reddit = RedditAPI()
    try:
        # Get first page
        first_page = reddit.get_subreddit_posts(subreddit, limit=25)
        posts = first_page["data"]["children"]
        
        # Get second page if available
        after_token = first_page["data"].get("after")
        if after_token:
            second_page = reddit.get_subreddit_posts(subreddit, limit=25, after=after_token)
            posts.extend(second_page["data"]["children"])
        
        return posts
    finally:
        reddit.close()
