"""Debug script to check the actual Reddit API response structure."""

import httpx
import json

def debug_reddit_api():
    """Debug the Reddit API response structure."""
    url = "https://www.reddit.com/r/LocalLlama/.json"
    headers = {"User-Agent": "RedditBrowser/0.1.0"}
    
    response = httpx.get(url, headers=headers, params={"limit": 5})
    response.raise_for_status()
    
    data = response.json()
    
    print("=== DEBUG: Reddit API Response Structure ===\n")
    
    # Print the structure of the response
    print("Top level keys:", list(data.keys()))
    print("Data keys:", list(data["data"].keys()))
    print("Number of children (posts):", len(data["data"]["children"]))
    
    # Print info about first post
    first_post = data["data"]["children"][0]["data"]
    print("\nFirst post keys:", list(first_post.keys()))
    
    print("\nFirst post details:")
    print(f"  Title: {first_post.get('title', 'N/A')[:100]}...")
    print(f"  Author: {first_post.get('author', 'N/A')}")
    print(f"  Score: {first_post.get('score', 'N/A')}")
    print(f"  Num comments: {first_post.get('num_comments', 'N/A')}")
    print(f"  Selftext: {first_post.get('selftext', 'N/A')[:100]}...")
    print(f"  URL: {first_post.get('url', 'N/A')}")
    print(f"  Stickied: {first_post.get('stickied', 'N/A')}")
    
    print("\n=== Sample of first 3 posts ===")
    for i, post in enumerate(data["data"]["children"][:3]):
        pdata = post["data"]
        print(f"\nPost {i+1}:")
        print(f"  Title: {pdata.get('title', 'N/A')[:60]}...")
        print(f"  Author: u/{pdata.get('author', 'N/A')}")
        print(f"  Score: {pdata.get('score', 'N/A')}")
        print(f"  Comments: {pdata.get('num_comments', 'N/A')}")

if __name__ == "__main__":
    debug_reddit_api()