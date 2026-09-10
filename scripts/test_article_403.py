#!/usr/bin/env python3
"""Standalone tester for the same subreddit URL the app loads."""

import os
import sys

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from reddit_browser.http_headers import get_default_headers


DEFAULT_SUBREDDIT = os.getenv("SUBREDDIT", "LocalLlama")


def main() -> int:
    url = f"https://www.reddit.com/r/{DEFAULT_SUBREDDIT}/.json"
    params = {"limit": 25}
    headers = get_default_headers()

    print(f"URL: {url}")
    print(f"Params: {params}")
    print(f"Headers: {headers}")

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Final URL: {response.url}")
        response.raise_for_status()
    except Exception as exc:
        print(f"Request failed: {exc}")
        return 1

    data = response.json()
    posts = data.get("data", {}).get("children", [])
    print(f"Posts returned: {len(posts)}")
    if posts:
        first = posts[0].get("data", {})
        print(f"First title: {first.get('title', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
