#!/usr/bin/env python3
"""Standalone script to load a Reddit post and test markup rendering."""

import os
import sys
import argparse
import html
import requests
import re

# Add src directory to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from reddit_browser.http_headers import get_default_headers
from rich.markup import MarkupError, render as render_markup
from rich.markup import escape as rich_escape


def linkify(text: str) -> str:
    """Wrap URLs in Rich link markup."""
    if not text:
        return text
    text = rich_escape(text)
    return re.sub(r"(https?://[^\s\]\)<>\"']+)", r'[link="\1"]\1[/link]', text, flags=re.IGNORECASE)


def fetch_posts(subreddit: str, limit: int = 25):
    url = f"https://www.reddit.com/r/{subreddit}/.json"
    headers = get_default_headers()
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data["data"]["children"]

def fetch_comments(permalink: str):
    url = f"https://www.reddit.com{permalink}.json"
    headers = get_default_headers()
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data[1]["data"]["children"] if len(data) > 1 else []


def build_content(post):
    title = html.unescape(post["data"]["title"])
    author = post["data"].get("author", "[deleted]")
    score = post["data"].get("score", 0)
    num_comments = post["data"].get("num_comments", 0)
    url = post["data"].get("url", "")
    selftext = html.unescape(post["data"].get("selftext", ""))

    content = (
        f"[bold][green]{rich_escape(title)}[/green][/bold]\n\n"
        f"Author: u/[green]{rich_escape(author)}[/green]\n"
        f"Score: [green]{score}[/green]\n"
        f"Comments: [green]{num_comments}[/green]\n"
        f"URL: [green]{linkify(url)}[/green]\n\n"
    )
    if selftext.strip():
        content += f"Content:\n[green]{linkify(selftext)}[/green]\n\n"
    return content

def iter_comment_bodies(comments):
    for item in comments:
        if item.get("kind") != "t1":
            continue
        data = item.get("data", {})
        body = html.unescape(data.get("body", ""))
        yield body
        replies = data.get("replies")
        if isinstance(replies, dict):
            children = replies.get("data", {}).get("children", [])
            yield from iter_comment_bodies(children)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subreddit", default="LocalLLaMA")
    parser.add_argument("--index", type=int, default=6, help="1-based index")
    parser.add_argument("--match", default="StepFun 3.5 Flash vs MiniMax 2.1")
    args = parser.parse_args()

    posts = fetch_posts(args.subreddit)
    idx = max(args.index - 1, 0)
    post = posts[idx] if idx < len(posts) else None

    if args.match:
        for p in posts:
            title = html.unescape(p["data"]["title"])
            if args.match.lower() in title.lower():
                post = p
                break

    if not post:
        print("No post found.")
        return

    title = html.unescape(post["data"]["title"])
    print(f"Loaded post: {title}")
    print(f"URL: {post['data'].get('url','')}")

    content = build_content(post)
    print("\nAttempting to render markup...")
    try:
        render_markup(content)
        print("Markup OK.")
    except MarkupError as e:
        print(f"MarkupError: {e}")

    print("\nChecking comment bodies for markup issues...")
    comments = fetch_comments(post["data"]["permalink"])
    for body in iter_comment_bodies(comments):
        snippet = body[:200].replace("\n", " ")
        wrapped = f"[green]{linkify(body)}[/green]"
        try:
            render_markup(wrapped)
        except MarkupError as e:
            print("First bad comment body:")
            print(snippet)
            print(f"MarkupError: {e}")
            break


if __name__ == "__main__":
    main()
