#!/usr/bin/env python3
"""Standalone Twikit session test using an existing cookies.json file."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys


def _truncate(value: object, max_len: int = 180) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _print_tweets(items, limit: int) -> None:
    """Print compact details for up to `limit` tweets."""
    shown = 0
    for item in items:
        if shown >= limit:
            break
        tweet_id = getattr(item, "id", "")
        text = getattr(item, "full_text", None) or getattr(item, "text", "")
        user = getattr(item, "user", None)
        user_name = getattr(user, "screen_name", None) or getattr(user, "name", None) or "unknown"
        print(f"{shown + 1}. @{user_name} | {tweet_id}")
        print(f"   {_truncate(text)}")
        shown += 1

    if shown == 0:
        print("No tweet items were returned to print.")


async def run(cookies_file: str, locale: str, check: str) -> int:
    try:
        from twikit import Client
    except ImportError:
        print("twikit is not installed. Install with: pip install twikit", file=sys.stderr)
        return 2

    if not os.path.exists(cookies_file):
        print(f"Cookies file not found: {cookies_file}", file=sys.stderr)
        return 2

    try:
        client = Client(locale)
    except TypeError:
        client = Client()

    try:
        client.load_cookies(cookies_file)
    except Exception as exc:
        print(f"Failed to load cookies from {cookies_file}: {exc}", file=sys.stderr)
        return 1

    try:
        if check == "notifications":
            result = await client.get_notifications("All", count=5)
            count = len(result)
            print(f"Cookie auth check passed via notifications. Retrieved {count} items.")
            print("Notification objects returned; tweet printing is skipped for this check.")
        elif check == "bookmarks":
            result = await client.get_bookmarks(count=5)
            count = len(result)
            print(f"Cookie auth check passed via bookmarks. Retrieved {count} items.")
            _print_tweets(result, limit=3)
        else:
            result = await client.get_latest_timeline(count=5)
            count = len(result)
            print(f"Cookie auth check passed via latest timeline. Retrieved {count} items.")
            _print_tweets(result, limit=3)
    except Exception as exc:
        print(
            "Cookie auth check failed. Cookies may be expired or blocked by anti-bot checks.\n"
            f"Details: {exc}",
            file=sys.stderr,
        )
        return 1

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Twikit auth using cookies.json")
    parser.add_argument(
        "--cookies-file",
        default="cookies.json",
        help="Path to Twikit cookie file (default: cookies.json)",
    )
    parser.add_argument(
        "--locale",
        default="en-US",
        help="Client locale (default: en-US)",
    )
    parser.add_argument(
        "--check",
        choices=["notifications", "bookmarks", "timeline"],
        default="notifications",
        help="Authenticated API call to run (default: notifications)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(run(args.cookies_file, args.locale, args.check))


if __name__ == "__main__":
    raise SystemExit(main())
