"""Twitter/X helpers built on top of Twikit."""

from __future__ import annotations

import asyncio
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional


def _tweet_text(tweet: Any) -> str:
    text = getattr(tweet, "full_text", None) or getattr(tweet, "text", "") or ""
    return str(text).strip()


def _tweet_author(tweet: Any) -> str:
    user = getattr(tweet, "user", None)
    if user is None:
        return "unknown"
    return (
        getattr(user, "screen_name", None)
        or getattr(user, "name", None)
        or "unknown"
    )


def _tweet_url(tweet: Any) -> str:
    tweet_id = str(getattr(tweet, "id", "") or "")
    author = _tweet_author(tweet)
    if tweet_id and author and author != "unknown":
        return f"https://x.com/{author}/status/{tweet_id}"
    if tweet_id:
        return f"https://x.com/i/status/{tweet_id}"
    return ""


def _tweet_quote_id(tweet: Any) -> str:
    """Best-effort extraction of quoted tweet ID from raw payload."""
    data = getattr(tweet, "_data", None) or {}
    quoted = data.get("quoted_status_result") or {}
    result = quoted.get("result") or {}
    return str(result.get("rest_id", "") or "")


def _extract_status_links(tweet: Any) -> List[Dict[str, str]]:
    """Extract referenced tweet links from tweet URL entities."""
    out: List[Dict[str, str]] = []
    seen: set[str] = set()
    pattern = re.compile(r"https?://(?:x|twitter)\.com/([^/]+)/status/(\d+)", re.IGNORECASE)

    urls = list(getattr(tweet, "urls", []) or [])
    for entry in urls:
        candidates: List[str] = []
        if isinstance(entry, dict):
            for key in ("expanded_url", "url", "display_url"):
                value = entry.get(key)
                if isinstance(value, str) and value:
                    candidates.append(value)
        elif isinstance(entry, str):
            candidates.append(entry)

        for candidate in candidates:
            match = pattern.search(candidate)
            if not match:
                continue
            handle, tweet_id = match.group(1), match.group(2)
            url = f"https://x.com/{handle}/status/{tweet_id}"
            if tweet_id in seen:
                continue
            seen.add(tweet_id)
            out.append({"id": tweet_id, "author": handle, "url": url})
    return out


def _tweet_image_urls(tweet: Any) -> List[str]:
    """Extract image URLs from tweet media objects."""
    urls: List[str] = []
    for media in list(getattr(tweet, "media", []) or []):
        media_type = str(getattr(media, "type", "") or "").lower()
        media_url = str(getattr(media, "media_url", "") or "").strip()
        if media_type == "photo" and media_url:
            urls.append(media_url)
    return urls


def _truncate_one_line(text: str, max_len: int = 140) -> str:
    one_line = " ".join((text or "").split())
    if len(one_line) <= max_len:
        return one_line
    return one_line[: max_len - 3] + "..."


def tweet_to_post(tweet: Any) -> Dict[str, Any]:
    """Convert a Twikit Tweet object to the app's post shape."""
    text = _tweet_text(tweet)
    tweet_id = str(getattr(tweet, "id", "") or "")
    author = _tweet_author(tweet)
    title_text = text or "[No text]"
    title = _truncate_one_line(f"@{author}: {title_text}")
    url = _tweet_url(tweet)
    image_urls = _tweet_image_urls(tweet)
    return {
        "source": "twitter",
        "data": {
            "id": tweet_id,
            "title": title,
            "author": author,
            "score": int(getattr(tweet, "favorite_count", 0) or 0),
            "num_comments": int(getattr(tweet, "reply_count", 0) or 0),
            "url": url,
            "permalink": url,
            "selftext": text,
            "twitter_tweet_id": tweet_id,
            "twitter_media_urls": image_urls,
        },
    }


def build_reply_tree(tweet: Any, max_nodes: int = 300) -> List[Dict[str, Any]]:
    """Build a comment-tree-like structure from tweet replies."""
    root_id = str(getattr(tweet, "id", "") or "")
    remaining = {"count": max_nodes}
    seen: set[str] = set()

    def convert(node: Any) -> Optional[Dict[str, Any]]:
        node_id = str(getattr(node, "id", "") or "")
        if not node_id or node_id in seen or remaining["count"] <= 0:
            return None

        seen.add(node_id)
        remaining["count"] -= 1

        replies_out: List[Dict[str, Any]] = []
        for child in list(getattr(node, "replies", []) or []):
            child_node = convert(child)
            if child_node is not None:
                replies_out.append(child_node)

        replies_out.sort(key=lambda item: item["data"].get("score", 0), reverse=True)

        text = _tweet_text(node) or "[No text]"
        return {
            "data": {
                "id": node_id,
                "author": _tweet_author(node),
                "body": text,
                "score": int(getattr(node, "favorite_count", 0) or 0),
            },
            "replies": replies_out,
            "level": 0,
        }

    roots: List[Dict[str, Any]] = []
    for reply in list(getattr(tweet, "replies", []) or []):
        if str(getattr(reply, "id", "") or "") == root_id:
            continue
        node = convert(reply)
        if node is not None:
            roots.append(node)

    roots.sort(key=lambda item: item["data"].get("score", 0), reverse=True)
    return roots


def build_reply_tree_from_conversation(
    tweets: List[Any], root_tweet_id: str, max_nodes: int = 300
) -> List[Dict[str, Any]]:
    """Build reply tree from conversation search results using in_reply_to links."""
    root_id = str(root_tweet_id or "")
    by_id: Dict[str, Dict[str, Any]] = {}
    children_by_parent: Dict[str, List[str]] = {}

    for tweet in tweets:
        tweet_id = str(getattr(tweet, "id", "") or "")
        if not tweet_id or tweet_id == root_id:
            continue
        if len(by_id) >= max_nodes:
            break

        by_id[tweet_id] = {
            "id": tweet_id,
            "author": _tweet_author(tweet),
            "body": _tweet_text(tweet) or "[No text]",
            "score": int(getattr(tweet, "favorite_count", 0) or 0),
            "in_reply_to": str(getattr(tweet, "in_reply_to", "") or ""),
        }
        parent_id = by_id[tweet_id]["in_reply_to"]
        if parent_id:
            children_by_parent.setdefault(parent_id, []).append(tweet_id)

    def build_node(tweet_id: str) -> Dict[str, Any]:
        data = by_id[tweet_id]
        replies = [build_node(child_id) for child_id in children_by_parent.get(tweet_id, []) if child_id in by_id]
        replies.sort(key=lambda item: item["data"].get("score", 0), reverse=True)
        return {
            "data": {
                "id": data["id"],
                "author": data["author"],
                "body": data["body"],
                "score": data["score"],
            },
            "replies": replies,
            "level": 0,
        }

    root_children = [tweet_id for tweet_id, data in by_id.items() if data.get("in_reply_to") == root_id]
    roots = [build_node(tweet_id) for tweet_id in root_children]
    roots.sort(key=lambda item: item["data"].get("score", 0), reverse=True)
    return roots


class TwitterAPI:
    """Thin Twikit wrapper with cookie-based auth."""

    def __init__(self, cookies_file: str = "cookies.json", locale: str = "en-US"):
        self.cookies_file = cookies_file
        self.locale = locale
        self._client = None

    async def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            from twikit import Client
        except ImportError as exc:
            raise RuntimeError("twikit not installed. Install with: pip install twikit") from exc

        if not os.path.exists(self.cookies_file):
            raise RuntimeError(
                f"Twitter cookies file not found: {self.cookies_file}. "
                "Create it with scripts/test_twikit_cookies.py setup."
            )

        try:
            client = Client(self.locale)
        except TypeError:
            client = Client()

        try:
            client.load_cookies(self.cookies_file)
        except Exception as exc:
            raise RuntimeError(f"Failed to load Twitter cookies: {exc}") from exc

        self._client = client
        return client

    async def get_latest_timeline(self, limit: int = 50):
        client = await self._get_client()
        items = await client.get_latest_timeline(count=limit)
        return list(items)

    async def get_tweet_by_id_safe(self, tweet_id: str):
        """Fetch a tweet by ID with a resilient fallback."""
        client = await self._get_client()
        tweet_id = str(tweet_id)
        try:
            return await client.get_tweet_by_id(tweet_id)
        except Exception:
            try:
                fetched = await client.get_tweets_by_ids([tweet_id])
                if not fetched:
                    return None
                return fetched[0]
            except Exception:
                return None

    async def get_tweet_context_entries(self, tweet: Any) -> List[Dict[str, str]]:
        """Return context tweets (quote/reply target) for detail rendering."""
        entries: List[Dict[str, str]] = []
        has_quote_ref = False

        try:
            quote = getattr(tweet, "quote", None)
            if quote is not None:
                has_quote_ref = True
                entries.append(
                    {
                        "kind": "quoted",
                        "label": "Quoted Tweet",
                        "author": _tweet_author(quote),
                        "text": _tweet_text(quote) or "[No text]",
                        "url": _tweet_url(quote),
                        "twitter_media_urls": _tweet_image_urls(quote),
                    }
                )
        except Exception:
            has_quote_ref = bool(_tweet_quote_id(tweet))

        if not has_quote_ref:
            has_quote_ref = bool(_tweet_quote_id(tweet))
        if has_quote_ref and not any(item.get("kind") == "quoted" for item in entries):
            quote_id = _tweet_quote_id(tweet)
            entries.append(
                {
                    "kind": "quoted_unavailable",
                    "label": "Quoted Tweet",
                    "author": "unavailable",
                    "text": "Quoted tweet exists, but could not be loaded (deleted/protected/unavailable).",
                    "url": f"https://x.com/i/status/{quote_id}" if quote_id else "",
                    "twitter_media_urls": [],
                }
            )

        parent = None
        parent_expected = False
        try:
            reply_to_chain = list(getattr(tweet, "reply_to", []) or [])
            if reply_to_chain:
                parent = reply_to_chain[-1]
                parent_expected = True
        except Exception:
            parent = None
        if parent is None:
            try:
                in_reply_to_id = str(getattr(tweet, "in_reply_to", "") or "")
                if in_reply_to_id:
                    parent_expected = True
                    parent = await self.get_tweet_by_id_safe(in_reply_to_id)
            except Exception:
                parent = None

        if parent is not None:
            entries.append(
                {
                    "kind": "in_reply_to",
                    "label": "Replying To",
                    "author": _tweet_author(parent),
                    "text": _tweet_text(parent) or "[No text]",
                    "url": _tweet_url(parent),
                    "twitter_media_urls": _tweet_image_urls(parent),
                }
            )
        elif parent_expected:
            in_reply_to_id = str(getattr(tweet, "in_reply_to", "") or "")
            entries.append(
                {
                    "kind": "in_reply_to_unavailable",
                    "label": "Replying To",
                    "author": "unavailable",
                    "text": "This tweet is a reply, but the parent tweet could not be loaded.",
                    "url": f"https://x.com/i/status/{in_reply_to_id}" if in_reply_to_id else "",
                    "twitter_media_urls": [],
                }
            )

        # Some "subtweeting" is just linking another tweet URL (not quote/reply metadata).
        self_id = str(getattr(tweet, "id", "") or "")
        existing_ids = {
            str(item.get("url", "")).rstrip("/").split("/")[-1]
            for item in entries
            if item.get("url")
        }
        for ref in _extract_status_links(tweet):
            ref_id = ref.get("id", "")
            if not ref_id or ref_id == self_id or ref_id in existing_ids:
                continue
            referenced = await self.get_tweet_by_id_safe(ref_id)
            if referenced is not None:
                entries.append(
                    {
                        "kind": "referenced",
                        "label": "Referenced Tweet",
                        "author": _tweet_author(referenced),
                        "text": _tweet_text(referenced) or "[No text]",
                        "url": _tweet_url(referenced) or ref.get("url", ""),
                        "twitter_media_urls": _tweet_image_urls(referenced),
                    }
                )
            else:
                entries.append(
                    {
                        "kind": "referenced_unavailable",
                        "label": "Referenced Tweet",
                        "author": ref.get("author", "unknown"),
                        "text": "Referenced tweet link detected, but content could not be loaded.",
                        "url": ref.get("url", ""),
                        "twitter_media_urls": [],
                    }
                )

        return entries

    async def get_tweet_and_reply_tree(self, tweet_id: str):
        client = await self._get_client()
        tweet_id = str(tweet_id)
        try:
            tweet = await client.get_tweet_by_id(tweet_id)
            return tweet, build_reply_tree(tweet)
        except Exception as exc:
            if "itemContent" not in str(exc):
                raise
            # Twikit parser can fail on some thread payload layouts.
            # Fall back to conversation search so callers can still show replies.
            results = await client.search_tweet(
                f"conversation_id:{tweet_id}",
                "Latest",
                count=80,
            )
            tweets = list(results)
            root = None
            for item in tweets:
                if str(getattr(item, "id", "") or "") == tweet_id:
                    root = item
                    break
            if root is None:
                fetched = await client.get_tweets_by_ids([tweet_id])
                root = fetched[0] if fetched else None
            if root is None:
                raise RuntimeError(f"Tweet {tweet_id} not found in fallback flow") from exc
            return root, build_reply_tree_from_conversation(tweets, tweet_id)

    def get_latest_timeline_sync(self, limit: int = 50):
        return _run_async(self.get_latest_timeline(limit))


def _run_async(coro):
    """Run async code from sync context, even if a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    def _runner():
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_runner).result()
