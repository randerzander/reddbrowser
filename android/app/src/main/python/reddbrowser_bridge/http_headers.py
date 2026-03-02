"""HTTP header helpers."""

from __future__ import annotations

import random

DEFAULT_USER_AGENTS = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
)


def get_default_headers(user_agent: str | None = None) -> dict[str, str]:
    return {"User-Agent": user_agent or random.choice(DEFAULT_USER_AGENTS)}
