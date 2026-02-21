#!/usr/bin/env python3
"""Standalone Twikit login test using credentials from .twitter_creds."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Dict


def load_creds(path: str) -> Dict[str, str]:
    """Load KEY=VALUE credentials from a flat file."""
    creds: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            creds[key.strip()] = value.strip()
    return creds


def resolve_auth_info_2(creds: Dict[str, str]) -> str | None:
    """Pick an email/phone-like secondary identifier for Twikit login."""
    for key in ("AUTH_INFO_2", "EMAIL", "PHONE", "PHONE_NUMBER"):
        value = creds.get(key)
        if value:
            return value
    return None


async def run_login(creds_path: str, locale: str, cookies_file: str) -> int:
    try:
        from twikit import Client
    except ImportError:
        print(
            "twikit is not installed. Install it with: pip install twikit",
            file=sys.stderr,
        )
        return 2

    if not os.path.exists(creds_path):
        print(f"Credentials file not found: {creds_path}", file=sys.stderr)
        return 2

    creds = load_creds(creds_path)
    username = creds.get("USERNAME")
    password = creds.get("PASSWORD")
    auth_info_2 = resolve_auth_info_2(creds)

    missing = [name for name, value in (("USERNAME", username), ("PASSWORD", password)) if not value]
    if missing:
        print(
            "Missing required credentials: "
            + ", ".join(missing)
            + f" in {creds_path}",
            file=sys.stderr,
        )
        return 2

    try:
        client = Client(locale)
    except TypeError:
        # Fallback for versions that don't accept locale in the constructor.
        client = Client()

    try:
        await client.login(
            auth_info_1=username,
            auth_info_2=auth_info_2,
            password=password,
            cookies_file=cookies_file,
        )
    except Exception as exc:
        print(f"Twikit login failed: {exc}", file=sys.stderr)
        return 1

    print("Twikit login successful.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Twikit login with .twitter_creds")
    parser.add_argument(
        "--creds",
        default=".twitter_creds",
        help="Path to credentials file (default: .twitter_creds)",
    )
    parser.add_argument(
        "--locale",
        default="en-US",
        help="Client locale for Twikit constructor (default: en-US)",
    )
    parser.add_argument(
        "--cookies-file",
        default="cookies.json",
        help="Path to cookies file to write/read during login (default: cookies.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(run_login(args.creds, args.locale, args.cookies_file))


if __name__ == "__main__":
    raise SystemExit(main())
