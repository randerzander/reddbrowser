"""AI and article extraction helpers for Android bridge."""

from __future__ import annotations

import re

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

from .http_headers import get_default_headers
from .text_utils import html_to_text


def _strip_thinking(text: str) -> str:
    if not text:
        return text
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", text)
    cleaned = re.sub(r"(?is)<thinking>.*?</thinking>", "", cleaned)
    cleaned = re.sub(r"(?is)```thinking.*?```", "", cleaned)
    cleaned = re.sub(r"(?is)^\s*(reasoning|thinking):.*?(\n\n|$)", "", cleaned)
    return cleaned.strip()


def extract_article_text_sync(url: str) -> str:
    if not url or not url.startswith("http"):
        raise ValueError("Invalid article URL")
    response = requests.get(url, headers=get_default_headers(), timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    article = soup.find("article") or soup.find("main") or soup.body
    content = article.get_text("\n", strip=True) if article else soup.get_text("\n", strip=True)
    title = (soup.title.string or "").strip() if soup.title else ""
    text = html_to_text(content)
    if title and title not in text:
        text = f"{title}\n\n{text}"
    if not text:
        raise ValueError("Article extraction produced no text")
    return text


def _client(api_key: str) -> OpenAI:
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is missing")
    return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")


def summarize_text(text: str, api_key: str, model: str) -> str:
    prompt = f"Please provide a concise summary of the following text:\n\n{text}\n\nSummary:"
    response = _client(api_key).chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1800,
    )
    return _strip_thinking(response.choices[0].message.content or "")


def summarize_comments(comments_text: str, api_key: str, model: str) -> str:
    truncated = comments_text[:6000]
    prompt = (
        "Summarize the key points, sentiment, and disagreements in these comments. "
        "Return 5-8 concise bullets.\n\n"
        f"{truncated}\n\nSummary:"
    )
    response = _client(api_key).chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1600,
    )
    return _strip_thinking(response.choices[0].message.content or "")


def ask_ai(context: str, user_prompt: str, api_key: str, model: str) -> str:
    prompt = f"{context}\n\nUser question:\n{user_prompt}\n\nProvide a direct helpful answer."
    response = _client(api_key).chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1200,
    )
    return _strip_thinking(response.choices[0].message.content or "")
