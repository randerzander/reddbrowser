"""Media handling and VLM integration for Reddit Browser."""

import os
import httpx
from .http_headers import get_default_headers
import tempfile
import subprocess
from typing import Optional
from urllib.parse import urlparse
import asyncio
import requests
import re
from .text_utils import html_to_text

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    # Preferred parser from pyreadability README usage
    from pyreadability import Readability as PyReadability
except ImportError:
    PyReadability = None

try:
    # Backward-compatible fallback for environments with readability-lxml
    from readability import Document as ReadabilityDocument
except ImportError:
    try:
        from readability.readability import Document as ReadabilityDocument
    except ImportError:
        ReadabilityDocument = None

READABILITY_AVAILABLE = (PyReadability is not None) or (ReadabilityDocument is not None)

# Common image extensions
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg']

def is_image_url(url: str) -> bool:
    """Check if the URL points to an image."""
    if not url:
        return False

    url_lower = url.lower()
    
    # Check for common image hosting domains
    image_domains = [
        'i.redd.it', 'i.imgur.com', 'imgur.com', 'flickr.com',
        'instagram.com', 'twitter.com', 'facebook.com',
        'cdn.discordapp.com', 'media.discordapp.net'
    ]

    for domain in image_domains:
        if domain in url_lower:
            if 'imgur.com' in domain and any(x in url_lower for x in ['/a/', '/gallery/', 'album']):
                return False
            return True

    for ext in IMAGE_EXTENSIONS:
        if url_lower.endswith(ext):
            return True

    return False

async def download_image(url: str) -> Optional[str]:
    """Download an image to a temporary file and return its path."""
    try:
        async with httpx.AsyncClient(headers=get_default_headers()) as client:
            response = await client.get(url)
            response.raise_for_status()
            return _write_temp_image(response.content, url, response.headers.get('content-type', ''))
    except Exception:
        return None


def download_image_sync(url: str) -> Optional[str]:
    """Download an image to a temporary file and return its path."""
    try:
        response = requests.get(url, headers=get_default_headers(), timeout=10)
        response.raise_for_status()
        return _write_temp_image(response.content, url, response.headers.get('content-type', ''))
    except Exception:
        return None

def open_image_in_viewer(image_path: str) -> Optional[str]:
    """Open an image in a GUI viewer."""
    viewers = [
        ['feh', image_path],
        ['xdg-open', image_path],
        ['eog', image_path],
        ['gpicview', image_path],
        ['gthumb', image_path],
        ['ristretto', image_path],
        ['shotwell', image_path],
    ]

    for viewer_cmd in viewers:
        try:
            subprocess.Popen(viewer_cmd)
            return viewer_cmd[0]
        except FileNotFoundError:
            continue
    return None

def _determine_image_extension(url: str, content_type: str) -> str:
    parsed_url = urlparse(url)
    file_ext = os.path.splitext(parsed_url.path)[1].lower()
    if file_ext:
        return file_ext
    content_type = (content_type or "").lower()
    if 'jpeg' in content_type or 'jpg' in content_type:
        return '.jpg'
    if 'png' in content_type:
        return '.png'
    if 'gif' in content_type:
        return '.gif'
    if 'webp' in content_type:
        return '.webp'
    if 'bmp' in content_type:
        return '.bmp'
    if 'svg' in content_type:
        return '.svg'
    return '.png'


def _write_temp_image(content: bytes, url: str, content_type: str) -> str:
    file_ext = _determine_image_extension(url, content_type)
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(content)
        return tmp_file.name

def extract_article_text_sync(url: str) -> str:
    """Extract main article text from a URL using requests + pyreadability."""
    if not READABILITY_AVAILABLE:
        return (
            "Error: pyreadability not installed. "
            "Install dependency: pip install git+https://github.com/randerzander/pyreadability.git"
        )
    if not url or not url.startswith("http"):
        return "Error: Invalid URL."

    try:
        response = requests.get(url, headers=get_default_headers(), timeout=10)
        response.raise_for_status()

        if PyReadability is not None:
            reader = PyReadability(response.text, url=url)
            article = reader.parse()
            title = (article.get("title") or "").strip()
            summary_html = article.get("content") or ""
        else:
            doc = ReadabilityDocument(response.text)
            title = (doc.title() or "").strip()
            try:
                summary_html = doc.summary(html_partial=True)
            except TypeError:
                summary_html = doc.summary()

        text = html_to_text(summary_html)
        if title and title not in text:
            text = f"{title}\n\n{text}"

        if not text.strip():
            return "Error: Article extraction produced no text."
        return text.strip()
    except Exception as e:
        return f"Error fetching article: {str(e)}"

async def extract_article_text(url: str) -> str:
    """Async wrapper to extract article text without blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, extract_article_text_sync, url)

async def generate_text_summary(text: str) -> str:
    """Generate a summary of the provided text using the Nvidia Nemotron model via OpenRouter API."""
    if not OPENAI_AVAILABLE:
        return "Error: OpenAI library not installed."

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "Error: OPENROUTER_API_KEY not set."

    try:
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

        # Use the Nvidia Nemotron model
        model = os.getenv("TEXT_SUMMARY_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free")

        def _call():
            return client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Please provide a concise summary of the following text:\n\n{text}\n\nSummary:"
                    }
                ],
                max_tokens=3000
            )

        response = await asyncio.to_thread(_call)
        return _strip_thinking(response.choices[0].message.content)
    except Exception as e:
        return f"Error generating summary: {str(e)}"

async def generate_comments_summary(comments_text: str) -> str:
    """Generate a concise summary of the top comments using the Nvidia Nemotron model via OpenRouter API."""
    if not OPENAI_AVAILABLE:
        return "Error: OpenAI library not installed."

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "Error: OPENROUTER_API_KEY not set."

    try:
        # Limit text length to avoid token limits
        if len(comments_text) > 4000:  # Rough character limit
            comments_text = comments_text[:4000] + "... (truncated)"

        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

        # Use the same model as text summarization
        model = os.getenv("TEXT_SUMMARY_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free")

        prompt = (
            "Summarize the key points, sentiments, and disagreements in these top comments. "
            "Keep it concise and readable in 5-8 short bullets.\n\n"
            f"{comments_text}\n\n"
            "Summary:"
        )

        def _call():
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )

        response = await asyncio.to_thread(_call)
        return _strip_thinking(response.choices[0].message.content)
    except Exception as e:
        return f"Error generating comment summary: {str(e)}"


async def generate_ai_response(prompt: str) -> str:
    """Generate a response to a custom prompt using the OpenRouter API."""
    if not OPENAI_AVAILABLE:
        return "Error: OpenAI library not installed."

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "Error: OPENROUTER_API_KEY not set."

    try:
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

        # Use the same model as text summarization to avoid rate-limit disparities
        model = os.getenv("AI_RESPONSE_MODEL", os.getenv("TEXT_SUMMARY_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free"))

        def _call():
            return client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1000
            )

        response = await asyncio.to_thread(_call)
        return _strip_thinking(response.choices[0].message.content)
    except Exception as e:
        return f"Error generating AI response: {str(e)}"
def _strip_thinking(text: str) -> str:
    """Remove model thinking traces from responses."""
    if not text:
        return text
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", text)
    cleaned = re.sub(r"(?is)<thinking>.*?</thinking>", "", cleaned)
    cleaned = re.sub(r"(?is)```thinking.*?```", "", cleaned)
    cleaned = re.sub(r"(?is)^\\s*(reasoning|thinking):.*?(\\n\\n|$)", "", cleaned)
    return cleaned.strip()
