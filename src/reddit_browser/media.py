"""Media handling and VLM integration for Reddit Browser."""

import os
import base64
import httpx
import tempfile
import subprocess
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

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
        async with httpx.AsyncClient(headers={"User-Agent": "RedditBrowser/0.1.0"}) as client:
            response = await client.get(url)
            response.raise_for_status()

            parsed_url = urlparse(url)
            file_ext = os.path.splitext(parsed_url.path)[1]
            if not file_ext:
                content_type = response.headers.get('content-type', '')
                if 'jpeg' in content_type or 'jpg' in content_type:
                    file_ext = '.jpg'
                elif 'png' in content_type:
                    file_ext = '.png'
                elif 'gif' in content_type:
                    file_ext = '.gif'
                else:
                    file_ext = '.png'

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(response.content)
                return tmp_file.name
    except Exception:
        return None

def open_image_in_viewer(image_path: str) -> Optional[str]:
    """Open an image in a GUI viewer."""
    viewers = [
        ['feh', image_path],
        ['xdg-open', image_path],
        ['eog', image_path],
        ['gpicview', image_path],
    ]

    for viewer_cmd in viewers:
        try:
            subprocess.Popen(viewer_cmd)
            return viewer_cmd[0]
        except FileNotFoundError:
            continue
    return None

async def generate_image_description(url: str = None, image_path: str = None) -> str:
    """Generate a description of the image using OpenRouter API."""
    if not OPENAI_AVAILABLE:
        return "Error: OpenAI library not installed."

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "Error: OPENROUTER_API_KEY not set."

    try:
        if image_path:
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
        elif url:
            async with httpx.AsyncClient(headers={"User-Agent": "RedditBrowser/0.1.0"}) as client:
                response = await client.get(url)
                response.raise_for_status()
                image_data = base64.b64encode(response.content).decode('utf-8')
        else:
            return "Error: No image source provided."

        # Determine mime type
        mime_type = 'image/jpeg'
        if url:
            if '.png' in url.lower(): mime_type = 'image/png'
            elif '.gif' in url.lower(): mime_type = 'image/gif'

        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

        # Use the same model as before
        model = os.getenv("VLM_MODEL", "qwen/qwen-2.5-vl-7b-instruct:free")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}}
                    ]
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating description: {str(e)}"


async def generate_text_summary(text: str) -> str:
    """Generate a summary of the provided text using the Nvidia Nemotron model via OpenRouter API."""
    if not OPENAI_AVAILABLE:
        return "Error: OpenAI library not installed."

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return "Error: OPENROUTER_API_KEY not set."

    try:
        # Limit text length to avoid token limits
        if len(text) > 2000:  # Rough character limit
            text = text[:2000] + "... (truncated)"

        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

        # Use the Nvidia Nemotron model
        model = os.getenv("TEXT_SUMMARY_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"Please provide a concise summary of the following text:\n\n{text}\n\nSummary:"
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating summary: {str(e)}"
