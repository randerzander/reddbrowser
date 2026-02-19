"""Text conversion utilities."""

from __future__ import annotations

import html as html_lib
import re


def html_to_text(value: str) -> str:
    """Best-effort conversion of HTML content to plain text."""
    if not value:
        return ""
    value = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", value)
    value = re.sub(r"(?i)<br\\s*/?>", "\n", value)
    value = re.sub(r"(?i)</p>", "\n\n", value)
    value = re.sub(r"(?s)<.*?>", " ", value)
    value = html_lib.unescape(value)
    value = re.sub(r"[ \\t\\r\\f\\v]+", " ", value)
    value = re.sub(r"\\n\\n\\n+", "\n\n", value)
    return value.strip()
