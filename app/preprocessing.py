from __future__ import annotations

import re
from typing import List
from bs4 import BeautifulSoup

_whitespace_re = re.compile(r"\s+")
# Keep URL matching simple but robust
_url_re = re.compile(r"(https?://[^\s<>\"]+)", re.IGNORECASE)
# Characters часто прилипают к URL в тексте/HTML
_url_trailing_punct_re = re.compile(r"[)\],.?!;:]+$")


def html_to_text(html: str) -> str:
    """
    Convert HTML -> plain text safely.
    """
    if not isinstance(html, str) or not html.strip():
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def normalize_text(text: str) -> str:
    """
    Normalize whitespace and remove null bytes.
    """
    if not isinstance(text, str):
        return ""
    text = text.replace("\x00", " ")
    text = _whitespace_re.sub(" ", text).strip()
    return text


def _clean_url(u: str) -> str:
    u = (u or "").strip()
    # strip typical trailing punctuation
    u = _url_trailing_punct_re.sub("", u)
    return u


def extract_urls(text: str) -> List[str]:
    """
    Extract unique URLs preserving order.
    """
    if not isinstance(text, str) or not text:
        return []
    raw = _url_re.findall(text)
    cleaned: List[str] = []
    seen = set()
    for u in raw:
        cu = _clean_url(u)
        if cu and cu not in seen:
            seen.add(cu)
            cleaned.append(cu)
    return cleaned


def clip(text: str, max_chars: int) -> str:
    """
    Clip string to max_chars with a marker.
    """
    if not isinstance(text, str):
        return ""
    if len(text) <= int(max_chars):
        return text
    return text[: int(max_chars)] + " …[truncated]"
