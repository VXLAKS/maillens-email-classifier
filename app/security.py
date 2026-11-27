from __future__ import annotations

import re
import ipaddress
from urllib.parse import urlparse
from typing import List

# Compile regex patterns once
_PROMPT_INJECTION_REGEXES = [
    re.compile(p, re.IGNORECASE) for p in [
        r"ignore\s+previous\s+instructions",
        r"system\s+prompt",
        r"developer\s+message",
        r"act\s+as\s+",
        r"follow\s+these\s+rules",
        r"do\s+not\s+tell\s+anyone",
    ]
]

# Keep "you are chatgpt" separate (too many false positives)
_CHATGPT_MENTION_RE = re.compile(r"\byou\s+are\s+chatgpt\b", re.IGNORECASE)

PHISHING_KEYWORDS = [
    # EN
    "verify your account", "password", "urgent", "immediately", "suspend",
    "confirm your identity", "wire transfer", "invoice overdue", "click the link",
    # RU (optional but useful)
    "подтвердите аккаунт", "подтвердите учетную запись", "пароль", "срочно", "немедленно",
    "заблокирован", "приостановлен", "подтвердите личность", "перевод", "счет просрочен",
    "перейдите по ссылке", "нажмите на ссылку",
]

_whitespace_re = re.compile(r"[\r\n\t]+")
_many_dashes_re = re.compile(r"-{3,}")


def detect_prompt_injection(text: str) -> bool:
    """
    Simple heuristic detection.
    Returns True if text looks like instructions to override system/developer behavior.
    """
    t = (text or "")
    if not t:
        return False

    # Strong patterns
    if any(r.search(t) for r in _PROMPT_INJECTION_REGEXES):
        return True

    # Weaker signal: explicit "you are chatgpt" alone is too noisy,
    # but if combined with "rules"/"instructions" style text - mark.
    if _CHATGPT_MENTION_RE.search(t) and ("instruction" in t.lower() or "rules" in t.lower()):
        return True

    return False


def _is_ip_host(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except Exception:
        return False


def phishing_risk_score(text: str, urls: List[str]) -> float:
    """
    Heuristic risk score in [0, 1].
    """
    t = (text or "").lower()
    score = 0.0

    # Keyword hits
    for kw in PHISHING_KEYWORDS:
        if kw in t:
            score += 0.08

    # URL volume signal (cap at 10)
    score += min(len(urls or []), 10) * 0.04

    # Suspicious URL signals
    for u in (urls or [])[:20]:
        try:
            parsed = urlparse(u)
            host = (parsed.hostname or "").lower().strip()
            if not host:
                continue

            # punycode domains
            if host.startswith("xn--"):
                score += 0.10

            # IP-based URLs
            if _is_ip_host(host):
                score += 0.10

            # many dashes often used in lookalike domains
            if _many_dashes_re.search(host) or host.count("-") >= 3:
                score += 0.05

        except Exception:
            continue

    return max(0.0, min(1.0, score))


def redact_for_logs(text: str, max_len: int = 240) -> str:
    if not text:
        return ""
    t = _whitespace_re.sub(" ", text).strip()
    if len(t) > max_len:
        return t[:max_len] + "…"
    return t
