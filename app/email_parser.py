from __future__ import annotations

import email
import email.policy
import io
import mimetypes
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

from .preprocessing import html_to_text, normalize_text, extract_urls, clip
from .config import settings


# ----------------------------
# Attachment text extractors
# ----------------------------
def _extract_text_from_pdf(data: bytes, max_pages: int) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        out: List[str] = []

        n_pages = len(reader.pages)
        for i in range(min(int(max_pages), n_pages)):
            try:
                out.append(reader.pages[i].extract_text() or "")
            except Exception:
                continue

        return normalize_text(" ".join(out))
    except Exception:
        return ""


def _extract_text_from_docx(data: bytes) -> str:
    try:
        import docx  # python-docx

        doc = docx.Document(io.BytesIO(data))
        out = [p.text for p in doc.paragraphs if p.text]
        return normalize_text(" ".join(out))
    except Exception:
        return ""


def _extract_text_from_txt(data: bytes) -> str:
    for enc in ("utf-8", "utf-16", "cp1251", "latin-1"):
        try:
            return normalize_text(data.decode(enc, errors="ignore"))
        except Exception:
            continue
    return ""


def _attachment_text(filename: str, content_type: str, data: bytes) -> Tuple[str, str]:
    """
    Returns (extracted_text, method)
    """
    fn = (filename or "").lower()
    ct = (content_type or "").lower()

    if fn.endswith(".pdf") or ct == "application/pdf":
        return _extract_text_from_pdf(data, settings.MAX_PDF_PAGES), "pdf"

    if fn.endswith(".docx") or ct in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ):
        return _extract_text_from_docx(data), "docx"

    if fn.endswith(".txt") or ct.startswith("text/"):
        return _extract_text_from_txt(data), "text"

    return "", "unsupported"


# ----------------------------
# Data models
# ----------------------------
@dataclass
class AttachmentInfo:
    filename: str
    content_type: str
    size_bytes: int
    extracted_text: str
    extraction: str


@dataclass
class ParsedEmail:
    filename: str
    subject: str
    from_: str
    to: str
    date: str
    body_text: str
    urls: List[str]
    attachments: List[AttachmentInfo]
    errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["from"] = d.pop("from_")
        return d


# ----------------------------
# Parsers
# ----------------------------
def parse_eml(filename: str, data: bytes) -> ParsedEmail:
    errors: List[str] = []

    msg = email.message_from_bytes(data, policy=email.policy.default)

    subject = str(msg.get("subject", "") or "")
    from_ = str(msg.get("from", "") or "")
    to = str(msg.get("to", "") or "")
    date = str(msg.get("date", "") or "")

    body_plain = ""
    body_html = ""
    attachments: List[AttachmentInfo] = []

    def handle_part(part) -> None:
        nonlocal body_plain, body_html, attachments, errors

        ctype = part.get_content_type()
        disp = part.get_content_disposition()  # inline/attachment/None
        filename_part = part.get_filename()

        # Body parts
        if ctype == "text/plain" and disp != "attachment":
            try:
                body_plain += part.get_content() or ""
            except Exception as e:
                errors.append(f"text/plain decode: {e}")
            return

        if ctype == "text/html" and disp != "attachment":
            try:
                body_html += part.get_content() or ""
            except Exception as e:
                errors.append(f"text/html decode: {e}")
            return

        # Attachments (or anything with filename)
        if disp == "attachment" or filename_part:
            try:
                payload = part.get_payload(decode=True) or b""
                fn = filename_part or "attachment"

                if len(payload) > settings.MAX_ATTACHMENT_BYTES:
                    errors.append(f"attachment too large: {fn} ({len(payload)} bytes)")
                    extracted, method = "", "skipped_too_large"
                else:
                    extracted, method = _attachment_text(fn, ctype, payload)

                attachments.append(
                    AttachmentInfo(
                        filename=fn,
                        content_type=ctype,
                        size_bytes=len(payload),
                        extracted_text=clip(extracted, 4000),
                        extraction=method,
                    )
                )
            except Exception as e:
                errors.append(f"attachment parse error: {e}")

    if msg.is_multipart():
        for part in msg.walk():
            if part.is_multipart():
                continue
            handle_part(part)
    else:
        ctype = msg.get_content_type()
        try:
            content = msg.get_content() or ""
        except Exception:
            content = ""
        if ctype == "text/html":
            body_html = content
        else:
            body_plain = content

    body = body_plain.strip()
    if not body and body_html:
        body = html_to_text(body_html)

    body = normalize_text(body)
    subject = normalize_text(subject)

    urls = extract_urls(subject + " " + body)

    return ParsedEmail(
        filename=filename,
        subject=clip(subject, 500),
        from_=clip(from_, 300),
        to=clip(to, 300),
        date=clip(date, 100),
        body_text=clip(body, settings.MAX_EMAIL_CHARS),
        urls=urls[:50],
        attachments=attachments[: settings.MAX_ATTACHMENTS],
        errors=errors,
    )


def parse_msg(filename: str, data: bytes) -> ParsedEmail:
    """
    Best-effort .msg support (optional dependency extract-msg).
    """
    errors: List[str] = []

    try:
        import extract_msg
    except Exception:
        errors.append("extract-msg not installed; .msg parsing unavailable")
        return ParsedEmail(
            filename=filename,
            subject="",
            from_="",
            to="",
            date="",
            body_text="",
            urls=[],
            attachments=[],
            errors=errors,
        )

    try:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".msg", delete=True) as f:
            f.write(data)
            f.flush()

            m = extract_msg.Message(f.name)
            try:
                m.process()

                subject = normalize_text(getattr(m, "subject", "") or "")
                from_ = normalize_text(getattr(m, "sender", "") or "")
                to = normalize_text(getattr(m, "to", "") or "")
                date = normalize_text(getattr(m, "date", "") or "")
                body = normalize_text(getattr(m, "body", "") or "")

                atts: List[AttachmentInfo] = []
                try:
                    for att in (getattr(m, "attachments", None) or [])[: settings.MAX_ATTACHMENTS]:
                        fn = getattr(att, "longFilename", None) or getattr(att, "shortFilename", None) or "attachment"
                        payload = getattr(att, "data", None) or b""

                        if len(payload) > settings.MAX_ATTACHMENT_BYTES:
                            errors.append(f"attachment too large: {fn} ({len(payload)} bytes)")
                            extracted, method = "", "skipped_too_large"
                            ctype = mimetypes.guess_type(fn)[0] or ""
                        else:
                            ctype = mimetypes.guess_type(fn)[0] or ""
                            extracted, method = _attachment_text(fn, ctype, payload)

                        atts.append(
                            AttachmentInfo(
                                filename=fn,
                                content_type=ctype,
                                size_bytes=len(payload),
                                extracted_text=clip(extracted, 4000),
                                extraction=method,
                            )
                        )
                except Exception as e:
                    errors.append(f"msg attachments parse error: {e}")

                urls = extract_urls(subject + " " + body)

                return ParsedEmail(
                    filename=filename,
                    subject=clip(subject, 500),
                    from_=clip(from_, 300),
                    to=clip(to, 300),
                    date=clip(date, 100),
                    body_text=clip(body, settings.MAX_EMAIL_CHARS),
                    urls=urls[:50],
                    attachments=atts,
                    errors=errors,
                )
            finally:
                # important: extract_msg may keep file handles
                try:
                    m.close()
                except Exception:
                    pass
    except Exception as e:
        errors.append(f"msg parse failed: {e}")
        return ParsedEmail(
            filename=filename,
            subject="",
            from_="",
            to="",
            date="",
            body_text="",
            urls=[],
            attachments=[],
            errors=errors,
        )


def parse_email(filename: str, data: bytes) -> ParsedEmail:
    fn = (filename or "").lower()

    if fn.endswith(".eml"):
        return parse_eml(filename or "email.eml", data)

    if fn.endswith(".msg"):
        return parse_msg(filename or "email.msg", data)

    # Fallback: treat as plain text
    try:
        body = data.decode("utf-8", errors="ignore")
    except Exception:
        body = ""

    body = normalize_text(body)
    urls = extract_urls(body)

    return ParsedEmail(
        filename=filename or "email.txt",
        subject="",
        from_="",
        to="",
        date="",
        body_text=clip(body, settings.MAX_EMAIL_CHARS),
        urls=urls[:50],
        attachments=[],
        errors=["unknown extension; treated as text"],
    )
