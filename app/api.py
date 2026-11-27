from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Any, Dict
import uuid
import time
import csv
import json
from pathlib import Path
import traceback

from app.email_parser import parse_email
from app.classifier import classify, make_decision, RawClassificationResult, DecisionResult
from app.security import detect_prompt_injection, phishing_risk_score

app = FastAPI(title="MailLens API")

# --- paths/config ---
BASE_DIR = Path(__file__).resolve().parents[1]  # repo root
RESULTS_DIR = BASE_DIR / "results"
CSV_PATH = RESULTS_DIR / "results.csv"
JSONL_PATH = RESULTS_DIR / "results.jsonl"  # optional if you want JSONL too
RESULTS_DIR.mkdir(exist_ok=True)

DEFAULT_THRESHOLD = 0.30

# Stable CSV schema (DON'T change per-row)
CSV_FIELDS = [
    "request_id",
    "filename",
    "predicted_label",
    "is_undefined",
    "max_similarity",
    "threshold",
    "topN_labels",
    "topN_similarity",
    "topN_confidence",
    "scores_by_category",
    "urls_count",
    "attachments_count",
    "prompt_injection_flag",
    "phishing_risk_score",
    "parse_errors",
    "latency_sec",
    "created_at_utc",
]


def _to_py(x: Any) -> Any:
    """Convert numpy/torch/pydantic types into JSON-serializable Python primitives."""
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None  # noqa

    if np is not None and isinstance(x, getattr(np, "generic", ())):
        return x.item()

    try:
        import torch  # type: ignore
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return x.item()
            return x.detach().cpu().tolist()
    except Exception:
        pass

    if isinstance(x, dict):
        return {str(k): _to_py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_py(v) for v in x]
    return x


def _parsed_to_dict(parsed: Any) -> Dict[str, Any]:
    """Normalize ParsedEmail (dict/dataclass/pydantic) -> dict"""
    if parsed is None:
        return {}
    if isinstance(parsed, dict):
        return parsed
    if hasattr(parsed, "model_dump"):
        try:
            return parsed.model_dump()
        except Exception:
            pass
    if hasattr(parsed, "to_dict"):
        try:
            return parsed.to_dict()
        except Exception:
            pass
    if hasattr(parsed, "__dict__"):
        try:
            return dict(parsed.__dict__)
        except Exception:
            pass
    return {}


def _collect_attachment_text(d: Dict[str, Any]) -> List[str]:
    """
    email_parser.py returns attachments as list of dicts/dataclasses, often with 'extracted_text'.
    We also support a few legacy shapes.
    """
    texts: List[str] = []

    # New shape: attachments: [{..., extracted_text, ...}, ...]
    atts = d.get("attachments")
    if isinstance(atts, list):
        for a in atts:
            if isinstance(a, dict):
                t = a.get("extracted_text") or a.get("text")
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())
            else:
                t = getattr(a, "extracted_text", None) or getattr(a, "text", None)
                if isinstance(t, str) and t.strip():
                    texts.append(t.strip())

    # Legacy flat fields
    for k in ("attachments_text", "attachment_texts", "attachments_body"):
        val = d.get(k)
        if isinstance(val, str) and val.strip():
            texts.append(val.strip())
        elif isinstance(val, list):
            cleaned = [v.strip() for v in val if isinstance(v, str) and v.strip()]
            texts.extend(cleaned)

    # de-dup preserve order
    seen = set()
    uniq: List[str] = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _extract_text(parsed: Any) -> str:
    """
    Build classification text:
      SUBJECT + body_text + optional attachments text + urls.
    """
    d = _parsed_to_dict(parsed)
    parts: List[str] = []

    subj = d.get("subject") or d.get("Subject") or d.get("email_subject")
    if isinstance(subj, str) and subj.strip():
        parts.append(f"SUBJECT: {subj.strip()}")

    # PRIORITIZE body_text (matches your email_parser.py)
    candidates = [
        "body_text",
        "text",
        "full_text",
        "combined_text",
        "content",
        "plain_text",
        "body",
        "html_text",
        "subject_and_body",
    ]

    for key in candidates:
        val = d.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
            break

    # extra bodies if exist
    for k in ("body_plain", "body_html", "plain", "html"):
        val = d.get(k)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())

    # attachment extracted text
    att_texts = _collect_attachment_text(d)
    if att_texts:
        parts.append("ATTACHMENTS_TEXT:\n" + "\n\n".join(att_texts))

    urls = d.get("urls") or d.get("links")
    if isinstance(urls, list) and urls:
        parts.append("URLS:\n" + "\n".join([str(u) for u in urls]))

    return "\n\n".join(parts).strip()


def _extract_counts_and_security(email_text: str, parsed: Any) -> Dict[str, Any]:
    """
    Consolidate counts + security signals.
    """
    d = _parsed_to_dict(parsed)

    urls = d.get("urls") or d.get("links") or []
    attachments = d.get("attachments") or d.get("attachments_meta") or []

    urls_count = d.get("urls_count")
    if urls_count is None:
        urls_count = len(urls) if isinstance(urls, list) else 0

    attachments_count = d.get("attachments_count")
    if attachments_count is None:
        attachments_count = len(attachments) if isinstance(attachments, list) else 0

    errors = d.get("errors") or d.get("parse_errors") or []
    if errors is None:
        errors = []

    # security from extracted text
    try:
        inj = bool(detect_prompt_injection(email_text))
    except Exception:
        inj = False

    try:
        prs = float(phishing_risk_score(email_text, urls if isinstance(urls, list) else []))
    except Exception:
        prs = 0.0

    return {
        "urls_count": int(urls_count or 0),
        "attachments_count": int(attachments_count or 0),
        "parse_errors": errors,
        "prompt_injection_flag": inj,
        "phishing_risk_score": prs,
    }


def _json_str(x: Any) -> str:
    """Store lists/dicts safely inside CSV cell."""
    try:
        return json.dumps(_to_py(x), ensure_ascii=False)
    except Exception:
        return json.dumps(str(x), ensure_ascii=False)


def _append_results(row: Dict[str, Any]) -> None:
    # Ensure stable field set and safe CSV values
    safe_row: Dict[str, Any] = {}
    for k in CSV_FIELDS:
        v = row.get(k)
        if isinstance(v, (list, dict)):
            safe_row[k] = _json_str(v)
        else:
            safe_row[k] = _to_py(v)

    write_header = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(safe_row)

    # Optional JSONL export (comment out if не надо)
    try:
        with JSONL_PATH.open("a", encoding="utf-8") as jf:
            jf.write(json.dumps(_to_py(row), ensure_ascii=False) + "\n")
    except Exception:
        pass


# -----------------------------
# Health
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "mode": "batch-ready"}


# -----------------------------
# Reset results (button endpoint)
# -----------------------------
@app.post("/results/reset")
def reset_results():
    deleted: List[str] = []
    for p in (CSV_PATH, JSONL_PATH):
        try:
            if p.exists():
                p.unlink()
                deleted.append(p.name)
        except Exception as e:
            return JSONResponse(
                {"status": "error", "detail": f"Failed to delete {p.name}: {e}"},
                status_code=500,
            )
    return {"status": "ok", "deleted": deleted}


# -----------------------------
# Batch classify
# -----------------------------
@app.post("/classify/batch")
async def classify_batch(
    files: List[UploadFile] = File(...),
    categories: str = Form(...),
    threshold: float = Form(DEFAULT_THRESHOLD),
    top_n: int = Form(5),
):
    category_list = [c.strip() for c in categories.split(",") if c.strip()]
    if not category_list:
        return JSONResponse({"detail": "categories is empty"}, status_code=400)

    results: List[Dict[str, Any]] = []

    for file in files:
        start = time.time()
        request_id = str(uuid.uuid4())

        try:
            raw_bytes = await file.read()
            parsed = parse_email(file.filename, raw_bytes)

            email_text = _extract_text(parsed)
            if not email_text or len(email_text.strip()) < 10:
                return JSONResponse(
                    {"detail": f"Empty/too short extracted text for file={file.filename}"},
                    status_code=400,
                )

            meta = _extract_counts_and_security(email_text=email_text, parsed=parsed)

            raw_res: RawClassificationResult = classify(
                email_text=email_text,
                categories=category_list,
                top_n=int(top_n),
            )
            dec: DecisionResult = make_decision(raw_res, threshold=float(threshold))

            row = {
                "request_id": request_id,
                "filename": file.filename,

                # MVP contract
                "predicted_label": dec.predicted_label,
                "is_undefined": dec.is_undefined,
                "max_similarity": float(dec.max_similarity),
                "threshold": float(dec.threshold),

                # ranking diagnostics
                "topN_labels": raw_res.topN_labels,
                "topN_similarity": raw_res.topN_similarity,
                "topN_confidence": raw_res.topN_confidence,
                "scores_by_category": raw_res.scores_by_category,

                # parser/meta + security
                "urls_count": meta["urls_count"],
                "attachments_count": meta["attachments_count"],
                "prompt_injection_flag": meta["prompt_injection_flag"],
                "phishing_risk_score": meta["phishing_risk_score"],
                "parse_errors": meta["parse_errors"],

                # ops
                "latency_sec": round(time.time() - start, 4),
                "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

            _append_results(row)
            results.append(_to_py(row))

        except Exception as e:
            print("ERROR in /classify/batch:", traceback.format_exc())
            return JSONResponse(
                content={
                    "request_id": request_id,
                    "filename": getattr(file, "filename", None),
                    "detail": str(e),
                },
                status_code=500,
            )

    return JSONResponse(content=_to_py(results))


# -----------------------------
# Export CSV
# -----------------------------
@app.get("/export/results.csv")
def export_csv():
    if not CSV_PATH.exists():
        return JSONResponse({"detail": "results.csv not found"}, status_code=404)
    return FileResponse(CSV_PATH, media_type="text/csv", filename="results.csv")
