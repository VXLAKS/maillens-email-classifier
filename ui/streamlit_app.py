import os
import json
from typing import Any, List, Tuple

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="MailLens", layout="wide")
st.title("MailLens — batch классификация писем")

# -----------------------------
# Inputs
# -----------------------------
uploaded_files = st.file_uploader(
    "Загрузите письма (.eml / .msg)",
    type=["eml", "msg"],
    accept_multiple_files=True,
)

categories = st.text_input(
    "Категории (через запятую)",
    "Финансы, HR, Техподдержка, Юридические вопросы, Спам/фишинг",
)

threshold = st.slider("Порог «Не определена»", 0.0, 1.0, 0.30, 0.05)
top_n = st.number_input("Top-N категорий", min_value=1, max_value=20, value=5, step=1)
only_undef = st.checkbox("Показать только «Не определена»", value=False)


def build_multipart(files) -> List[Tuple[str, Tuple[str, bytes, str]]]:
    """Streamlit UploadedFile -> requests multipart list."""
    parts: List[Tuple[str, Tuple[str, bytes, str]]] = []
    for f in files:
        content = f.getvalue()  # bytes
        parts.append(("files", (f.name, content, "application/octet-stream")))
    return parts


def _pretty(x: Any) -> Any:
    """Pretty JSON string for dict/list values (for display only)."""
    if isinstance(x, (dict, list)):
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)
    return x


def _ensure_bool_series(s: pd.Series) -> pd.Series:
    """Make sure 'is_undefined' is real bools even if API returned strings."""
    def to_bool(v: Any) -> bool:
        if v is None:
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "y", "да")
        return bool(v)

    return s.apply(to_bool)


def _build_dataframes(results: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(results)

    # Ensure expected columns exist
    required = ["filename", "predicted_label", "is_undefined", "max_similarity", "threshold"]
    for col in required:
        if col not in df.columns:
            df[col] = None

    # Keep a raw DF with proper bools for filtering
    df_raw = df.copy()
    df_raw["is_undefined"] = _ensure_bool_series(df_raw["is_undefined"])

    # Display DF (pretty-print complex columns)
    df_view = df_raw.copy()
    for col in df_view.columns:
        if col != "is_undefined":
            df_view[col] = df_view[col].apply(_pretty)

    preferred = [
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
        "request_id",
    ]
    cols = [c for c in preferred if c in df_view.columns] + [c for c in df_view.columns if c not in preferred]
    df_view = df_view[cols]

    return df_raw, df_view


def _render_error(resp: requests.Response) -> None:
    st.error(f"Ошибка: HTTP {resp.status_code}")
    try:
        st.json(resp.json())
    except Exception:
        st.code((resp.text or "")[:4000])


# -----------------------------
# Action: classify
# -----------------------------
if st.button("Классифицировать"):
    if not uploaded_files:
        st.error("Загрузите хотя бы один файл.")
    else:
        with st.spinner("Обрабатываем письма..."):
            multipart_files = build_multipart(uploaded_files)
            form = {
                "categories": categories,
                "threshold": f"{float(threshold):.4f}",
                "top_n": str(int(top_n)),
            }

            try:
                r = requests.post(
                    f"{API_URL}/classify/batch",
                    files=multipart_files,
                    data=form,
                    timeout=300,
                )
            except requests.RequestException as e:
                st.error(f"Ошибка запроса к API: {e}")
                r = None

        if r is None:
            pass
        elif r.status_code != 200:
            _render_error(r)
        else:
            try:
                results = r.json()
            except Exception:
                st.error("Сервер вернул не-JSON ответ.")
                st.code((r.text or "")[:4000])
            else:
                df_raw, df_view = _build_dataframes(results)

                if only_undef:
                    mask = df_raw["is_undefined"] == True  # noqa: E712
                    df_view = df_view.loc[mask]

                st.success(f"Обработано писем: {len(results)}")
                if only_undef:
                    st.info(f"Показано «Не определена»: {len(df_view)}")

                st.dataframe(df_view, use_container_width=True)

                total = len(df_raw)
                if total > 0:
                    undef_cnt = int(df_raw["is_undefined"].sum())
                    st.caption(f"Undefined: {undef_cnt}/{total} ({(undef_cnt/total*100):.1f}%)")

st.divider()

# -----------------------------
# Download CSV
# -----------------------------
st.subheader("Скачать результаты (CSV)")

col_a, col_b = st.columns([1, 1])

with col_a:
    if st.button("Получить results.csv"):
        try:
            csv_resp = requests.get(f"{API_URL}/export/results.csv", timeout=30)
        except requests.RequestException as e:
            st.error(f"Не удалось получить results.csv: {e}")
        else:
            if csv_resp.status_code == 200 and csv_resp.content:
                st.download_button(
                    label="Скачать results.csv",
                    data=csv_resp.content,
                    file_name="results.csv",
                    mime="text/csv",
                )
            else:
                st.info("results.csv пока не найден. Сначала выполните классификацию.")

# -----------------------------
# Reset results.csv  ADD THIS
# -----------------------------
with col_b:
    if st.button("🗑 Очистить results.csv"):
        try:
            resp = requests.post(f"{API_URL}/results/reset", timeout=10)
        except requests.RequestException as e:
            st.error(f"Ошибка запроса к API: {e}")
        else:
            if resp.status_code == 200:
                st.success("results.csv очищен")
                # Optional: clear UI cached state if you later add session_state tables
                st.rerun()
            else:
                _render_error(resp)
