from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    # Model
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small")
    DEVICE: str = os.getenv("DEVICE", "cuda" if os.getenv("USE_CUDA", "0") == "1" else "cpu")

    # Inference
    TOP_N_DEFAULT: int = int(os.getenv("TOP_N_DEFAULT", "5"))
    THRESHOLD_UNDEFINED: float = float(os.getenv("THRESHOLD_UNDEFINED", "0.30"))
    SOFTMAX_TEMPERATURE: float = float(os.getenv("SOFTMAX_TEMPERATURE", "0.07"))  # lower -> sharper

    # Limits
    MAX_EMAIL_CHARS: int = int(os.getenv("MAX_EMAIL_CHARS", "20000"))
    MAX_ATTACHMENT_BYTES: int = int(os.getenv("MAX_ATTACHMENT_BYTES", str(10 * 1024 * 1024)))
    MAX_PDF_PAGES: int = int(os.getenv("MAX_PDF_PAGES", "5"))
    MAX_ATTACHMENTS: int = int(os.getenv("MAX_ATTACHMENTS", "10"))

    # Paths
    RESULTS_DIR: str = os.getenv("RESULTS_DIR", "results")
    LOGS_DIR: str = os.getenv("LOGS_DIR", "logs")

settings = Settings()
