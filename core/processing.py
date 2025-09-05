from __future__ import annotations
import os
import re
from typing import Final
import PyPDF2

# Reasonable guardrails for uploaded files
MAX_FILE_SIZE_BYTES: Final[int] = 10 * 1024 * 1024  # 10MB


def clean_resume_text(text: str) -> str:
    """Normalize whitespace and newlines in resume text."""
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)  # collapse >1 blank lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n ", "\n", text)
    return text.strip()


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF using PyPDF2 with basic error handling."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    file_size = os.path.getsize(pdf_path)
    if file_size > MAX_FILE_SIZE_BYTES:
        raise ValueError("File too large (>10MB). Please upload a smaller file.")

    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            texts = []
            for page in reader.pages:
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""
                texts.append(page_text)
        return clean_resume_text("\n".join(texts))
    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {e}")


def read_resume_file(path: str) -> str:
    """Read a resume from PDF or TXT and return normalized text."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return clean_resume_text(f.read())
    raise ValueError(f"Unsupported format: {ext}. Use .pdf or .txt")