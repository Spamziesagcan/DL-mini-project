from __future__ import annotations

import re
from pathlib import Path

import pdfplumber


def _normalize_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def extract_text_from_pdf(path: str) -> str:
    """Extract clean plain text from a PDF file."""
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    extracted_pages: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            normalized_text = _normalize_text(page_text)
            if normalized_text:
                extracted_pages.append(normalized_text)

    return "\n\n".join(extracted_pages).strip()
