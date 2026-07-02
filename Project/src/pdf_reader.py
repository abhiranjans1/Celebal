"""
PDF resume text extraction.

Tries `pypdf` first (fast, pure-Python, handles most text-based PDFs).
Falls back to `pdfplumber` (slower, better layout handling) if pypdf
returns little/no text -- e.g. for PDFs with unusual encoding or complex
multi-column layouts. If both extract almost nothing, the PDF is likely
a scanned image without a text layer, which needs OCR.
"""
from pathlib import Path

MIN_USABLE_CHARS = 50 


def _extract_with_pypdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts).strip()


def _extract_with_pdfplumber(path: str) -> str:
    import pdfplumber
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts).strip()


def extract_resume_text(path: str) -> dict:
    """Returns {'text': str, 'method': str, 'warning': str|None}."""
    path = str(path)
    if not Path(path).exists():
        return {"text": "", "method": None, "warning": f"File not found: {path}"}

    # 1. try pypdf
    try:
        text = _extract_with_pypdf(path)
    except Exception as e:
        text = ""
        print(f"[pdf_reader] pypdf failed: {e}")

    if len(text) >= MIN_USABLE_CHARS:
        return {"text": text, "method": "pypdf", "warning": None}

    # 2. fall back to pdfplumber
    try:
        text2 = _extract_with_pdfplumber(path)
    except Exception as e:
        text2 = ""
        print(f"[pdf_reader] pdfplumber failed: {e}")

    if len(text2) >= MIN_USABLE_CHARS:
        return {"text": text2, "method": "pdfplumber", "warning": None}

    # 3. both failed / near-empty -> likely a scanned image PDF (no text layer)
    best = text2 if len(text2) > len(text) else text
    return {
        "text": best,
        "method": "pypdf+pdfplumber (low yield)",
        "warning": (
            "Very little text could be extracted from this PDF. It may be a "
            "scanned image without a text layer (OCR would be needed), or a "
            "non-standard format. Consider pasting the resume text directly instead."
        ),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.pdf_reader <path_to_resume.pdf>")
    else:
        result = extract_resume_text(sys.argv[1])
        print(f"Extraction method: {result['method']}")
        if result["warning"]:
            print(f"WARNING: {result['warning']}")
        print(f"Extracted {len(result['text'])} characters")
        print(result["text"][:500])
