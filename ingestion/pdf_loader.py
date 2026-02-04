import re
import io
import pdfplumber
from PyPDF2 import PdfReader


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"-\s+", "", text)
    return text.strip()


def extract_text_pdfplumber_from_bytes(pdf_bytes: bytes) -> str:
    parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
    return "\n".join(parts)


def extract_text_pypdf2_from_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            parts.append(t)
    return "\n".join(parts)


def extract_text_ocr_from_bytes(pdf_bytes: bytes) -> str:
    """
    OCR fallback for scanned PDFs.
    Requires:
      pip install pymupdf pillow pytesseract
      brew install tesseract
    """
    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image

    parts = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in doc:
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        t = pytesseract.image_to_string(img)
        if t:
            parts.append(t)
    return "\n".join(parts)


def load_contract_pdf_bytes(pdf_bytes: bytes, use_ocr_fallback: bool = True) -> str:
    """
    Main entrypoint for Streamlit uploads (bytes):
      1) pdfplumber
      2) PyPDF2
      3) OCR (optional)
    """
    text = ""

    # 1) pdfplumber
    try:
        text = extract_text_pdfplumber_from_bytes(pdf_bytes)
    except Exception as e:
        print("[pdfplumber] failed:", e)

    # 2) PyPDF2 fallback
    if not text or len(text.strip()) < 200:
        try:
            text = extract_text_pypdf2_from_bytes(pdf_bytes)
        except Exception as e:
            print("[PyPDF2] failed:", e)

    # 3) OCR fallback
    if use_ocr_fallback and (not text or len(text.strip()) < 200):
        try:
            print("[OCR] Using OCR fallback (scanned PDF suspected)...")
            text = extract_text_ocr_from_bytes(pdf_bytes)
        except Exception as e:
            print("[OCR] failed:", e)

    return clean_text(text)
