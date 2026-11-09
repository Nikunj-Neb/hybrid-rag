"""Extract PDF text, using local OCR when a page has a broken text layer."""
from functools import lru_cache
import re
import threading

_ocr_lock = threading.Lock()


def readable_text(text: str) -> bool:
    visible = re.sub(r"\s", "", text)
    if len(visible) < 30:
        return False
    letters = sum(char.isalpha() for char in visible)
    return letters / len(visible) >= 0.60 and text.count("\ufffd") / len(visible) < 0.005


@lru_cache(maxsize=1)
def ocr_engine():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def extract_pdf(raw: bytes):
    import pymupdf
    import numpy as np
    pages = []
    with pymupdf.open(stream=raw, filetype="pdf") as pdf:
        if len(pdf) > 100:
            raise ValueError("Maximum 100 PDF pages")
        for number, page in enumerate(pdf, 1):
            text = page.get_text("text", sort=True)
            if not readable_text(text):
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                with _ocr_lock:
                    result, _ = ocr_engine()(pixels)
                text = "\n".join(item[1] for item in result or [] if item[2] >= 0.45)
            pages.append((number, text))
    return pages
