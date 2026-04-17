"""PDF document loader with smart routing for broken Thai text.

Reads PDFs via PyMuPDF (fitz) and heuristically checks if Thai text is broken.
If broken, converts the page to an image and uses the OpenTyphoon Vision API
(typhoon-ocr) to extract and fix the text.
"""

import base64
import logging
import os
import re
import tempfile
from io import BytesIO
from typing import List

import fitz  # PyMuPDF
import pythainlp.util
import requests
from langchain_core.documents import Document
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)

MAX_OCR_PAGES_PER_FILE = 10
VISION_MODEL = "typhoon-ocr"
VISION_API_URL = "https://api.opentyphoon.ai/v1/chat/completions"


def is_broken_thai(text: str) -> bool:
    """Heuristically check if text is broken Thai.
    Checks for zero-width spaces or detached vowels.
    """
    if '\u200b' in text or '\u200c' in text:
        return True
    
    if re.search(r'[ก-๙]\s+[ะ-ฺเ-๎]', text):
        return True
        
    return False


def clean_text(text: str) -> str:
    """Fix common spacing artefacts produced by PDF text extraction.
    
    1. Remove zero-width spaces.
    2. Fix detached vowels formatting.
    3. Remove unnatural spaces between Thai consonants.
    4. Normalize using PyThaiNLP.
    """
    text = text.replace('\u200b', '').replace('\u200c', '')
    
    # Fix detached vowels
    text = re.sub(r'\s+([ะ-ฺเ-๎])', r'\1', text)
    
    # Fix spaces between Thai consonants (run twice for overlaps)
    text = re.sub(r'([ก-๙])\s{1,2}([ก-๙])', r'\1\2', text)
    text = re.sub(r'([ก-๙])\s{1,2}([ก-๙])', r'\1\2', text)
    
    text = pythainlp.util.normalize(text)
    
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _call_typhoon_vision_api(image_base64: str, prompt: str) -> str:
    """Send base64 image to OpenTyphoon Vision API."""
    api_key = os.environ.get("TYPHOON_API_KEY")
    if not api_key:
        logger.error("TYPHOON_API_KEY not set. Cannot use Vision API.")
        return ""
        
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 2000,
    }
    
    try:
        response = requests.post(VISION_API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            logger.error("Vision API Error %d: %s", response.status_code, response.text)
    except Exception as exc:
        logger.error("Vision API request failed: %s", exc)
    return ""


def process_uploaded_pdf(file_bytes: bytes) -> List[Document]:
    """Parse PDF bytes by writing to a temporary file, then loading."""
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name
        return process_uploaded_pdf_path(temp_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def process_uploaded_pdf_path(file_path: str) -> List[Document]:
    """Parse a PDF from a path with Smart Routing for broken Thai text."""
    logger.info("Loading PDF via Smart Routing: %s", file_path)
    documents = []
    
    doc = fitz.open(file_path)
    ocr_pages_used = 0
    
    for i, page in enumerate(doc):
        raw_text = page.get_text()
        
        if is_broken_thai(raw_text):
            logger.warning("Broken Thai detected on page %d of %s", i+1, os.path.basename(file_path))
            
            if ocr_pages_used < MAX_OCR_PAGES_PER_FILE:
                logger.info("Routing page %d to Typhoon OCR...", i+1)
                
                # convert_from_path uses 1-indexed pages
                images = convert_from_path(file_path, first_page=i+1, last_page=i+1, dpi=200)
                if images:
                    buffered = BytesIO()
                    images[0].save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    
                    prompt = (
                        "Extract all Thai text from this image. "
                        "Fix all broken characters, tone marks, and spacing. "
                        "Output ONLY the clean text."
                    )
                    
                    extracted_text = _call_typhoon_vision_api(img_str, prompt)
                    
                    if extracted_text:
                        raw_text = extracted_text
                    ocr_pages_used += 1
            else:
                logger.warning("OCR limit reached (%d). Skipping Vision API for page %d.", MAX_OCR_PAGES_PER_FILE, i+1)
        
        cleaned_text = clean_text(raw_text)
        
        if cleaned_text:
            documents.append(Document(
                page_content=cleaned_text,
                metadata={"source": file_path, "page": i}
            ))
            
    doc.close()
    
    logger.info(
        "Extracted %d pages (OCR used on %d) from %s", 
        len(documents), ocr_pages_used, os.path.basename(file_path)
    )
    return documents