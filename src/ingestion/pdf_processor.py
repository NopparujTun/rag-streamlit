import asyncio
import io
import logging
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional

import fitz  # PyMuPDF
import pythainlp.util

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
* Preserve layout using markdown headers
* Extract ALL text (no summarization)
* Convert tables to markdown tables
* Convert formulas to LaTeX
* Describe images in detail using:
  ![description](image)
* Maintain reading order (multi-column aware)
* No hallucinations
* Output raw markdown only
"""


def clean_markdown_thai(text: str) -> str:
    """Fix common spacing artifacts produced by PDF text extraction without destroying markdown."""
    text = text.replace('\u200b', '').replace('\u200c', '')
    
    # Fix detached following vowels (ะ-ฺ) but NOT leading vowels (เ-๎) unless they are clearly detached inside a word
    # A safer approach is to only fix detached upper/lower marks or just let pythainlp normalize do its job.
    # We will just rely on pythainlp normalize for character-level fixes to avoid destroying word boundaries.
    
    text = pythainlp.util.normalize(text)
    
    return text


def analyze_pdf(pdf_path: str) -> Dict[str, Any]:
    """Analyze a PDF to determine its characteristics."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Failed to open PDF for analysis: {pdf_path}. Error: {e}")
        raise

    num_pages = len(doc)
    if num_pages == 0:
        return {"is_scanned": False, "has_images": False, "layout_complexity": "simple"}

    total_text_len = 0
    total_images = 0
    max_blocks_per_page = 0

    for page in doc:
        text = page.get_text("text")
        total_text_len += len(text.strip())
        
        image_list = page.get_images()
        total_images += len(image_list)
        
        blocks = page.get_text("blocks")
        if len(blocks) > max_blocks_per_page:
            max_blocks_per_page = len(blocks)

    doc.close()

    avg_text_per_page = total_text_len / num_pages
    avg_images_per_page = total_images / num_pages

    # Heuristics for detection
    is_scanned = avg_text_per_page < 50 and total_images > 0
    has_images = avg_images_per_page >= 0.5
    
    if max_blocks_per_page > 50:
        layout_complexity = "complex"
    elif max_blocks_per_page > 20:
        layout_complexity = "medium"
    else:
        layout_complexity = "simple"

    return {
        "is_scanned": is_scanned,
        "has_images": has_images,
        "layout_complexity": layout_complexity
    }


def _process_simple_pipeline(pdf_path: str) -> str:
    """Fast text extraction using PyMuPDF4LLM."""
    try:
        import pymupdf4llm
    except ImportError:
        logger.error("pymupdf4llm not installed. Please install it to use the simple pipeline.")
        raise
        
    return pymupdf4llm.to_markdown(pdf_path)


def _process_medium_pipeline(pdf_path: str) -> str:
    """OCR pipeline using Docling."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        logger.error("docling not installed. Please install it to use the medium OCR pipeline.")
        raise
        
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    return result.document.export_to_markdown()


async def _process_page_gemini(page_image_bytes: bytes, page_num: int, api_key: str) -> str:
    """Process a single page image with Gemini."""
    try:
        import google.generativeai as genai
        from PIL import Image
    except ImportError:
        logger.error("google-generativeai or Pillow not installed.")
        raise
        
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel(
        model_name='gemini-1.5-pro',
        system_instruction=SYSTEM_PROMPT
    )
    
    image = Image.open(io.BytesIO(page_image_bytes))
    
    response = await model.generate_content_async([image])
    return f"# Page {page_num}\n{response.text.strip()}\n"


async def _process_complex_pipeline_async(pdf_path: str, api_key: str) -> str:
    """VLM pipeline asynchronously converting pages to images and calling Gemini."""
    doc = fitz.open(pdf_path)
    tasks = []
    
    # Render at 300 DPI
    zoom = 300 / 72
    mat = fitz.Matrix(zoom, zoom)
    
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        tasks.append(_process_page_gemini(img_bytes, i + 1, api_key))
        
    results = await asyncio.gather(*tasks, return_exceptions=True)
    doc.close()
    
    markdown_output = ""
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            logger.error(f"Error processing page {i+1} with Gemini: {res}")
            markdown_output += f"# Page {i+1}\n[Error processing page: {res}]\n---\n"
        else:
            markdown_output += f"{res}\n---\n"
            
    return markdown_output.strip()


def _process_complex_pipeline(pdf_path: str, api_key: str) -> str:
    """Synchronous wrapper for the complex VLM pipeline."""
    return asyncio.run(_process_complex_pipeline_async(pdf_path, api_key))


def validate_markdown(md_text: str) -> Dict[str, Any]:
    """Validate output markdown quality."""
    word_count = len(md_text.split())
    header_presence = "#" in md_text
    
    # Heuristic for table presence in markdown
    table_presence = "|" in md_text and "-|-" in md_text.replace(" ", "")
    
    # Rough check for LaTeX formulas
    formula_presence = "$" in md_text or "\\(" in md_text or "\\[" in md_text
    
    lines = md_text.split('\n')
    avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0
    
    return {
        "word_count": word_count,
        "header_presence": header_presence,
        "table_presence": table_presence,
        "formula_presence": formula_presence,
        "avg_line_length": round(avg_line_length, 2)
    }


def process_pdf(pdf_path: str, output_dir: str, api_key: Optional[str] = None) -> str:
    """
    Process a PDF and return path to generated Markdown file.
    Routes to the appropriate pipeline based on document analysis.
    """
    path_obj = Path(pdf_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
    pdf_name = path_obj.stem
    out_dir_obj = Path(output_dir)
    out_dir_obj.mkdir(parents=True, exist_ok=True)
    
    out_file = out_dir_obj / f"{pdf_name}.md"
    
    # Caching check
    if out_file.exists():
        logger.info(f"Skipping {pdf_path}, already processed. Output exists at {out_file}.")
        return str(out_file)
        
    logger.info(f"Analyzing PDF: {pdf_path}")
    analysis = analyze_pdf(pdf_path)
    logger.info(f"Analysis results for {pdf_name}: {analysis}")
    
    pipeline_selected = ""
    md_text = ""
    
    try:
        # Decision Routing Logic
        if analysis.get("is_scanned"):
            pipeline_selected = "OCR pipeline (Docling)"
            logger.info(f"Selected pipeline: {pipeline_selected}")
            md_text = _process_medium_pipeline(pdf_path)
            
        elif analysis.get("has_images"):
            pipeline_selected = "VLM pipeline (Gemini)"
            logger.info(f"Selected pipeline: {pipeline_selected}")
            if not api_key:
                api_key = os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("API key required for VLM pipeline (Gemini).")
            md_text = _process_complex_pipeline(pdf_path, api_key)
            
        else:
            pipeline_selected = "Fast text extraction (PyMuPDF4LLM)"
            logger.info(f"Selected pipeline: {pipeline_selected}")
            md_text = _process_simple_pipeline(pdf_path)
            
    except Exception as e:
        logger.error(f"Pipeline '{pipeline_selected}' failed for {pdf_path}: {e}")
        raise
        
    # Clean the extracted Thai text before saving
    md_text = clean_markdown_thai(md_text)
        
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(md_text)
        
    logger.info(f"Output successfully saved to {out_file}")
    
    # Quality validation
    validation = validate_markdown(md_text)
    logger.info(f"Validation metrics for {pdf_name}: {validation}")
    
    return str(out_file)


def process_folder(input_dir: str, output_dir: str, api_key: Optional[str] = None) -> Dict[str, str]:
    """Batch process a folder of PDFs."""
    input_path = Path(input_dir)
    results = {}
    
    for pdf_file in input_path.glob("*.pdf"):
        try:
            out_path = process_pdf(str(pdf_file), output_dir, api_key)
            results[str(pdf_file)] = out_path
        except Exception as e:
            logger.error(f"Failed to process {pdf_file}: {e}")
            results[str(pdf_file)] = f"Error: {e}"
            
    return results
