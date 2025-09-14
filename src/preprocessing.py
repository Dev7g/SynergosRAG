# src/preprocessing.py

import fitz  # PyMuPDF
import os
from typing import List, Tuple
from paddleocr import PaddleOCR
from PIL import Image

# -------------------------------------------------------------------------
# PDF PROCESSING
# -------------------------------------------------------------------------

def process_pdf(pdf_path: str, output_folder: str) -> Tuple[List[str], List[str]]:
    """
    Processes a PDF file to extract text from each page and save each page as an image.
    """
    # Create output directories if they don't exist
    text_output_dir = os.path.join(output_folder, "text")
    image_output_dir = os.path.join(output_folder, "images")
    os.makedirs(text_output_dir, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)

    text_file_paths: List[str] = []
    image_file_paths: List[str] = []

    doc_name = os.path.splitext(os.path.basename(pdf_path))[0]

    print(f"Processing document: {doc_name} with {len(doc)} pages.")

    for page_num, page in enumerate(doc):
        # 1. Extract and save text
        text = page.get_text()
        text_file_path = os.path.join(text_output_dir, f"{doc_name}_page_{page_num + 1}.txt")
        with open(text_file_path, "w", encoding="utf-8") as text_file:
            text_file.write(text)
        text_file_paths.append(text_file_path)

        # 2. Render and save page as an image (high DPI for OCR)
        pix = page.get_pixmap(dpi=300)
        image_file_path = os.path.join(image_output_dir, f"{doc_name}_page_{page_num + 1}.png")
        pix.save(image_file_path)
        image_file_paths.append(image_file_path)

    doc.close()
    print(f"Finished processing. Extracted {len(text_file_paths)} text files and {len(image_file_paths)} images.")
    return text_file_paths, image_file_paths


# -------------------------------------------------------------------------
# OCR USING PADDLEOCR
# -------------------------------------------------------------------------

# Initialize PaddleOCR once
ocr_engine = PaddleOCR(use_angle_cls=True, lang="en")

def safe_ocr(image_path: str) -> str:
    """
    Runs OCR on an image with resizing safeguard.
    Returns extracted text or empty string if failed.
    """
    try:
        # Downscale very large images before OCR (avoid PaddleOCR crash)
        with Image.open(image_path) as im:
            if im.width > 2000 or im.height > 2000:
                im.thumbnail((2000, 2000))
                im.save(image_path)

        result = ocr_engine.predict(image_path)  # FIX: use predict()
        if not result or result[0] is None:
            return ""

        # PaddleOCR returns [[[box], (text, confidence)], ...]
        texts = [line[1][0] for line in result[0]]
        return "\n".join(texts)

    except Exception as e:
        print(f"OCR failed on {image_path}: {e}")
        return ""


def perform_ocr_on_images(image_paths: List[str]) -> List[str]:
    """
    Performs OCR on a list of image files using PaddleOCR.
    """
    all_ocr_texts = []
    print(f"Performing OCR on {len(image_paths)} images...")

    for image_path in image_paths:
        ocr_text = safe_ocr(image_path)
        all_ocr_texts.append(ocr_text)

    print("OCR processing complete.")
    return all_ocr_texts


# -------------------------------------------------------------------------
# COMBINE TEXT + OCR
# -------------------------------------------------------------------------

def combine_text_and_ocr(text_file_paths: List[str], ocr_texts: List[str], output_folder: str):
    """
    Combines text extracted from PDF parsing with text from OCR.
    Appends OCR text to the parsed text for each page.
    """
    combined_texts_dir = os.path.join(output_folder, "combined_text")
    os.makedirs(combined_texts_dir, exist_ok=True)

    for i, text_file_path in enumerate(text_file_paths):
        with open(text_file_path, "r", encoding="utf-8") as f:
            parsed_text = f.read()

        ocr_text = ocr_texts[i] if i < len(ocr_texts) else ""
        combined_text = parsed_text + "\n\n--- OCR TEXT ---\n\n" + ocr_text

        base_name = os.path.basename(text_file_path)
        combined_file_path = os.path.join(combined_texts_dir, base_name)

        with open(combined_file_path, "w", encoding="utf-8") as f:
            f.write(combined_text)

    print(f"Combined text files saved in {combined_texts_dir}")
