# src/utils/pdf_parser.py
"""
PDF dosyalarından metin çıkarmak için yardımcı fonksiyonlar.
"""

import logging
from pathlib import Path
from typing import Union

import PyPDF2

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: Union[str, Path]) -> str:
    """
    PDF dosyasından metin çıkarır.

    Args:
        file_path: PDF dosyasının yolu

    Returns:
        str: Çıkarılan metin
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")
        
    extracted_text = ""
    
    try:
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                
                if page_text:
                    extracted_text += page_text + "\n\n"
                    
        logger.info(f"Metin çıkarıldı: {file_path}")
        
        if not extracted_text.strip():
            logger.warning(f"Çıkarılan metin boş: {file_path}")
            
        return extracted_text.strip()
        
    except Exception as e:
        logger.error(f"PDF işlenirken hata oluştu {file_path}: {e}")
        raise