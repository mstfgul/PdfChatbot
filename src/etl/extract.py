# src/etl/extract.py
"""
PDF dosyalarından veri çıkaran modül.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.models.document import Document
from src.utils.pdf_parser import extract_text_from_pdf

logger = logging.getLogger(__name__)


class PDFExtractor:
    """PDF dosyalarından metin çıkaran sınıf."""

    def __init__(self, upload_dir: Optional[str] = None):
        """
        PDFExtractor sınıfının başlatıcısı.

        Args:
            upload_dir: PDF dosyalarının yükleneceği dizin
        """
        self.upload_dir = upload_dir or "uploads"
        os.makedirs(self.upload_dir, exist_ok=True)

    def save_uploaded_file(self, uploaded_file) -> Path:
        """
        Yüklenen dosyayı kaydeder.

        Args:
            uploaded_file: Yüklenen dosya

        Returns:
            Path: Kaydedilen dosyanın yolu
        """
        file_path = Path(self.upload_dir) / uploaded_file.name
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        logger.info(f"Dosya kaydedildi: {file_path}")
        return file_path

    def extract_from_file(self, file_path: Union[str, Path]) -> Document:
        """
        Dosyadan metin çıkarır.

        Args:
            file_path: Dosya yolu

        Returns:
            Document: Çıkarılan metni içeren belge
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")
            
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Desteklenmeyen dosya türü: {file_path.suffix}")
            
        text = extract_text_from_pdf(file_path)
        
        return Document(
            filename=file_path.name,
            file_path=str(file_path),
            text=text,
            metadata={"source": str(file_path)},
        )

    def extract_from_directory(self, directory: Union[str, Path]) -> List[Document]:
        """
        Belirtilen dizindeki tüm PDF dosyalarından metin çıkarır.

        Args:
            directory: Dizin yolu

        Returns:
            List[Document]: Çıkarılan metinleri içeren belge listesi
        """
        directory = Path(directory)
        
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Geçerli bir dizin değil: {directory}")
            
        documents = []
        
        for file_path in directory.glob("*.pdf"):
            try:
                document = self.extract_from_file(file_path)
                documents.append(document)
            except Exception as e:
                logger.error(f"Dosya işlenirken hata oluştu {file_path}: {e}")
                
        return documents

    def batch_extract(self, file_paths: List[Union[str, Path]]) -> List[Document]:
        """
        Birden fazla dosyadan metin çıkarır.

        Args:
            file_paths: Dosya yolları listesi

        Returns:
            List[Document]: Çıkarılan metinleri içeren belge listesi
        """
        documents = []
        
        for file_path in file_paths:
            try:
                document = self.extract_from_file(file_path)
                documents.append(document)
            except Exception as e:
                logger.error(f"Dosya işlenirken hata oluştu {file_path}: {e}")
                
        return documents


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


# src/models/document.py
"""
Belge veri modelini tanımlar.
"""

from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Belge veri modeli."""
    
    filename: str
    file_path: str
    text: str
    metadata: Dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    chunks: Optional[list] = None