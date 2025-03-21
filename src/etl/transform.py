# src/etl/transform.py
"""
Çıkarılan verileri dönüştüren ve işleyen modül.
"""

import logging
import re
from typing import Dict, List, Optional, Union

from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.models.document import Document

logger = logging.getLogger(__name__)


class TextTransformer:
    """Metin dönüştürme ve işleme sınıfı."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        """
        TextTransformer sınıfının başlatıcısı.

        Args:
            chunk_size: Metin parçasının maksimum boyutu
            chunk_overlap: Metin parçaları arasındaki örtüşme miktarı
            separators: Metni bölmek için kullanılacak ayırıcılar
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
        )

    def clean_text(self, text: str) -> str:
        """
        Metni temizler.

        Args:
            text: Temizlenecek metin

        Returns:
            str: Temizlenmiş metin
        """
        # Birden fazla boşlukları tek boşluğa indir
        text = re.sub(r"\s+", " ", text)
        # Gereksiz karakterleri temizle
        text = re.sub(r"[^\w\s.,;:!?'\"()-]", "", text)
        # Metin kenarlarındaki boşlukları temizle
        text = text.strip()
        
        return text

    def split_text(self, text: str) -> List[str]:
        """
        Metni parçalara ayırır.

        Args:
            text: Parçalanacak metin

        Returns:
            List[str]: Metin parçaları listesi
        """
        if not text:
            return []
            
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Metin {len(chunks)} parçaya ayrıldı")
        
        return chunks

    def process_document(self, document: Document) -> Document:
        """
        Belgeyi işler ve dönüştürür.

        Args:
            document: İşlenecek belge

        Returns:
            Document: İşlenmiş belge
        """
        if not document.text:
            logger.warning(f"Boş belge: {document.filename}")
            return document
            
        # Metni temizle
        cleaned_text = self.clean_text(document.text)
        document.text = cleaned_text
        
        # Metni parçalara ayır
        chunks = self.split_text(cleaned_text)
        
        # Parça meta verilerini ekle
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update({
                "chunk_id": i,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "filename": document.filename,
            })
            
            processed_chunks.append({
                "text": chunk,
                "metadata": chunk_metadata,
            })
            
        document.chunks = processed_chunks
        
        return document

    def batch_process(self, documents: List[Document]) -> List[Document]:
        """
        Birden fazla belgeyi işler.

        Args:
            documents: İşlenecek belgeler listesi

        Returns:
            List[Document]: İşlenmiş belge listesi
        """
        processed_documents = []
        
        for document in documents:
            try:
                processed_document = self.process_document(document)
                processed_documents.append(processed_document)
            except Exception as e:
                logger.error(f"Belge işlenirken hata oluştu {document.filename}: {e}")
                
        return processed_documents