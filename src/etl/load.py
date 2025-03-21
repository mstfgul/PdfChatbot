# src/etl/load.py
"""
İşlenmiş verileri bir vektör veritabanına yükleyen modül.
"""

import logging
from typing import Dict, List, Optional

from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from src.models.document import Document
from src.utils.embeddings import get_embeddings_model

logger = logging.getLogger(__name__)


class VectorStoreLoader:
    """İşlenmiş belgeleri vektör veritabanına yükleyen sınıf."""

    def __init__(self, embedding_model: Optional[OpenAIEmbeddings] = None):
        """
        VectorStoreLoader sınıfının başlatıcısı.

        Args:
            embedding_model: Gömme (embedding) modeli
        """
        self.embedding_model = embedding_model or get_embeddings_model()
        self.vector_store = None

    def create_vector_store(self, texts: List[str], metadatas: List[Dict]) -> FAISS:
        """
        Vektör veritabanı oluşturur.

        Args:
            texts: Metin listesi
            metadatas: Meta veri listesi

        Returns:
            FAISS: Oluşturulan vektör veritabanı
        """
        if not texts:
            logger.warning("Boş metin listesi, vektör veritabanı oluşturulamadı")
            return None
            
        try:
            vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embedding_model,
                metadatas=metadatas
            )
            
            logger.info(f"{len(texts)} metin parçası vektör veritabanına yüklendi")
            return vector_store
            
        except Exception as e:
            logger.error(f"Vektör veritabanı oluşturulurken hata oluştu: {e}")
            raise

    def load_document(self, document: Document) -> FAISS:
        """
        Bir belgeyi vektör veritabanına yükler.

        Args:
            document: Yüklenecek belge

        Returns:
            FAISS: Güncellenen vektör veritabanı
        """
        if not document.chunks:
            logger.warning(f"Belge parçaları bulunamadı: {document.filename}")
            return self.vector_store
            
        texts = [chunk["text"] for chunk in document.chunks]
        metadatas = [chunk["metadata"] for chunk in document.chunks]
        
        # Eğer vektör veritabanı daha önce oluşturulmadıysa, yeni oluştur
        if self.vector_store is None:
            self.vector_store = self.create_vector_store(texts, metadatas)
        else:
            # Mevcut vektör veritabanına ekle
            self.vector_store.add_texts(texts=texts, metadatas=metadatas)
            logger.info(f"{len(texts)} metin parçası vektör veritabanına eklendi")
            
        return self.vector_store

    def batch_load(self, documents: List[Document]) -> FAISS:
        """
        Birden fazla belgeyi vektör veritabanına yükler.

        Args:
            documents: Yüklenecek belgeler listesi

        Returns:
            FAISS: Güncellenen vektör veritabanı
        """
        all_texts = []
        all_metadatas = []
        
        for document in documents:
            if not document.chunks:
                logger.warning(f"Belge parçaları bulunamadı: {document.filename}")
                continue
                
            all_texts.extend([chunk["text"] for chunk in document.chunks])
            all_metadatas.extend([chunk["metadata"] for chunk in document.chunks])
            
        if not all_texts:
            logger.warning("Yüklenecek metin parçası bulunamadı")
            return self.vector_store
            
        # Eğer vektör veritabanı daha önce oluşturulmadıysa, yeni oluştur
        if self.vector_store is None:
            self.vector_store = self.create_vector_store(all_texts, all_metadatas)
        else:
            # Mevcut vektör veritabanına ekle
            self.vector_store.add_texts(texts=all_texts, metadatas=all_metadatas)
            logger.info(f"{len(all_texts)} metin parçası vektör veritabanına eklendi")
            
        return self.vector_store

    def get_vector_store(self) -> FAISS:
        """
        Mevcut vektör veritabanını döndürür.

        Returns:
            FAISS: Vektör veritabanı
        """
        return self.vector_store

    def save_vector_store(self, save_path: str) -> None:
        """
        Vektör veritabanını diske kaydeder.

        Args:
            save_path: Kayıt yolu
        """
        if self.vector_store is None:
            logger.warning("Kaydedilecek vektör veritabanı bulunamadı")
            return
            
        try:
            self.vector_store.save_local(save_path)
            logger.info(f"Vektör veritabanı kaydedildi: {save_path}")
        except Exception as e:
            logger.error(f"Vektör veritabanı kaydedilirken hata oluştu: {e}")
            raise

    def load_vector_store(self, load_path: str) -> FAISS:
        """
        Vektör veritabanını diskten yükler.

        Args:
            load_path: Yükleme yolu

        Returns:
            FAISS: Yüklenen vektör veritabanı
        """
        try:
            self.vector_store = FAISS.load_local(
                load_path, self.embedding_model
            )
            logger.info(f"Vektör veritabanı yüklendi: {load_path}")
            return self.vector_store
        except Exception as e:
            logger.error(f"Vektör veritabanı yüklenirken hata oluştu: {e}")
            raise


# src/utils/embeddings.py
"""
Gömme (embedding) modelleri için yardımcı fonksiyonlar.
"""

import logging
import os
from typing import Optional

from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


def get_embeddings_model(model_name: Optional[str] = None) -> OpenAIEmbeddings:
    """
    OpenAI gömme modelini döndürür.

    Args:
        model_name: Kullanılacak model adı

    Returns:
        OpenAIEmbeddings: Gömme modeli
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY bulunamadı. Lütfen .env dosyasında tanımlayın."
        )
        
    model_name = model_name or "text-embedding-3-small"
    
    try:
        embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=api_key,
        )
        
        logger.info(f"Gömme modeli oluşturuldu: {model_name}")
        return embeddings
        
    except Exception as e:
        logger.error(f"Gömme modeli oluşturulurken hata oluştu: {e}")
        raise