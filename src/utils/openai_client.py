# src/utils/openai_client.py
"""
OpenAI API ile iletişim kurmak için yardımcı fonksiyonlar.
"""

import logging
import os
from typing import Dict, List, Optional

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)


def get_openai_client() -> OpenAI:
    """
    OpenAI istemcisini döndürür.

    Returns:
        OpenAI: OpenAI istemcisi
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY bulunamadı. Lütfen .env dosyasında tanımlayın."
        )
        
    client = OpenAI(api_key=api_key)
    logger.info("OpenAI istemcisi oluşturuldu")
    
    return client


def get_openai_chat() -> ChatOpenAI:
    """
    OpenAI sohbet modelini döndürür.

    Returns:
        ChatOpenAI: OpenAI sohbet modeli
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY bulunamadı. Lütfen .env dosyasında tanımlayın."
        )
        
    chat = ChatOpenAI(
        model_name="gpt-4-turbo-preview",
        temperature=0,
        openai_api_key=api_key,
    )
    
    logger.info("OpenAI sohbet modeli oluşturuldu")
    return chat


class QASystem:
    """Soru-cevap sistemi."""

    def __init__(self, vector_store, model_name: str = "gpt-4-turbo-preview"):
        """
        QASystem sınıfının başlatıcısı.

        Args:
            vector_store: Vektör veritabanı
            model_name: Kullanılacak model adı
        """
        self.vector_store = vector_store
        self.model_name = model_name
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY bulunamadı. Lütfen .env dosyasında tanımlayın."
            )
            
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=0,
            openai_api_key=api_key,
        )
        
        # Prompt şablonu oluştur
        template = """
        Yüklenen PDF makalelerden aşağıdaki soruyu yanıtla. 
        Yalnızca verilen belgelerde bulunan bilgileri kullan. 
        Eğer yanıt belgelerde bulunamıyorsa, "Bu sorunun yanıtını verilerin içinde bulamadım." de.
        
        Belgeler:
        {context}
        
        Soru: {question}
        
        Yanıt:
        """
        
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # QA zinciri oluştur
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        self.qa_chain = self._create_qa_chain()
        logger.info("QA sistemi oluşturuldu")

    def _create_qa_chain(self) -> RetrievalQA:
        """
        QA zinciri oluşturur.

        Returns:
            RetrievalQA: QA zinciri
        """
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt}
        )

    def answer_question(self, question: str) -> Dict:
        """
        Soruyu yanıtlar.

        Args:
            question: Soru

        Returns:
            Dict: Yanıt ve kaynak belgeler
        """
        if not question:
            return {
                "answer": "Bir soru sormanız gerekiyor.",
                "sources": []
            }
            
        try:
            result = self.qa_chain({"query": question})
            
            sources = []
            for doc in result.get("source_documents", []):
                metadata = doc.metadata
                sources.append({
                    "filename": metadata.get("filename", "Bilinmeyen"),
                    "chunk_id": metadata.get("chunk_id", -1),
                    "text": doc.page_content[:150] + "..."  # İlk 150 karakteri göster
                })
                
            return {
                "answer": result["result"],
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Soru yanıtlanırken hata oluştu: {e}")
            return {
                "answer": "Sorunuzu yanıtlarken bir hata oluştu. Lütfen tekrar deneyin.",
                "sources": []
            }