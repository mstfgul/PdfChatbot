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