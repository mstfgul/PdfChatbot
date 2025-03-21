# src/config.py
"""
Uygulama konfigürasyonu.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()


# Loglama konfigürasyonu
def setup_logging(log_level: str = "INFO") -> None:
    """
    Loglama sistemini kurar.

    Args:
        log_level: Log seviyesi
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Geçersiz log seviyesi: {log_level}")
        
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# Uygulama konfigürasyonu
class AppConfig:
    """Uygulama konfigürasyon sınıfı."""

    def __init__(self):
        """AppConfig sınıfının başlatıcısı."""
        # Uygulama dizinleri
        self.base_dir = Path(__file__).parent.parent
        self.upload_dir = self.base_dir / "uploads"
        self.vector_store_dir = self.base_dir / "vector_store"
        
        # Dizinleri oluştur
        self.upload_dir.mkdir(exist_ok=True)
        self.vector_store_dir.mkdir(exist_ok=True)
        
        # OpenAI API anahtarı
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logging.warning(
                "OPENAI_API_KEY bulunamadı. Lütfen .env dosyasında tanımlayın."
            )
            
        # Varsayılan metin işleme parametreleri
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        
        # Varsayılan model parametreleri
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")

    def to_dict(self) -> Dict:
        """
        Konfigürasyonu sözlük olarak döndürür.

        Returns:
            Dict: Konfigürasyon sözlüğü
        """
        return {
            "upload_dir": str(self.upload_dir),
            "vector_store_dir": str(self.vector_store_dir),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
        }


# Global konfigürasyon nesnesi
config = AppConfig()

# Loglama sistemini kur
setup_logging(os.getenv("LOG_LEVEL", "INFO"))