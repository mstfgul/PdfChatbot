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