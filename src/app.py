# src/app.py
"""
PDF makale soru-cevap Streamlit uygulaması.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import streamlit as st
from dotenv import load_dotenv

from src import config
from etl.extract import PDFExtractor
from etl.transform import TextTransformer
from etl.load import VectorStoreLoader
from utils.openai_client import QASystem

# .env dosyasını yükle
load_dotenv()

# Loglama
logger = logging.getLogger(__name__)


def init_session_state():
    """Streamlit oturum durumunu başlatır."""
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
        
    if "documents" not in st.session_state:
        st.session_state.documents = []
        
    if "processed_documents" not in st.session_state:
        st.session_state.processed_documents = []
        
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
        
    if "qa_system" not in st.session_state:
        st.session_state.qa_system = None
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def display_header():
    """Sayfa başlığını ve açıklamayı gösterir."""
    st.title("PDF Makale Soru-Cevap Sistemi")
    st.markdown(
        """
        Bu uygulama, PDF makalelerine sorular sormanıza ve onlardan cevaplar almanıza olanak tanır.
        Başlamak için PDF dosyalarınızı yükleyin.
        """
    )


def check_api_key() -> bool:
    """
    OpenAI API anahtarını kontrol eder.

    Returns:
        bool: API anahtarı geçerli mi
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        st.error(
            "OpenAI API anahtarı bulunamadı. Lütfen yan menüden API anahtarınızı girin."
        )
        return False
        
    return True


def upload_documents():
    """Belge yükleme bölümünü gösterir."""
    with st.expander("PDF Makaleleri Yükle", expanded=True):
        uploaded_files = st.file_uploader(
            "PDF dosyalarınızı yükleyin",
            type="pdf",
            accept_multiple_files=True,
        )
        
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            
            if st.button("Dosyaları İşle"):
                with st.spinner("Dosyalar işleniyor..."):
                    process_documents(uploaded_files)


def process_documents(uploaded_files):
    """
    Yüklenen belgeleri işler.

    Args:
        uploaded_files: Yüklenen dosyalar
    """
    try:
        # ETL pipeline bileşenlerini oluştur
        extractor = PDFExtractor(upload_dir=str(config.upload_dir))
        transformer = TextTransformer(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        loader = VectorStoreLoader()
        
        # PDF dosyalarını yükle ve metin çıkar (Extract)
        documents = []
        for uploaded_file in uploaded_files:
            file_path = extractor.save_uploaded_file(uploaded_file)
            document = extractor.extract_from_file(file_path)
            documents.append(document)
            
        st.session_state.documents = documents
        
        # Metni işle ve dönüştür (Transform)
        processed_documents = transformer.batch_process(documents)
        st.session_state.processed_documents = processed_documents
        
        # Vektör veritabanına yükle (Load)
        vector_store = loader.batch_load(processed_documents)
        st.session_state.vector_store = vector_store
        
        # QA sistemini oluştur
        qa_system = QASystem(vector_store, model_name=config.llm_model)
        st.session_state.qa_system = qa_system
        
        # Başarı mesajı
        doc_count = len(documents)
        chunk_count = sum(len(doc.chunks) if doc.chunks else 0 for doc in processed_documents)
        st.success(
            f"{doc_count} dosya başarıyla işlendi ve {chunk_count} metin parçası vektör veritabanına yüklendi."
        )
        
    except Exception as e:
        st.error(f"Dosyalar işlenirken bir hata oluştu: {str(e)}")
        logger.error(f"Dosyalar işlenirken hata: {e}", exc_info=True)


def display_documents():
    """İşlenen belgeleri gösterir."""
    if st.session_state.documents:
        with st.expander("İşlenen PDF Makaleler", expanded=False):
            for i, doc in enumerate(st.session_state.documents):
                st.markdown(f"**{i+1}. {doc.filename}**")
                
                if st.checkbox(f"Makale içeriğini göster ({doc.filename})", key=f"show_doc_{i}"):
                    st.text_area(
                        f"Makale içeriği ({doc.filename})",
                        value=doc.text[:1000] + "..." if len(doc.text) > 1000 else doc.text,
                        height=200,
                        key=f"doc_content_{i}",
                    )


def display_qa_interface():
    """Soru-cevap arayüzünü gösterir."""
    if st.session_state.qa_system:
        st.header("Makalelere Soru Sor")
        
        # Kullanıcı sorusu
        user_question = st.text_input("Sorunuzu yazın:")
        
        if st.button("Soru Sor"):
            if not user_question:
                st.warning("Lütfen bir soru girin.")
            else:
                with st.spinner("Yanıt aranıyor..."):
                    # Soruyu yanıtla
                    response = st.session_state.qa_system.answer_question(user_question)
                    
                    # Sohbet geçmişine ekle
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": response["answer"],
                        "sources": response["sources"],
                        "timestamp": time.strftime("%H:%M:%S"),
                    })
                    
        # Sohbet geçmişini göster
        if st.session_state.chat_history:
            st.subheader("Sohbet Geçmişi")
            
            for i, item in enumerate(st.session_state.chat_history):
                # Soru
                message_container = st.container()
                with message_container:
                    col1, col2 = st.columns([0.85, 0.15])
                    with col1:
                        st.markdown(f"**Soru ({item['timestamp']}):**")
                        st.markdown(f"{item['question']}")
                        
                    with col2:
                        if st.button("✕", key=f"delete_qa_{i}"):
                            st.session_state.chat_history.pop(i)
                            st.rerun()
                            
                # Cevap
                st.markdown("**Yanıt:**")
                st.markdown(item["answer"])
                
                # Kaynaklar
                if item["sources"]:
                    with st.expander("Kaynaklar", expanded=False):
                        for j, source in enumerate(item["sources"]):
                            st.markdown(
                                f"**{source['filename']}** (Parça {source['chunk_id']})"
                            )
                            st.markdown(f"*Alıntı:* {source['text']}")
                            if j < len(item["sources"]) - 1:
                                st.markdown("---")
                                
                st.markdown("---")


def display_sidebar():
    """Kenar çubuğunu gösterir."""
    st.sidebar.title("Ayarlar")
    
    # API anahtarı girişi
    api_key = st.sidebar.text_input(
        "OpenAI API Anahtarı",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
    )
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        
    # Chunk ayarları
    st.sidebar.subheader("Metin İşleme Ayarları")
    chunk_size = st.sidebar.slider(
        "Parça Boyutu",
        min_value=100,
        max_value=4000,
        value=config.chunk_size,
        step=100,
    )
    
    chunk_overlap = st.sidebar.slider(
        "Parça Örtüşmesi",
        min_value=0,
        max_value=500,
        value=config.chunk_overlap,
        step=50,
    )
    
    # Model ayarları
    st.sidebar.subheader("Model Ayarları")
    embedding_model = st.sidebar.selectbox(
        "Gömme (Embedding) Modeli",
        options=["text-embedding-3-small", "text-embedding-3-large"],
        index=0,
    )
    
    llm_model = st.sidebar.selectbox(
        "Dil Modeli",
        options=["gpt-4-turbo-preview", "gpt-3.5-turbo"],
        index=0,
    )
    
    # Ayarları kaydet
    if st.sidebar.button("Ayarları Kaydet"):
        config.chunk_size = chunk_size
        config.chunk_overlap = chunk_overlap
        config.embedding_model = embedding_model
        config.llm_model = llm_model
        
        os.environ["CHUNK_SIZE"] = str(chunk_size)
        os.environ["CHUNK_OVERLAP"] = str(chunk_overlap)
        os.environ["EMBEDDING_MODEL"] = embedding_model
        os.environ["LLM_MODEL"] = llm_model
        
        st.sidebar.success("Ayarlar kaydedildi!")
        
    # Hakkında bilgisi
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Hakkında")
    st.sidebar.markdown(
        "Bu uygulama, PDF makalelerini analiz edip sorularınızı "
        "yanıtlamak için OpenAI API'sini kullanır."
    )


def main():
    """Ana uygulama fonksiyonu."""
    # Oturum durumunu başlat
    init_session_state()
    
    # Kenar çubuğunu göster
    display_sidebar()
    
    # Başlık ve açıklamayı göster
    display_header()
    
    # API anahtarını kontrol et
    if not check_api_key():
        return
        
    # Dosya yükleme bölümünü göster
    upload_documents()
    
    # İşlenen belgeleri göster
    display_documents()
    
    # Soru-cevap arayüzünü göster
    display_qa_interface()


if __name__ == "__main__":
    main()