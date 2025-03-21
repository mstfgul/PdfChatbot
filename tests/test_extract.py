# tests/test_extract.py
"""
PDF çıkarma modülü için test dosyası.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.etl.extract import PDFExtractor
from src.models.document import Document


class TestPDFExtractor:
    """PDFExtractor sınıfı için test sınıfı."""

    def setup_method(self):
        """Her test öncesi çalışır."""
        self.test_dir = "test_uploads"
        os.makedirs(self.test_dir, exist_ok=True)
        self.extractor = PDFExtractor(upload_dir=self.test_dir)

    def teardown_method(self):
        """Her test sonrası çalışır."""
        # Test dizinini temizle
        for file in Path(self.test_dir).glob("*"):
            file.unlink()
        os.rmdir(self.test_dir)

    @patch("src.utils.pdf_parser.extract_text_from_pdf")
    def test_extract_from_file(self, mock_extract_text):
        """extract_from_file fonksiyonunu test eder."""
        # Mock
        mock_extract_text.return_value = "Test metin içeriği"
        
        # Test dosyası oluştur
        test_file = Path(self.test_dir) / "test.pdf"
        with open(test_file, "wb") as f:
            f.write(b"Test PDF içeriği")

        # Test
        document = self.extractor.extract_from_file(test_file)
        
        # Doğrulama
        assert isinstance(document, Document)
        assert document.filename == "test.pdf"
        assert document.text == "Test metin içeriği"
        
        # Mock fonksiyonunun çağrıldığını doğrula
        mock_extract_text.assert_called_once_with(test_file)

    def test_extract_from_file_not_found(self):
        """Olmayan dosya için extract_from_file fonksiyonunu test eder."""
        with pytest.raises(FileNotFoundError):
            self.extractor.extract_from_file("olmayan_dosya.pdf")

    def test_extract_from_file_invalid_extension(self):
        """Geçersiz uzantı için extract_from_file fonksiyonunu test eder."""
        # Test dosyası oluştur
        test_file = Path(self.test_dir) / "test.txt"
        with open(test_file, "w") as f:
            f.write("Test metin içeriği")

        with pytest.raises(ValueError):
            self.extractor.extract_from_file(test_file)

    @patch("src.etl.extract.PDFExtractor.extract_from_file")
    def test_batch_extract(self, mock_extract_from_file):
        """batch_extract fonksiyonunu test eder."""
        # Mock
        doc1 = Document(filename="test1.pdf", file_path="test1.pdf", text="Test 1")
        doc2 = Document(filename="test2.pdf", file_path="test2.pdf", text="Test 2")
        mock_extract_from_file.side_effect = [doc1, doc2]
        
        # Test
        file_paths = ["test1.pdf", "test2.pdf"]
        documents = self.extractor.batch_extract(file_paths)
        
        # Doğrulama
        assert len(documents) == 2
        assert documents[0].filename == "test1.pdf"
        assert documents[1].filename == "test2.pdf"
        
        # Mock fonksiyonlarının çağrıldığını doğrula
        assert mock_extract_from_file.call_count == 2


# tests/test_transform.py
"""
Metin dönüştürme modülü için test dosyası.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.etl.transform import TextTransformer
from src.models.document import Document


class TestTextTransformer:
    """TextTransformer sınıfı için test sınıfı."""

    def setup_method(self):
        """Her test öncesi çalışır."""
        self.transformer = TextTransformer(chunk_size=100, chunk_overlap=20)

    def test_clean_text(self):
        """clean_text fonksiyonunu test eder."""
        # Test girdisi
        text = "  Bu  bir   test   metnidir.  \n\n  Fazla   boşluklar  temizlenmelidir.  "
        
        # Test
        cleaned_text = self.transformer.clean_text(text)
        
        # Doğrulama
        assert "  " not in cleaned_text  # Çift boşluk olmamalı
        assert cleaned_text.startswith("Bu")  # Baştaki boşluklar temizlenmeli
        assert cleaned_text.endswith("temizlenmelidir.")  # Sondaki boşluklar temizlenmeli

    @patch("src.etl.transform.RecursiveCharacterTextSplitter.split_text")
    def test_split_text(self, mock_split_text):
        """split_text fonksiyonunu test eder."""
        # Mock
        mock_split_text.return_value = ["Parça 1", "Parça 2", "Parça 3"]
        
        # Test
        text = "Bu bir test metnidir. Parçalara ayrılmalıdır."
        chunks = self.transformer.split_text(text)
        
        # Doğrulama
        assert len(chunks) == 3
        assert chunks[0] == "Parça 1"
        assert chunks[1] == "Parça 2"
        assert chunks[2] == "Parça 3"
        
        # Mock fonksiyonunun çağrıldığını doğrula
        mock_split_text.assert_called_once_with(text)

    def test_split_text_empty(self):
        """Boş metin için split_text fonksiyonunu test eder."""
        chunks = self.transformer.split_text("")
        assert chunks == []

    @patch("src.etl.transform.TextTransformer.clean_text")
    @patch("src.etl.transform.TextTransformer.split_text")
    def test_process_document(self, mock_split_text, mock_clean_text):
        """process_document fonksiyonunu test eder."""
        # Mock
        mock_clean_text.return_value = "Temizlenmiş metin"
        mock_split_text.return_value = ["Parça 1", "Parça 2"]
        
        # Test belgesi oluştur
        document = Document(
            filename="test.pdf",
            file_path="test.pdf",
            text="Test metin",
            metadata={"source": "test.pdf"}
        )
        
        # Test
        processed_doc = self.transformer.process_document(document)
        
        # Doğrulama
        assert processed_doc.text == "Temizlenmiş metin"
        assert len(processed_doc.chunks) == 2
        assert processed_doc.chunks[0]["text"] == "Parça 1"
        assert processed_doc.chunks[0]["metadata"]["chunk_id"] == 0
        assert processed_doc.chunks[1]["text"] == "Parça 2"
        assert processed_doc.chunks[1]["metadata"]["chunk_id"] == 1
        
        # Meta verilerin doğru şekilde kopyalandığını doğrula
        assert processed_doc.chunks[0]["metadata"]["source"] == "test.pdf"
        assert processed_doc.chunks[0]["metadata"]["filename"] == "test.pdf"


# tests/test_load.py
"""
Veri yükleme modülü için test dosyası.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.etl.load import VectorStoreLoader
from src.models.document import Document


class TestVectorStoreLoader:
    """VectorStoreLoader sınıfı için test sınıfı."""

    def setup_method(self):
        """Her test öncesi çalışır."""
        self.embedding_model = MagicMock()
        self.loader = VectorStoreLoader(embedding_model=self.embedding_model)

    @patch("src.etl.load.FAISS.from_texts")
    def test_create_vector_store(self, mock_from_texts):
        """create_vector_store fonksiyonunu test eder."""
        # Mock
        mock_vector_store = MagicMock()
        mock_from_texts.return_value = mock_vector_store
        
        # Test
        texts = ["Metin 1", "Metin 2"]
        metadatas = [{"source": "test1.pdf"}, {"source": "test2.pdf"}]
        vector_store = self.loader.create_vector_store(texts, metadatas)
        
        # Doğrulama
        assert vector_store == mock_vector_store
        mock_from_texts.assert_called_once_with(
            texts=texts,
            embedding=self.embedding_model,
            metadatas=metadatas
        )

    def test_create_vector_store_empty(self):
        """Boş metin listesi için create_vector_store fonksiyonunu test eder."""
        vector_store = self.loader.create_vector_store([], [])
        assert vector_store is None

    @patch("src.etl.load.VectorStoreLoader.create_vector_store")
    def test_load_document_new_store(self, mock_create_vector_store):
        """Yeni vektör veritabanı için load_document fonksiyonunu test eder."""
        # Mock
        mock_vector_store = MagicMock()
        mock_create_vector_store.return_value = mock_vector_store
        
        # Test belgesi oluştur
        document = Document(
            filename="test.pdf",
            file_path="test.pdf",
            text="Test metin",
            chunks=[
                {"text": "Parça 1", "metadata": {"source": "test.pdf", "chunk_id": 0}},
                {"text": "Parça 2", "metadata": {"source": "test.pdf", "chunk_id": 1}},
            ]
        )
        
        # Test
        vector_store = self.loader.load_document(document)
        
        # Doğrulama
        assert vector_store == mock_vector_store
        mock_create_vector_store.assert_called_once_with(
            ["Parça 1", "Parça 2"],
            [
                {"source": "test.pdf", "chunk_id": 0},
                {"source": "test.pdf", "chunk_id": 1},
            ]
        )

    @patch("src.etl.load.FAISS")
    def test_load_document_existing_store(self, mock_faiss):
        """Mevcut vektör veritabanı için load_document fonksiyonunu test eder."""
        # Mock
        mock_vector_store = MagicMock()
        self.loader.vector_store = mock_vector_store
        
        # Test belgesi oluştur
        document = Document(
            filename="test.pdf",
            file_path="test.pdf",
            text="Test metin",
            chunks=[
                {"text": "Parça 1", "metadata": {"source": "test.pdf", "chunk_id": 0}},
                {"text": "Parça 2", "metadata": {"source": "test.pdf", "chunk_id": 1}},
            ]
        )
        
        # Test
        vector_store = self.loader.load_document(document)
        
        # Doğrulama
        assert vector_store == mock_vector_store
        mock_vector_store.add_texts.assert_called_once_with(
            texts=["Parça 1", "Parça 2"],
            metadatas=[
                {"source": "test.pdf", "chunk_id": 0},
                {"source": "test.pdf", "chunk_id": 1},
            ]
        )

    def test_load_document_no_chunks(self):
        """Parçasız belge için load_document fonksiyonunu test eder."""
        # Test belgesi oluştur (parçasız)
        document = Document(
            filename="test.pdf",
            file_path="test.pdf",
            text="Test metin"
        )
        
        # Test
        vector_store = self.loader.load_document(document)
        
        # Doğrulama
        assert vector_store is None


# tests/test_utils.py
"""
Yardımcı modüller için test dosyası.
"""

import os
import pytest
from unittest.mock import MagicMock, patch

from src.utils.pdf_parser import extract_text_from_pdf
from src.utils.embeddings import get_embeddings_model
from src.utils.openai_client import get_openai_client, get_openai_chat, QASystem


class TestPDFParser:
    """PDF işleyici yardımcı fonksiyonları için test sınıfı."""

    @patch("src.utils.pdf_parser.PyPDF2.PdfReader")
    def test_extract_text_from_pdf(self, mock_pdf_reader):
        """extract_text_from_pdf fonksiyonunu test eder."""
        # Mock
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Sayfa 1 içeriği"
        
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Sayfa 2 içeriği"
        
        mock_reader_instance = MagicMock()
        mock_reader_instance.pages = [mock_page1, mock_page2]
        
        mock_pdf_reader.return_value = mock_reader_instance
        
        # Test dosyası oluştur
        test_dir = "test_utils"
        os.makedirs(test_dir, exist_ok=True)
        test_file = os.path.join(test_dir, "test.pdf")
        
        with open(test_file, "wb") as f:
            f.write(b"Test PDF içeriği")
            
        try:
            # Test
            text = extract_text_from_pdf(test_file)
            
            # Doğrulama
            assert "Sayfa 1 içeriği" in text
            assert "Sayfa 2 içeriği" in text
            
            # Mock fonksiyonlarının çağrıldığını doğrula
            mock_pdf_reader.assert_called_once()
            assert mock_page1.extract_text.call_count == 1
            assert mock_page2.extract_text.call_count == 1
            
        finally:
            # Temizlik
            os.remove(test_file)
            os.rmdir(test_dir)

    def test_extract_text_from_pdf_not_found(self):
        """Olmayan dosya için extract_text_from_pdf fonksiyonunu test eder."""
        with pytest.raises(FileNotFoundError):
            extract_text_from_pdf("olmayan_dosya.pdf")


class TestEmbeddings:
    """Gömme (embedding) yardımcı fonksiyonları için test sınıfı."""

    @patch("src.utils.embeddings.OpenAIEmbeddings")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_api_key"})
    def test_get_embeddings_model(self, mock_embeddings):
        """get_embeddings_model fonksiyonunu test eder."""
        # Mock
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        # Test
        embeddings = get_embeddings_model()
        
        # Doğrulama
        assert embeddings == mock_embeddings_instance
        mock_embeddings.assert_called_once_with(
            model="text-embedding-3-small",
            openai_api_key="test_api_key"
        )

    def test_get_embeddings_model_no_api_key(self):
        """API anahtarı olmadan get_embeddings_model fonksiyonunu test eder."""
        # API anahtarını geçici olarak temizle
        original_key = os.environ.pop("OPENAI_API_KEY", None)
        
        try:
            with pytest.raises(ValueError):
                get_embeddings_model()
        finally:
            # API anahtarını geri yükle
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key


class TestOpenAIClient:
    """OpenAI istemci yardımcı fonksiyonları için test sınıfı."""

    @patch("src.utils.openai_client.OpenAI")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_api_key"})
    def test_get_openai_client(self, mock_openai):
        """get_openai_client fonksiyonunu test eder."""
        # Mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Test
        client = get_openai_client()
        
        # Doğrulama
        assert client == mock_client
        mock_openai.assert_called_once_with(api_key="test_api_key")

    @patch("src.utils.openai_client.ChatOpenAI")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_api_key"})
    def test_get_openai_chat(self, mock_chat_openai):
        """get_openai_chat fonksiyonunu test eder."""
        # Mock
        mock_chat = MagicMock()
        mock_chat_openai.return_value = mock_chat
        
        # Test
        chat = get_openai_chat()
        
        # Doğrulama
        assert chat == mock_chat
        mock_chat_openai.assert_called_once_with(
            model_name="gpt-4-turbo-preview",
            temperature=0,
            openai_api_key="test_api_key"
        )

    @patch("src.utils.openai_client.ChatOpenAI")
    @patch("src.utils.openai_client.RetrievalQA.from_chain_type")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test_api_key"})
    def test_qa_system_init(self, mock_retrieval_qa, mock_chat_openai):
        """QASystem sınıfının başlatıcısını test eder."""
        # Mock
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        mock_qa_chain = MagicMock()
        mock_retrieval_qa.return_value = mock_qa_chain
        
        mock_vector_store = MagicMock()
        mock_retriever = MagicMock()
        mock_vector_store.as_retriever.return_value = mock_retriever
        
        # Test
        qa_system = QASystem(mock_vector_store)
        
        # Doğrulama
        assert qa_system.vector_store == mock_vector_store
        assert qa_system.llm == mock_llm
        assert qa_system.qa_chain == mock_qa_chain
        mock_vector_store.as_retriever.assert_called_once()
        mock_retrieval_qa.assert_called_once()