# PDF Makale Soru-Cevap Uygulaması

Bu proje, PDF makalelerini analiz edip üzerinde soru-cevap işlemleri yapabilen bir Streamlit web uygulamasıdır. Uygulama, yüklenen PDF'leri işleyerek OpenAI API aracılığıyla sorulara cevap verir.

## Özellikler

- PDF dosyalarını yükleme ve işleme
- Metin çıkarma ve dönüştürme
- Vektör veritabanına belge yükleme
- OpenAI API ile semantik arama ve soru-cevap
- Kullanıcı dostu Streamlit arayüzü

## Kurulum

### Gereksinimler

- Python 3.8+
- OpenAI API anahtarı

### Adımlar

1. Bu repoyu klonlayın:
   ```
   git clone https://github.com/yourusername/PdfChatbot.git
   cd PdfChatbot
   ```

2. Sanal ortam oluşturun ve etkinleştirin:
   ```
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # veya
   venv\Scripts\activate  # Windows
   ```

3. Bağımlılıkları yükleyin:
   ```
   pip install -e .
   ```

4. `.env` dosyası oluşturun ve OpenAI API anahtarınızı ekleyin:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Kullanım

1. Uygulamayı başlatın:
   ```
   streamlit run src/app.py
   ```

2. Web tarayıcınızda `localhost:8501` adresine gidin
3. PDF makale(ler) yükleyin
4. Sorularınızı sorun ve yanıtlarınızı alın

## ETL Pipeline

Uygulama, aşağıdaki ETL (Extract, Transform, Load) sürecini takip eder:

1. **Extract**: PDF dosyalarından metin çıkarılır
2. **Transform**: Metin parçalara ayrılır ve işlenir
3. **Load**: İşlenmiş metin vektör veritabanına yüklenir

## Geliştirme

### Kod Standartları

Bu proje flake8, black ve isort kullanarak kod formatı standartlarını uygular:

```
flake8 src tests
black src tests
isort src tests
```

### Testler

Testleri çalıştırmak için:

```
pytest
```

## Lisans

Bu proje MIT Lisansı ile lisanslanmıştır.
