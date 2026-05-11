# 🏛️ İSTE Akıllı Mevzuat Asistanı (RAG Platformu)

İSTE Akıllı Mevzuat Asistanı, İskenderun Teknik Üniversitesi (İSTE) öğrencileri ve personeli için yönetmelik ve mevzuat sorgulama süreçlerini hızlandırmak amacıyla geliştirilmiş, **Retrieval-Augmented Generation (RAG)** mimarisine sahip yapay zeka tabanlı bir platformdur.

Kullanıcılar; karmaşık yönetmelik PDF'leri arasında kaybolmadan, doğrudan doğal dille sorular sorabilir ve anlamsal arama (Semantic Search) sayesinde saniyeler içinde Llama 3.3 destekli kesin ve referanslı cevaplar alabilirler.

## 🧭 Özellikler

🤖 **Gelişmiş RAG Mimarisi:**

* **Akıllı Metin Parçalama (Smart Chunking):** Resmi belgeler rastgele değil, özel Regex algoritmalarıyla doğrudan "MADDE" başlıklarına göre anlamsal bütünlük korunarak vektörleştirilir.
* **Çok Dilli Vektör Uzayı:** Türkçe hukuk ve eğitim terimlerini (AKTS, GNO vb.) en yüksek doğrulukla anlamlandırmak için `paraphrase-multilingual` Embedding modeli kullanılmıştır.
* **Niyet Analizi (Intent Filtering):** Yapay zeka, genel sohbetler ile mevzuat sorularını ayırt ederek halüsinasyon (uydurma) yapmadan yalnızca veritabanındaki kesin kanıtlara dayalı yanıt verir.

🔍 **Orijinal Kaynak Gösterimi:** Asistan bir cevap ürettiğinde, cevabın alındığı orijinal yönetmelik PDF'i arayüzde otomatik olarak açılır ve şeffaf bir kanıt sunar.

⚙️ **Dinamik Yönetim Paneli (Live Ingestion):** Sistem kapatılmadan, yönetici paneli üzerinden yeni yönetmelik PDF'leri yüklenerek sistem anında eğitilebilir.

⚡ **Canlı Sistem Telemetrisi:** Yapay zekanın yanıt verme süresi, veritabanı (ChromaDB) tarama hızı ve toplam vektör sayısı gibi metrikler her sorguda milisaniye cinsinden arayüze yansıtılır.

📄 **Resmi Rapor Çıktısı & Sesli Komut:** Asistanla yapılan görüşmeler tek tıkla resmi bir PDF raporu olarak indirilebilir. Ayrıca Web Speech API entegrasyonu ile sisteme sesli komut verilebilir.

## 🛠️ Kullanılan Teknolojiler

| Teknoloji | Açıklama |
| --- | --- |
| **FastAPI (Python)** | Sistemin asenkron ve yüksek performanslı backend altyapısı |
| **ChromaDB** | Milyonlarca metin parçasını saniyeler içinde tarayan Vektör Veritabanı |
| **LangChain & Groq** | Llama 3.3 yapay zeka modelinin RAG mimarisiyle orkestrasyonu |
| **SentenceTransformers** | Metinlerin matematiksel vektörlere dönüştürülmesi (Embedding) |
| **TailwindCSS & JS** | Cam efekti (Glassmorphism) barındıran, kurumsal İSTE temalı modern arayüz |
| **html2pdf.js** | Arayüzdeki görüşmelerin anlık PDF raporlarına dönüştürülmesi |

## 🧩 Proje Yapısı

```text
iste-mevzuat-rag/
│
├── main.py             # FastAPI sunucusu ve AI/Veritabanı rotaları (Backend)
├── ingest.py           # PDF'leri okuyan, parçalayan ve ChromaDB'ye kaydeden script
├── index.html          # Kullanıcı ve Admin panelini barındıran SPA Arayüzü (Frontend)
│
├── data/               # Sistemin beslendiği yönetmelik PDF dosyalarının bulunduğu klasör
│   ├── yazokulu.pdf
│   └── lisans_yonetmeligi.pdf
│
├── chroma_db/          # Vektör veritabanının lokal olarak saklandığı klasör (Otomatik oluşur)
├── .env                # API Anahtarı (Gizli tutulmalıdır)
└── requirements.txt    # Gerekli Python kütüphaneleri listesi

```

## 🚀 Kurulum ve Çalıştırma

**1. Depoyu Klonlayın:**

```bash
git clone https://github.com/KULLANICI_ADINIZ/iste-mevzuat-rag.git
cd iste-mevzuat-rag

```

**2. Sanal Ortam Oluşturun ve Bağımlılıkları Yükleyin:**

```bash
python -m venv venv
source venv/bin/activate  # Windows için: venv\Scripts\activate
pip install fastapi uvicorn chromadb langchain-groq pypdf sentence-transformers python-dotenv pydantic

```

**3. API Anahtarını Tanımlayın:**
Ana dizinde bir `.env` dosyası oluşturun ve Groq API anahtarınızı ekleyin:

```env
GROQ_API_KEY=sizin_api_anahtariniz_buraya

```

**4. Veritabanını Oluşturun (Eğitim):**
`data` klasörüne PDF dosyalarınızı ekledikten sonra sistemi eğitmek için vektörizasyon scriptini çalıştırın:

```bash
python ingest.py

```

**5. Sunucuyu Başlatın:**

```bash
uvicorn main:app --reload

```

Tarayıcınızdan `http://localhost:8000` adresine giderek sistemi kullanmaya başlayabilirsiniz.

## 🎓 Akademik Atıf

Bu proje, İskenderun Teknik Üniversitesi (İSTE) Bilgisayar Mühendisliği Bölümü Bitirme Projesi kapsamında geliştirilmiştir. Yapay zeka destekli Vektörel Arama (Semantic Search) ve RAG mimarilerinin kurumsal veri analizindeki performansını ölçmek amacıyla tasarlanmıştır.

**Geliştirici:** Deniz Çelik

**Bölüm:** Bilgisayar Mühendisliği (İSTE)

**Danışman:** Halil İbrahim Okur
