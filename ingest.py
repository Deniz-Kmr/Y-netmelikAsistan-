import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
import re
import os
import glob

# 1. Yerel Vektör Modeli
# TÜRKÇE DESTEKLİ GELİŞMİŞ MODEL
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")

# 2. ChromaDB Bağlantısı
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="yonetmelikler",
    embedding_function=sentence_transformer_ef
)

def clean_and_chunk_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"

    # PDF'ten gelen kırık satırları ve tireleri temizle ki cümleler bütünleşsin
    full_text = full_text.replace("-\n", "").replace("\n", " ")

    # Maddelere göre böl (Senin harika Regex mantığın)
    chunks = re.split(r'(?i)(?=madde\s+\d+)', full_text)
    return [c.strip() for c in chunks if len(c.strip()) > 50]

# 3. 'data' klasöründeki tüm PDF'leri bul
pdf_dosyalari = glob.glob("data/*.pdf")

if not pdf_dosyalari:
    print("Hata: 'data' klasöründe hiç PDF bulunamadı! Lütfen dosyalarınızı ekleyin.")
else:
    print(f"Toplam {len(pdf_dosyalari)} adet PDF bulundu. İşlem başlıyor...\n")
    toplam_madde = 0

    for pdf_dosyasi in pdf_dosyalari:
        dosya_adi = os.path.basename(pdf_dosyasi)
        print(f"Okunuyor: {dosya_adi} ...")
        
        maddeler = clean_and_chunk_pdf(pdf_dosyasi)

        for i, metin in enumerate(maddeler):
            # Madde numarasını tespit et
            match = re.search(r'(?i)madde\s+(\d+)', metin)
            madde_no = match.group(1) if match else f"Genel-{i}"

            collection.add(
                documents=[metin],
                metadatas=[{"kaynak": dosya_adi, "madde_no": str(madde_no)}],
                ids=[f"{dosya_adi}_{i}"] # ID'lerde boşluk veya karmaşıklık olmasın diye sadeleştirdik
            )
        
        toplam_madde += len(maddeler)
        print(f" > {dosya_adi} içinden {len(maddeler)} madde kaydedildi.")

    print(f"\nMUHTEŞEM! Tüm dosyalardan toplam {toplam_madde} madde başarıyla ChromaDB'ye yüklendi.")