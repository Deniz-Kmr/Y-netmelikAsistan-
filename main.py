import os
import time
import shutil
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- KLASÖR KONTROLLERİ VE PDF SUNUCUSU ---
os.makedirs("data", exist_ok=True)
app.mount("/pdfs", StaticFiles(directory="data"), name="pdfs")

# --- 1. VERİTABANI VE AI YAPILANDIRMASI ---
# TÜRKÇE DESTEKLİ GELİŞMİŞ MODEL
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db") 
collection = chroma_client.get_or_create_collection(name="yonetmelikler", embedding_function=sentence_transformer_ef)

llm = ChatGroq(temperature=0.1, model_name="llama-3.3-70b-versatile", api_key=os.environ.get("GROQ_API_KEY"))

# Canlı Logları hafızada tutacağımız liste
query_logs = [] 

# --- AKILLI PROMPT (Selamlaşma ve Alakasız Soru Kontrolü) ---
PROMPT_TEMPLATE = """Sen İSTE Yönetmelik Uzmanısın. 
SANA VERİLEN BAĞLAMI DİKKATLİCE ANALİZ ET.

TALİMATLAR:
1. Sadece sana verilen "Bağlam" (yönetmelik maddeleri) içindeki bilgilere dayanarak profesyonel bir cevap yaz.
2. Eğer öğrencinin sorusu "merhaba", "nasılsın", "selam", "günaydın" gibi genel bir selamlaşma ise, nazikçe selamını al ve onlara "Yönetmelik dışı genel sohbet yapamıyorum, İSTE yönetmelikleri hakkında ne sormak istersiniz?" şeklinde cevap ver. Bu durumda bağlamı görmezden gel.
3. Eğer öğrencinin sorusunun cevabı bağlamda YOKSA, "Bu konu hakkında yönetmeliklerde net bir bilgi bulamadım." de. Kendi bilginden uydurma.
4. Cevabını Markdown (kalın yazılar, listeler) kullanarak düzenle.

Bağlam (Resmi Mevzuat):
{context}

Öğrencinin Sorusu: {question}"""

class QueryRequest(BaseModel):
    message: str
    source: Optional[str] = "all" # Filtreleme için eklendi

# --- 2. ANA SAYFA ---
@app.get("/", response_class=HTMLResponse)
async def get_index():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Hata: index.html dosyası bulunamadı!</h1>"

# --- 3. ARAYÜZ İÇİN PDF LİSTESİNİ GETİR ---
@app.get("/sources")
async def get_sources():
    try:
        files = [f for f in os.listdir("data") if f.endswith(".pdf")]
        return {"sources": files}
    except Exception:
        return {"sources": []}

# --- 4. AKILLI RAG SORGUSU (METRİKLER, FİLTRE VE LOGLAMA İLE) ---
@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        total_start = time.time()
        
        # Filtreleme mantığı
        where_filter = {"kaynak": request.source} if request.source != "all" else None
        
        db_start = time.time()
        results = collection.query(
            query_texts=[request.message], 
            n_results=10, 
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        db_time = time.time() - db_start
        
        context_parts = []
        source_data = []
        
        if not results['documents'] or not results['documents'][0]:
            context_text = "Mevcut değil"
        else:
            for i, (doc, meta, dist) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
                kaynak = meta.get('kaynak', 'Bilinmeyen')
                madde = meta.get('madde_no', '?')
                uyum_skoru = max(0, min(100, round(100 - (dist * 45), 1)))
                
                context_parts.append(f"Kaynak: {kaynak}, Md: {madde}\n{doc}")
                source_data.append({
                    "title": f"{kaynak} - Madde {madde}", 
                    "content": doc,
                    "score": f"%{uyum_skoru}",
                    "filename": kaynak
                })
            context_text = "\n\n".join(context_parts)

        prompt = PROMPT_TEMPLATE.format(context=context_text, question=request.message)
        
        llm_start = time.time()
        response = llm.invoke([HumanMessage(content=prompt)])
        llm_time = time.time() - llm_start
        total_time = time.time() - total_start
        
        log_entry = {
            "time": time.strftime("%H:%M:%S"),
            "query": request.message,
            "latency": round(total_time, 2)
        }
        query_logs.insert(0, log_entry)
        
        return {
            "answer": response.content,
            "sources": source_data,
            "metrics": {"total_time": round(total_time, 2), "db_time": round(db_time * 1000, 0), "llm_time": round(llm_time, 2)}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. ADMIN PANELİ: İSTATİSTİKLER ---
@app.get("/admin/stats")
async def get_stats():
    return {
        "total_documents": collection.count(),
        "total_queries": len(query_logs),
        "logs": query_logs[:10]
    }

# --- 6. ADMIN PANELİ: CANLI PDF YÜKLEME ---
@app.post("/admin/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_path = f"data/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        
        # Gelişmiş parçalama
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600, 
            chunk_overlap=150, 
            separators=["\nMADDE", "\n\n", "\n", ".", " ", ""]
        )
        docs = text_splitter.split_documents(pages)
        
        texts = [doc.page_content for doc in docs]
        metadatas = [{"kaynak": file.filename, "madde_no": f"Sayfa {doc.metadata['page']}"} for doc in docs]
        ids = [f"{file.filename}_{i}_{int(time.time())}" for i in range(len(docs))]
        
        collection.add(documents=texts, metadatas=metadatas, ids=ids)
        
        return {"status": "success", "message": f"{file.filename} başarıyla öğrenildi! ({len(docs)} yeni vektör oluşturuldu)"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))