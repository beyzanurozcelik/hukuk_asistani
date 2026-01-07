import logging
import os
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.vectordb.embedding import get_embedding_model
from langchain_google_genai import ChatGoogleGenerativeAI
import os 

# Env yükle
load_dotenv()

# Logger
logger = logging.getLogger(__name__)

# --- 1. AYARLAR ---
DB_PATH = "./chromadb_summaries"  # Özetlerin olduğu ayrı DB
COLLECTION_NAME = "legal_summaries"
MODEL_NAME = "gemma3:4b-it-qat"


load_dotenv()

# --- API KEY ve MODEL ADI KONTROLÜ ---
google_api_key = os.getenv("GOOGLE_API_KEY")
gemini_model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash-lite") # Varsayılan: 1.5-flash

if not google_api_key:
    logger.error("🚨 GOOGLE_API_KEY bulunamadı! Lütfen .env dosyanızı kontrol edin.")
    # Programın burada durmasını istersen:
    # sys.exit(1) 
else:
    logger.info(f"🔑 API Key yüklendi. Hedef Model: {gemini_model_name}")

# --- 3. MODEL AYARLARI ---

# ANALİZ ve CEVAP İÇİN: .env'den gelen modeli kullanıyoruz
llm = ChatGoogleGenerativeAI(
    model=gemini_model_name,
    temperature=0,
    max_retries=2,
    google_api_key=google_api_key 
)

# --- 2. BAĞLANTI FONKSİYONLARI ---
def get_summary_db():
    """Özet veritabanına bağlanır."""
    if not os.path.exists(DB_PATH):
        logger.warning("⚠️ Özet veritabanı bulunamadı! vectorize.py ile özet oluşturulmalı.")
        return None
        
    embedding_function = get_embedding_model()

    
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME
    )

# --- 3. SUMMARIZE NODE (DÜĞÜM) ---
def summarize_node(state: dict):
    """
    Supervisor 'Q3' dediğinde çalışan fonksiyon.
    1. Kullanıcı sorusuna göre en alakalı özeti 'Mini-Retriever' ile bulur.
    2. Gemma 3 ile bu özeti kullanıcıya sunar.
    """
    question = state["question"]
    logger.info(f"📝 [SUMMARIZER] Özetleme modu devrede: {question}")
    
    # A. Veritabanına Bağlan
    db = get_summary_db()
    
    context_text = ""
    
    if db:
        # B. Mini-Retriever: Soruyla en alakalı 3 özeti getir
        # Eğer kullanıcı "KVKK" dediyse KVKK özeti en üste gelir.
        # Eğer "Neler var?" dediyse genel başlıklar gelir.
        try:
            results = db.similarity_search(question, k=3)
            
            # Gelen dökümanları birleştir
            for i, doc in enumerate(results):
                source = doc.metadata.get("source", "Bilinmiyor")
                context_text += f"\n--- BELGE: {source} ---\n{doc.page_content}\n"
                
            logger.info(f"📚 {len(results)} adet ilgili özet bulundu.")
            
        except Exception as e:
            logger.error(f"Özet ararken hata: {e}")
            context_text = "Veritabanı hatası nedeniyle özetlere erişilemedi."
    else:
        context_text = "Sistemde henüz hazır özet bulunmuyor. Lütfen belgeleri indeksleyin."

    # C. Sentezleme (Synthesis): Gemma 3 Cevabı Yazıyor
    # Not: Burada temperature biraz açık olabilir (0.3), daha doğal konuşsun.
    #llm = ChatOllama(model=MODEL_NAME, temperature=0.3)
    
    prompt_template = ChatPromptTemplate.from_template(
        """Sen yardımsever bir Hukuk Asistanısın. Kullanıcı genel bir bilgi veya özet istedi.
        Aşağıda veritabanımızdaki ilgili belgelerin HAZIR ÖZETLERİ var.
        
        GÖREVİN:
        Bu özetleri kullanarak kullanıcının sorusuna net, anlaşılır ve toparlayıcı bir cevap ver.
        Eğer kullanıcı "neler var?" gibi genel bir şey sorduysa belgeleri listele ve kısaca içeriklerinden bahset.
        
        --- BULUNAN ÖZETLER ---
        {context}
        -----------------------
        
        KULLANICI SORUSU: {question}
        
        CEVAP:"""
    )
    
    chain = prompt_template | llm
    response = chain.invoke({"context": context_text, "question": question})
    
    logger.info("✅ Özet cevabı üretildi.")
    
    # State'i güncelle ve response'u dön
    return {"response": response.content}