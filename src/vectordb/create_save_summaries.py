import os
import logging
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from src.vectordb.embedding import get_embedding_model
import os


# --- AYARLAR ---
load_dotenv()

# Veri kaynakları ve hedef DB
DATA_PATH = "./data"
SUMMARY_DB_PATH = "./chromadb_summaries"
COLLECTION_NAME = "legal_summaries"

# Model ayarları (RTX 5060 gücüyle)
LLM_MODEL = "gemma3:12b"



# Logger kurulumu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# API Key Kontrolü
if not os.getenv("GOOGLE_API_KEY"):
    logger.error("🚨 GOOGLE_API_KEY bulunamadı! .env dosyanızı kontrol edin.")
    # İstersen manuel giriş açabilirsin:
    # os.environ["GOOGLE_API_KEY"] = input("Google API Key giriniz: ")

# --- 3. MODEL AYARLARI (GEMINI'YE GEÇİŞ) ---
from langchain_google_genai import ChatGoogleGenerativeAI

# ANALİZ İÇİN: Gemini 1.5 Flash (Çok hızlı, ucuz ve JSON çıktısı mükemmel)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_retries=2,
)

def get_files(directory):
    """Klasördeki PDF dosyalarını listeler."""
    return [f for f in os.listdir(directory) if f.endswith('.pdf')]

def generate_summary_with_llm(text, filename):
    """
    Dökümanın metnini LLM'e verir ve özet ister.
    Çok uzun dökümanlar için ilk 25.000 karakteri (yaklaşık 10-15 sayfa) baz alır.
    Genelde kanunların amacı, kapsamı ve tanımları baştadır.
    """
    #llm = ChatOllama(model=LLM_MODEL, temperature=0)
    
    # Metni çok şişirmemek için kırpıyoruz (Token limitini patlatmamak için)
    truncated_text = text[:25000] 
    
    prompt = PromptTemplate.from_template(
        """Aşağıdaki hukuki metni analiz et ve kapsamlı bir özet çıkar.
        
        GÖREVLER:
        1. Bu belgenin AMACI nedir?
        2. KAPSADIĞI ana konular nelerdir?
        3. Varsa önemli TANIMLAR veya CEZAİ YAPTIRIMLAR nelerdir?
        4. Maddeler halinde, net ve anlaşılır bir Türkçe ile özetle.
        
        BELGE ADI: {filename}
        
        METİN (Kısaltılmış):
        {text}
        
        ÖZET:"""
    )
    
    chain = prompt | llm
    logger.info(f"🤖 {filename} için Gemma 3 düşünüyor...")
    response = chain.invoke({"text": truncated_text, "filename": filename})
    
    return response.content

def create_summary_db(reset=False):
    """Özetleri oluşturur ve ChromaDB'ye kaydeder."""
    
    # 1. Eğer reset isteniyorsa eski DB'yi sil
    if reset and os.path.exists(SUMMARY_DB_PATH):
        logger.warning(f"🗑️ Eski özet veritabanı siliniyor: {SUMMARY_DB_PATH}")
        shutil.rmtree(SUMMARY_DB_PATH)

    # 2. Dosyaları Bul
    pdf_files = get_files(DATA_PATH)
    if not pdf_files:
        logger.error("❌ Data klasöründe PDF bulunamadı!")
        return

    logger.info(f"📂 Bulunan Dosyalar: {pdf_files}")
    
    summary_docs = []

    # 3. Her PDF için Döngü
    for pdf_file in pdf_files:
        file_path = os.path.join(DATA_PATH, pdf_file)
        logger.info(f"📄 İşleniyor: {pdf_file}")
        
        try:
            # PDF'i yükle ve metni birleştir
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            full_text = " ".join([p.page_content for p in pages])
            
            # LLM ile Özetle
            summary_text = generate_summary_with_llm(full_text, pdf_file)
            
            # Document objesi oluştur (Metadata çok önemli!)
            # Metadata'ya 'source' ekliyoruz ki Supervisor "kvkk.pdf" diyerek bulabilsin.
            doc = Document(
                page_content=summary_text,
                metadata={
                    "source": pdf_file,       # Örn: kvkk.pdf
                    "original_length": len(full_text),
                    "type": "summary"
                }
            )
            summary_docs.append(doc)
            logger.info(f"✅ {pdf_file} özeti hazırlandı.")
            
        except Exception as e:
            logger.error(f"❌ {pdf_file} işlenirken hata: {e}")

    # 4. ChromaDB'ye Kaydet
    if summary_docs:
        logger.info("💾 Özetler veritabanına yazılıyor...")
        embedding_fn = get_embedding_model()
         
        
        db = Chroma.from_documents(
            documents=summary_docs,
            embedding=embedding_fn,
            persist_directory=SUMMARY_DB_PATH,
            collection_name=COLLECTION_NAME
        )
        logger.info(f"🎉 İşlem Tamam! {len(summary_docs)} belge özeti kaydedildi.")
    else:
        logger.warning("⚠️ Kaydedilecek özet bulunamadı.")

if __name__ == "__main__":
    # İlk çalıştırmada reset=True yapıyoruz ki temiz başlasın
    create_summary_db(reset=True)