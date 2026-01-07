import os
import sys
import logging
from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
import os
from langchain_google_genai import ChatGoogleGenerativeAI


# --- 1. LOGGER YAPILANDIRMASI ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

logger.info("🚀 Rag Agent modülü yüklenmeye başladı...")

# --- 2. TOOL IMPORTLARI ---
try:
    logger.info("📦 Search Tool'lar (point/broad) içe aktarılıyor...")
    # Tools dosyanızın yeri src/tools/search_tools.py varsayılmıştır
    from src.tools.search_tools import point_search_tool, broad_search_tool
    logger.info("✅ Tool'lar başarıyla yüklendi.")
except Exception as e:
    logger.error(f"❌ Tool'lar yüklenirken hata oluştu: {e}")
    raise

from langgraph.graph import StateGraph, END

# LLM Ayarı
"""llm = ChatOllama(
    model="gemma3:12b", 
    temperature=0.0, # Analiz için 0 yaptık, kararlı olsun
)
llm1 = ChatOllama(
    model="gemma3:4b-it-qat", 
    temperature=0.0, # Analiz için 0 yaptık, kararlı olsun
)"""

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
llm1 = ChatGoogleGenerativeAI(
    model=gemini_model_name,
    temperature=0,
    max_retries=2,
    google_api_key=google_api_key 
)

# --- 3. DİNAMİK DOSYA LİSTESİ ALUCU ---
def get_available_files(data_path="./data"):
    """Data klasöründeki PDF dosyalarının listesini çeker."""
    try:
        files = [f for f in os.listdir(data_path) if f.endswith('.pdf')]
        return files if files else ["Veritabanında dosya bulunamadı."]
    except Exception:
        return ["Veri klasörü okunamadı."]

# --- 4. STATE VE ŞEMA ---
class RagAgentState(TypedDict):
    question: str
    decision: str
    target_source: Optional[str] # <--- YENİ: Hedef Dosya Adı
    search_query: str
    retrieved_context: str
    response: str

class AnalysisResult(BaseModel):
    decision: str = Field(description="'Q1' (Nokta Atışı) veya 'Q2' (Geniş Arama)")
    target_source: Optional[str] = Field(
        description="Eğer kullanıcı belirli bir belgeyi kastediyorsa tam dosya adı (Listeden seç), yoksa None",
        default=None
    )

# --- 5. DÜĞÜMLER (NODES) ---

def analyzer_node(state: RagAgentState):
    logger.info("🧠 [ANALIZER] Soru ve Hedef Kaynak analiz ediliyor...")
    
    # Mevcut dosyaları çekip prompta gömüyoruz
    available_files = get_available_files()
    files_str = ", ".join(available_files)
    
    structured_llm = llm1.with_structured_output(AnalysisResult)
    
    prompt = f"""Sen uzman bir Hukuk Bilgi Mimarı ve Arama Yöneticisisin.
    
    ### MEVCUT KAYNAKLAR (DOSYALAR):
    [{files_str}]
    
    Görevin kullanıcı sorusunu analiz ederek 3 çıktı üretmektir:
    
    1. STRATEJİ SEÇİMİ (decision) - KRİTİK ADIM
    
    **Q1 (ODAKLI ARAMA - "Bul ve Getir"):**
        - Belirli bir belge ile ilgili soru soruluyorsa.
        - Bir terimin resmi tanımı soruluyorsa 
        - Belirli bir sayı, süre veya limit soruluyorsa.
        - "Listele", "Say", "Nedir" gibi net olgusal talepler.
        
    **Q2 (GENİŞ/KEŞİF ARAMA - "Araştır ve Sentezle"):**
        - Süreç ve Prosedür soruları
        - Yükümlülükler ve genel sorumluluklar
        - Senaryo ve Örnek Olaylar
        - Kıyaslama soruları 
    2. **target_source (Hedef Kaynak):**
       - Kullanıcı sorusunda yukarıdaki dosya listesinden birine atıf yapıyor mu? (Örn: "KVKK'da", "Yönetmelikte").
       - EĞER YAPIYORSA: Listeden en uygun dosya adını TAM OLARAK kopyala (Örn: 'KVKK_Kanunu.pdf').
       - EĞER YAPMIYORSA veya GENEL SORUYORSA: null (None) döndür.
    
    HAM SORU: {state['question']}"""
    
    result = structured_llm.invoke(prompt)
    
    source_log = result.target_source if result.target_source else "TÜMÜ"
    logger.info(f"⚖️ KARAR: {result.decision} | KAYNAK: {source_log} | SORGU: {state['question']}")
    
    return {
        "decision": result.decision,
        "search_query": state['question'],
        "target_source": result.target_source # State'e kaydet
    }

def search_node(state: RagAgentState):
    decision = state["decision"]
    query = state["search_query"]
    target = state["target_source"] # State'den oku
    
    # Tool'lara parametreleri sözlük (dict) olarak geçiyoruz
    tool_args = {"query": query, "target_source": target}
    
    if decision == "Q1":
        logger.info(f"🎯 [SEARCH] Nokta Atışı Tetiklendi -> Kaynak: {target if target else 'None'}")
        context = point_search_tool.invoke(tool_args)
    else:
        logger.info(f"🌐 [SEARCH] Geniş Arama Tetiklendi -> Kaynak: {target if target else 'None'}")
        context = broad_search_tool.invoke(tool_args)
        
    logger.info(f"📚 [SEARCH] Veri çekildi (Uzunluk: {len(context)} karakter)")
    return {"retrieved_context": context}

def quality_control_node(state: RagAgentState):
    # İleride buraya "Context boşsa tekrar ara" mantığı eklenebilir
    return state

def responder_node(state: RagAgentState):
    logger.info("✍️ [RESPONDER] Cevap yazılıyor...")
    
    # Prompt'a hangi kaynağa bakıldığını da ekleyelim ki LLM bilsin
    source_info = f"Odaklanılan Kaynak: {state['target_source']}" if state['target_source'] else "Kaynak: Tüm Veritabanı"
    
    prompt = f"""Sen profesyonel bir hukuk asistanısın. Aşağıdaki bağlamı kullanarak soruyu cevapla.

    BAĞLAM:
    {state['retrieved_context']}
    
    SORU: {state['question']}
    
    Cevabı hukuki dille, maddelere atıf yaparak ve net bir şekilde ver. Cevabında kaynakları belirt."""
    
    res = llm1.invoke(prompt)
    logger.info("✅ [RESPONDER] İşlem tamam.")
    return {"response": res.content}

# --- 6. WORKFLOW KURULUMU ---

def create_rag_agent():
    workflow = StateGraph(RagAgentState)

    workflow.add_node("analizer", analyzer_node)
    workflow.add_node("search", search_node)
    workflow.add_node("quality_control", quality_control_node)
    workflow.add_node("responder", responder_node)

    workflow.set_entry_point("analizer")
    workflow.add_edge("analizer", "search")
    workflow.add_edge("search", "quality_control")
    workflow.add_edge("quality_control", "responder")
    workflow.add_edge("responder", END)

    return workflow.compile()

rag_agent = create_rag_agent()

# --- 7. TEST ---
if __name__ == "__main__":
    # Test Senaryosu 1: Kaynak Belirtilmiş
    print("\n--- TEST 1: Kaynaklı Sorgu ---")
    try:
        # Örnek: Data klasöründe 'KVKK_Kanunu.pdf' olduğunu varsayıyoruz
        q1 = "KVKK metninde madde 5 ne diyor?" 
        final_state = rag_agent.invoke({"question": q1})
        print(f"CEVAP: {final_state['response'][:200]}...") 
    except Exception as e:
        print(f"Hata: {e}")

    # Test Senaryosu 2: Genel Sorgu
    print("\n--- TEST 2: Genel Sorgu ---")
    try:
        q2 = "Veri sorumlusunun yükümlülükleri nelerdir?"
        final_state = rag_agent.invoke({"question": q2})
        print(f"CEVAP: {final_state['response'][:200]}...")
    except Exception as e:
        print(f"Hata: {e}")