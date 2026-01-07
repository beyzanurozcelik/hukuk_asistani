import logging
from typing import TypedDict, Literal
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Logger
logger = logging.getLogger(__name__)

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

# --- 1. STATE & SCHEMA ---
class RagAgentState(TypedDict):
    """Tüm ajanlar arasında dolaşan ortak hafıza."""
    question: str
    next_step: str
    response: str
    # İleride eklenebilir: history: list, documents: list vb.

class SupervisorDecision(BaseModel):
    decision: Literal["Q3", "RAG"] = Field(
        description="Q3: Özet/Genel Bakış, RAG: Doküman Analizi/Detay"
    )
    reasoning: str = Field(description="Kararın gerekçesi")

# --- 2. LLM SETUP ---
#llm = ChatOllama(model="gemma3:4b-it-qat", temperature=0)
supervisor_chain = llm.with_structured_output(SupervisorDecision)

# --- 3. NODE FONKSİYONU ---
def supervisor_node(state: RagAgentState):
    """Niyet okuyan ve rotayı belirleyen düğüm."""
    logger.info("👑 [SUPERVISOR] Rota belirleniyor...")
    
    prompt = f"""Hukuk Asistanı Yöneticisisin. Kullanıcının niyetine göre rotayı belirle:

    1. Q3 (Özet/Genel): 
       - Genel özet, belge sorgusu ("neler var?", "bu nedir?" vb.), selamlaşma ("Merhaba") veya detay belirtilmeyen her türlü query. 
       - Niyet net değilse DEFAULT olarak bunu seç.

    2. RAG (Analiz/Detay): 
       - Spesifik madde ("Madde 11"), hukuki tanım, senaryo analizi veya detaylı mevzuat sorgusu.

    SORU: {state['question']}"""
    
    try:
        result = supervisor_chain.invoke(prompt)
        logger.info(f"➡️ Karar: {result.decision} ({result.reasoning})")
        return {"next_step": result.decision}
    except Exception as e:
        logger.error(f"Hata: {e}, Q3 seçiliyor.")
        return {"next_step": "Q3"}

# --- 4. WORKFLOW KURULUMU (Senin İstediğin Format) ---
def create_supervisor_agent():
    workflow = StateGraph(RagAgentState)

    # Düğümü ekle
    workflow.add_node("supervisor", supervisor_node)

    # Giriş noktası
    workflow.set_entry_point("supervisor")

    # Yol Ayrımı (Conditional Edges)
    # Not: Buradaki END'ler Main Graph'te gerçek düğümlere (Summarizer/Analyzer) bağlanacak.
    # Şu an bu ajanın görevi kararı verip çıkmak.
    workflow.add_conditional_edges(
        "supervisor",
        lambda x: x["next_step"],
        {
            "Q3": END,   # Main Graph'te -> Summarizer
            "RAG": END   # Main Graph'te -> RAG Agent (Mavi Kutu)
        }
    )

    return workflow.compile()

# Ajanı oluştur
supervisor_agent = create_supervisor_agent()