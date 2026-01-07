import sys
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

# --- 1. AYARLAR VE IMPORTLAR ---
# Python'un 'src' modülünü bulabilmesi için yol ayarı (Senin kodundaki gibi)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Senin Supervisor yapını barındıran graph'ı çekiyoruz
    from src.main_graph import app as graph_app
except ImportError as e:
    raise RuntimeError(f"❌ HATA: Modüller yüklenemedi. 'src.main_graph' bulunamadı. Detay: {e}")

# --- 2. FASTAPI KURULUMU ---
app = FastAPI(
    title="Hukuk Asistanı API",
    description="Supervisor mimarili (Router -> Summarizer | RAG Agent) AI Asistanı",
    version="1.0.0"
)

# --- 3. VERİ MODELLERİ (Pydantic) ---
# Kullanıcıdan gelecek veri formatı
class ChatRequest(BaseModel):
    question: str

# Kullanıcıya döneceğimiz veri formatı
class ChatResponse(BaseModel):
    response: str           # Asistanın cevabı
    route: Optional[str]    # Hangi yola gitti? (RAG veya Summarizer vb.)
    rag_decision: Optional[str] = None # Eğer RAG ise analiz türü
    elapsed_time: float     # Süre

# --- 4. ENDPOINT ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Kullanıcı sorusunu alır, Graph'ı (Supervisor) çalıştırır ve sonucu döner.
    """
    start_time = time.time()
    
    # Boş soru kontrolü
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Soru boş olamaz.")

    print(f"📩 Yeni İstek Geldi: {request.question}")

    # --- LANGGRAPH / LANGCHAIN INVOKE ---
    # Senin CLI'daki mantığın aynısı:
    initial_state = {
        "question": request.question,
        "next_step": None,
        "response": None
    }

    try:
        # Graph'ı çalıştırıyoruz (Senin app.invoke kısmı)
        result = graph_app.invoke(initial_state)
        
        elapsed = time.time() - start_time

        # Sonuçları ayıklama
        final_response = result.get("response", "⚠️ Cevap üretilemedi.")
        route_decision = result.get("next_step", "Bilinmiyor")
        rag_details = result.get("decision", None) # Eğer varsa detay

        # Konsola log basalım (Opsiyonel, debug için iyi olur)
        print(f"🧭 Rota: {route_decision}")
        print(f"✅ Cevap üretildi ({elapsed:.2f}sn)")

        return ChatResponse(
            response=final_response,
            route=route_decision,
            rag_decision=rag_details,
            elapsed_time=round(elapsed, 2)
        )

    except Exception as e:
        print(f"❌ HATA: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 5. SAĞLIK KONTROLÜ (Opsiyonel) ---
@app.get("/")
async def root():
    return {"status": "active", "message": "Hukuk Asistanı API Hazır 🚀"}