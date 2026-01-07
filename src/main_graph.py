import logging
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# --- 1. MODÜLLERİ İÇE AKTAR (COMPILED AGENT'LARI ALIYORUZ) ---
try:
    # A. Supervisor'ın KENDİSİNİ alıyoruz (Compiled Graph)
    # Artık supervisor_node fonksiyonuyla işimiz yok.
    from src.agents.supervisor_agent import supervisor_agent
    
    # B. RAG Agent'ın KENDİSİNİ alıyoruz (Compiled Graph)
    from src.agents.rag_agent import rag_agent
    
    # C. Summarizer (Bu tek bir node/tool olduğu için fonksiyon olarak kalabilir)
    from src.agents.summarize_node import summarize_node
    
except ImportError as e:
    print(f"❌ Import hatası: {e}")
    raise

# Env ve Logger
load_dotenv()
logger = logging.getLogger(__name__)

# --- 2. ORTAK STATE (Hafıza) ---
# Subgraph'lar arası veri kaybı olmaması için geniş bir state tutuyoruz.
class MainAgentState(TypedDict):
    question: str               
    next_step: Optional[str]    # Supervisor çıktısı
    
    # RAG Agent çıktıları
    decision: Optional[str]     
    search_query: Optional[str]
    retrieved_context: Optional[str]
    
    response: Optional[str]     # Nihai cevap

# --- 3. ANA ORKESTRA (MAIN GRAPH) ---

def create_main_graph():
    
    workflow = StateGraph(MainAgentState)

    # --- DÜĞÜMLERİ EKLE (SUBGRAPHS & NODES) ---
    
    # 1. Supervisor SUBGRAPH
    # LangGraph, bir düğüm yerine başka bir 'Compiled Graph' koymana izin verir.
    # Supervisor işini bitirip END'e ulaştığında, çıktısını buraya bırakır.
    workflow.add_node("supervisor_brain", supervisor_agent)
    
    # 2. RAG SUBGRAPH (Mavi Kutu)
    workflow.add_node("rag_machinery", rag_agent)
    
    # 3. Summarizer NODE
    workflow.add_node("summarizer_tool", summarize_node)

    # --- AKIŞ ---

    # Giriş -> Supervisor Brain (Subgraph)
    workflow.set_entry_point("supervisor_brain")

    # Supervisor Subgraph'ı işini bitirdiğinde state'teki 'next_step'e bakıyoruz
    workflow.add_conditional_edges(
        "supervisor_brain", 
        lambda x: x["next_step"], 
        {
            "Q3": "summarizer_tool",  # Özetçiye git
            "RAG": "rag_machinery"    # Mavi Kutuya git
        }
    )

    # --- BİTİŞ BAĞLANTILARI ---
    
    # Summarizer işini bitirince -> END
    workflow.add_edge("summarizer_tool", END)
    
    # RAG Makinesi işini bitirince -> END
    workflow.add_edge("rag_machinery", END)

    return workflow.compile()

# --- 4. APP ---
app = create_main_graph()