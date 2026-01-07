import logging
import sys
import os
import pickle
from typing import Optional
from langchain.tools import tool
from src.vectordb.vectorize import get_chroma_client
from src.tools.utils import rerank_documents
from langchain_community.retrievers import BM25Retriever

# Logger Ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

# Veritabanı Bağlantısı
vector_db = get_chroma_client()

# BM25 Önbelleği (RAM'de tutarak hızı koruyoruz)
_CACHED_BM25 = None

def get_bm25_retriever():
    """Pickle dosyasından hazır BM25 indeksini bir kez yükler."""
    global _CACHED_BM25
    if _CACHED_BM25 is None:
        # Yolunu kendi yapına göre kontrol et (chromadb_bm25 klasörü demiştik)
        pkl_path = "./chromadb/bm25_retriever.pkl"
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                _CACHED_BM25 = pickle.load(f)
            logger.info("✅ BM25 Hazır Paket RAM'e yüklendi.")
        else:
            logger.error("❌ BM25 pkl bulunamadı! Hibrit arama tam kapasite çalışamayabilir.")
    return _CACHED_BM25

#region Point Search
@tool
def point_search_tool(query: str, target_source: Optional[str] = None) -> str:
    """
    NOKTA ATIŞI ARAMA (Precision Search):
    Belirli bir konu hakkında net bilgi arar.
    
    Args:
        query (str): Arama sorgusu.
        target_source (str, optional): Eğer belirli bir belge içinde aranacaksa dosya adı (örn: 'KVKK.pdf'). Yoksa None.
    """

    # 1. KANAT: Vektör Araması
    try:
        vector_results = vector_db.similarity_search(query, k=10)
        #logger.info(f"vector search: {vector_results}")
    except Exception as e:
        logger.error(f"Vektör arama hatası: {e}")
        vector_results = []
    
    # 2. KANAT: BM25 (Kelime bazlı)
    bm25 = get_bm25_retriever()
    bm25_results = []
    
    if bm25:
        # Eğer hedef kaynak varsa, BM25'ten daha fazla veri çekip sonra filtreliyoruz
        # (Çünkü BM25'te native filter yok, Python tarafında eliyoruz)
        bm25.k = 10
        raw_bm25 = bm25.invoke(query)
        
        if target_source:
            bm25_results = [doc for doc in raw_bm25 if doc.metadata.get("source") == target_source]
        else:
            bm25_results = raw_bm25
        
        #logger.info(f"bm25: {bm25_results}")

    # 3. ADIM: Adayları birleştir
    combined_results = {doc.page_content: doc for doc in (vector_results + bm25_results)}.values()
    
    # Eğer filtreleme sonucu eldeki veri sıfırsa erken dön
    if not combined_results:
        msg = f"'{target_source}' kaynağında aradığınız bilgi bulunamadı." if target_source else "Sonuç bulunamadı."
        return msg

    # 4. ADIM: Reranking
    final_docs = rerank_documents(query, list(combined_results), top_k=3) # Point search olduğu için az ve öz
    
    # ADIM 5: Formatlama
    context = ""
    for i, doc in enumerate(final_docs):
        src = doc.metadata.get("source", "Bilinmiyor")
        madde = doc.metadata.get("madde_no", "-")
        context += f"--- SONUÇ {i+1} (KAYNAK: {src} | {madde}) ---\n{doc.page_content}\n\n"
        
    return context if context else "Aradığınız kriterlere uygun net bir bilgi bulunamadı."

#region Broad Search
@tool
def broad_search_tool(query: str) -> str:
    """
    GENİŞ ARAMA (Discovery Search):
    Konuyu anlamak için tüm kaynaklardan geniş kapsamlı ve çeşitli bilgi toplar.
    
    Args:
        query (str): Arama sorgusu.
    """
    logger.info(f"🌐 GENİŞ ARAMA Başlatıldı: {query}")
    
    # ADIM 1: MMR Arama (Vektör Çeşitliliği - Filtresiz)
    mmr_docs = vector_db.max_marginal_relevance_search(
        query, 
        k=20, 
        fetch_k=30, 
        lambda_mult=0.5
    )
    
    # ADIM 2: BM25 (Anahtar kelime takviyesi)
    bm25 = get_bm25_retriever()
    bm25_docs = []
    
    if bm25:
        bm25.k = 10 # Havuzu geniş tutuyoruz
        bm25_docs = bm25.invoke(query)
    
    # ADIM 3: Birleştirme & Gruplama
    all_candidates = list({doc.page_content: doc for doc in (mmr_docs + bm25_docs)}.values())
    
    docs_by_source = {}
    for doc in all_candidates:
        source = doc.metadata.get("source", "Bilinmiyor")
        if source not in docs_by_source:
            docs_by_source[source] = []
        
        # İçerik tekrarını önle
        if doc.page_content not in [d.page_content for d in docs_by_source[source]]:
            docs_by_source[source].append(doc)

    # ADIM 4: Round Robin (Sırayla Seçme - Adil Dağılım)
    # Her kaynaktan eşit sayıda veri alarak çeşitliliği garanti ediyoruz
    diverse_selection = []
    max_items_per_source = 3 # Her kaynaktan en fazla 3 tane al
    
    for i in range(max_items_per_source):
        for source in docs_by_source:
            if i < len(docs_by_source[source]):
                diverse_selection.append(docs_by_source[source][i])
                    
    # ADIM 5: Reranking (Final Kalite Kontrol)
    final_docs = rerank_documents(query, diverse_selection, top_k=6)
    
    context = ""
    for i, doc in enumerate(final_docs):
        src = doc.metadata.get("source", "Bilinmiyor")
        madde = doc.metadata.get("madde_no", "-")
        context += f"--- DOKÜMAN {i+1} (KAYNAK: {src} | {madde}) ---\n{doc.page_content}\n\n"
        
    return context if context else "Bulunamadı."