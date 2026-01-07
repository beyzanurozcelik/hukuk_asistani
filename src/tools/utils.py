import os
import logging
from typing import List
from sentence_transformers import CrossEncoder
import sys

# Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Çıktıyı doğrudan terminale zorlar
    ],
    force=True  # Başka kütüphanelerin (LangChain vb.) ayarlarını ezer
)
logger = logging.getLogger(__name__)

# RERANKER: Cross-Encoder (MULTILINGUAL)
RERANKER_MODEL_NAME = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
RERANKER_MODEL = None

try:
    logger.info(f"--- Multilingual Reranker Modeli Yükleniyor: {RERANKER_MODEL_NAME} ---")
    RERANKER_MODEL = CrossEncoder(RERANKER_MODEL_NAME)
    logger.info("✅ Multilingual Reranker Hazır.")
except Exception as e:
    logger.error(f"Reranker yüklenirken kritik hata: {e}")

#region ReRanker
def rerank_documents(query: str, docs, top_k=5):
    """
    Multi-Stage Retrieval - Aşama 2: Reranking (Yeniden Sıralama)
    """
    if not docs:
        return []
    if RERANKER_MODEL is None:
        logger.warning("Reranker modeli yüklü değil, ham vektör sonuçları dönülüyor.")
        return docs[:top_k]

    logger.info(f"Reranking işlemi {len(docs)} doküman üzerinde yapılıyor...")
    
    pairs = [[query, doc.page_content] for doc in docs]
    
    try:
        scores = RERANKER_MODEL.predict(pairs)
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        
        if scored_docs:
            logger.info(f"En iyi eşleşme skoru: {scored_docs[0][1]:.4f}")
            logger.info(f"En kötü aday skoru: {scored_docs[-1][1]:.4f}")

        final_results = [doc for doc, score in scored_docs[:top_k]]
        return final_results
        
    except Exception as e:
        logger.error(f"Reranking sırasında hata: {e}")
        return docs[:top_k]