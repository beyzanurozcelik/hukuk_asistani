import logging
from langchain_huggingface import HuggingFaceEmbeddings

# Logger Yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_embedding_model():
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    logger.info(f"Embedding modeli yükleniyor: {model_name}")
    
    try:
        encode_kwargs = {'normalize_embeddings': True}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs=encode_kwargs
        )
        logger.info("Embedding modeli başarıyla yüklendi.")
        return embeddings
    except Exception as e:
        logger.error(f"Embedding modeli yüklenirken hata oluştu: {str(e)}")
        raise