import os
import shutil
import logging
import sys
import stat
import time
import re
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from src.vectordb.embedding import get_embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import pickle
from langchain_community.retrievers import BM25Retriever

load_dotenv()

# Logger Ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

PERSIST_DIR = "./chromadb"

def get_chroma_client():
    embedding_function = get_embedding_model()
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedding_function,
        collection_name="legal_rag_collection"
    )

def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clear_database():
    if os.path.exists(PERSIST_DIR):
        try:
            shutil.rmtree(PERSIST_DIR, onexc=remove_readonly)
            logger.warning(f"⚠️ Veritabanı temizlendi.")
            time.sleep(1)
        except Exception:
            pass

def regex_madde_split(full_text, source_name):
    """
    Metni 'MADDE X' ibarelerine göre böler.
    """
    logger.info(f"✂️ Regex ile Madde Madde bölünüyor... ({len(full_text)} karakter)")
    
    # --- REGEX DESENİ ---
    # (?=...) : Lookahead. Yani "MADDE" kelimesini gördüğün yerden böl ama kelimeyi silme.
    # \n : Yeni satır başındaki maddeleri arar (Cümle içindekileri almaz).
    # MADDE\s+\d+ : "MADDE" + Boşluk + Sayı (Örn: MADDE 1, MADDE 14)
    pattern = r"(?=\nMADDE\s+\d+)"
    
    chunks = re.split(pattern, full_text)
    
    # Eğer hiç madde bulamazsa (Örn: Giriş kısmı, Önsöz veya Madde içermeyen belge)
    if len(chunks) < 2:
        logger.warning("⚠️ Metinde 'MADDE' yapısı bulunamadı. Standart paragraf bölmeye geçiliyor.")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " "]
        )
        return splitter.create_documents([full_text], metadatas=[{"source": source_name, "split_method": "recursive"}])

    final_docs = []
    
    for chunk in chunks:
        clean_chunk = chunk.strip()
        
        # Çok kısa parçaları (sayfa no, çöp karakter) atla
        if len(clean_chunk) < 20:
            continue
            
        # --- METADATA ZEKASI ---
        # Chunk'ın hangi madde olduğunu bulup veritabanına etiket olarak ekleyelim.
        # Bu, ileride "Bana sadece Madde 5'i getir" dediğinde hayat kurtarır.
        madde_match = re.search(r"(MADDE\s+\d+)", clean_chunk)
        madde_tag = madde_match.group(1) if madde_match else "Giriş/Diğer"
        
        enriched_content = f" {source_name} |{madde_tag} \n---\n{clean_chunk}"
        
        final_docs.append(Document(
            page_content=enriched_content, # <--- Vektör artık bunu kullanacak!
            metadata={
                "source": source_name,
                "madde_no": madde_tag,
                "split_method": "regex_madde"
            }
        ))
        
    logger.info(f"✅ Başarılı: {len(final_docs)} adet madde tespit edildi.")
    return final_docs

def process_and_save_pdfs(reset_db=False):
    if reset_db:
        clear_database()
        
    data_path = "./data"
    all_final_documents = []

    if not os.path.exists(data_path):
        logger.error("Data klasörü yok!")
        return

    pdf_files = [f for f in os.listdir(data_path) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        file_path = os.path.join(data_path, pdf_file)
        logger.info(f"📂 Dosya Yükleniyor: {pdf_file}")
        
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            # ÖNEMLİ: Regex'in sayfa geçişlerinde çalışabilmesi için
            # tüm sayfaları tek bir dev metin (string) haline getiriyoruz.
            full_text = "\n".join([p.page_content for p in pages])
            
            # Regex Splitter'ı çağır
            file_docs = regex_madde_split(full_text, source_name=pdf_file)
            
            all_final_documents.extend(file_docs)
                
        except Exception as e:
            logger.error(f"{pdf_file} hata: {e}")

    if all_final_documents:
        db = get_chroma_client()
        
        # Batch Processing (Veri tabanı şişmesin diye 100'erli ekliyoruz)
        batch_limit = 100
        for i in range(0, len(all_final_documents), batch_limit):
            batch = all_final_documents[i : i + batch_limit]
            db.add_documents(batch)
            logger.info(f"💾 {len(batch)} kayıt veritabanına yazıldı...")
            
        logger.info(f"✅ TÜM İŞLEM TAMAM: Toplam {len(all_final_documents)} chunk hazır.")
    else:
        logger.warning("Veri yok.")
    
    if all_final_documents:
        # 1. ChromaDB (Vektörler) kaydediliyor
        db = get_chroma_client()
        db.add_documents(all_final_documents)
        
        # 2. BM25 Objesini Oluştur ve "Pişir"
        logger.info("🍳 BM25 indeksi hesaplanıyor ve donduruluyor...")
        bm25_retriever = BM25Retriever.from_documents(all_final_documents)
        
        # 3. Hazır objeyi diske yaz
        with open("./chromadb/bm25_retriever.pkl", "wb") as f:
            pickle.dump(bm25_retriever, f)
            
        logger.info("✅ BM25 'hazır paket' olarak kaydedildi.")

if __name__ == "__main__":
    process_and_save_pdfs(reset_db=True)