import sys
import os
import time

# Proje ana dizinini path'e ekliyoruz
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from agents.summarize_node import summarize_node
except ImportError as e:
    print("❌ HATA: Modül bulunamadı. Lütfen dosya yollarını kontrol edin.")
    print(f"Detay: {e}")
    sys.exit(1)

def run_test():
    print("📋 SUMMARIZER NODE TESTİ (Model: Gemma 3 12B)\n")
    print("⚠️  ÖNEMLİ: Bu testin çalışması için 'chromadb_summaries' klasörünün dolu olması gerekir.")
    print("-" * 60)

    # Test Senaryoları
    test_queries = [
        "Elimizdeki belgeleri genel olarak özetle.",   # Genel tarama
        "KVKK hakkında özet bilgi ver.",               # Spesifik dosya hedefli (Similarity Search çalışmalı)
        "Veri güvenliği ile ilgili neler var?"         # Konu bazlı arama
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n🧪 TEST {i}: '{query}'")
        
        start_time = time.time()
        
        # Node sadece bir 'state' sözlüğü bekler
        mock_state = {"question": query}
        
        try:
            print("⏳ Özetler taranıyor ve sentezleniyor...")
            
            # Düğümü çalıştırıyoruz
            result = summarize_node(mock_state)
            
            elapsed = time.time() - start_time
            
            # Çıktıyı göster
            response = result.get("response", "Cevap yok")
            
            print(f"\n📝 GEMMA 3 CEVABI:\n{'-'*20}")
            print(f"\033[96m{response}\033[0m") # Cyan rengiyle yazdıralım
            print(f"{'-'*20}")
            print(f"⏱️  Süre: {elapsed:.2f} sn")
            
        except Exception as e:
            print(f"❌ HATA: {e}")
            
        print("-" * 60)

if __name__ == "__main__":
    run_test()