import sys
import os
import time

# Proje ana dizinini path'e ekleyelim ki 'src' modülünü bulabilsin
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from src.agents.supervisor_agent import supervisor_agent
except ImportError as e:
    print("❌ HATA: Modül bulunamadı. Lütfen bu dosyayı projenin ana dizininde çalıştırın.")
    print(f"Detay: {e}")
    sys.exit(1)

def run_test():
    print("🚀 SUPERVISOR AGENT TESTİ BAŞLIYOR (Model: Gemma 3 12B)\n")
    print("-" * 60)

    # Test Senaryoları: Hem Özet (Q3) hem Analiz (RAG) hem de Belirsiz durumlar
    test_scenarios1 = [
        # SENARYO 1: Açıkça Özet İsteyenler (Beklenen: Q3)
        "Elimizdeki belgeleri kısaca özetle.",
        "KVKK kanununda genel olarak neler var?",
        "Bana bir genel bakış sun.",
        
        # SENARYO 2: Spesifik Analiz İsteyenler (Beklenen: RAG)
        "Madde 11 kapsamında ilgili kişinin hakları nelerdir?",
        "Açık rıza aranmayan haller hangileridir?",
        "Veri sorumlusunun teknik yükümlülükleri hakkında analiz yap.",
        
        # SENARYO 3: Belirsiz / Kısa / Selamlaşma (Beklenen: Q3 - Default Kuralı)
        "Selam",
        "Merhaba kolay gelsin",
        "KVKK nedir?",  # Sadece konu başlığı
    ]

    test_scenarios = [
        "kvkk madde 11i özetler misin",
        "Veri güvenliği ile ilgili neler var?"         # Konu bazlı arama

    ]

    for i, query in enumerate(test_scenarios, 1):
        print(f"\n🧪 TEST {i}: '{query}'")
        
        start_time = time.time()
        
        # State sözlüğü oluşturup ajanı tetikliyoruz
        # Not: LangGraph state yapısı dict olarak da verilebilir.
        initial_state = {"question": query, "next_step": "", "response": ""}
        
        try:
            result = supervisor_agent.invoke(initial_state)
            
            # Sonucu al
            decision = result.get("next_step", "HATA")
            elapsed = time.time() - start_time
            
            # Görselleştirme
            if decision == "Q3":
                print(f"👉 KARAR: \033[94m{decision} (ÖZET/GENEL)\033[0m") # Mavi
            elif decision == "RAG":
                print(f"👉 KARAR: \033[92m{decision} (ANALİZ/DETAY)\033[0m") # Yeşil
            else:
                print(f"👉 KARAR: {decision}")
                
            print(f"⏱️  Süre: {elapsed:.2f} sn")
            
        except Exception as e:
            print(f"❌ HATA OLUŞTU: {e}")
            
        print("-" * 60)

if __name__ == "__main__":
    run_test()