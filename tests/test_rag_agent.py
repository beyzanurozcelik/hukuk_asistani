import os
from src.agents.rag_agent import rag_agent

def run_test(question: str):
    print(f"\n" + "="*50)
    print(f"🤔 KULLANICI SORUSU: {question}")
    print("="*50)
    
    # LangGraph akışını başlatıyoruz
    inputs = {"question": question}
    
    try:
        # Ajanın tüm düğümlerden geçişini ve nihai sonucunu alıyoruz
        result = rag_agent.invoke(inputs)
        
        print(f"\n🧠 ANALİZER KARARI: {result['decision']}")
        print(f"🔎 OPTİMİZE SORGU: {result['search_query']}")
        print("\n--------------------------------------------------")
        print(f"🤖 ASİSTAN CEVABI:\n{result['response']}")
        print("--------------------------------------------------")
        
    except Exception as e:
        print(f"❌ TEST HATASI: {e}")

if __name__ == "__main__":
    # SENARYO 1: Nokta Atışı (Q1) - Spesifik Madde Sorusu
    run_test("kvkk madde 11 nedir")
    
    # SENARYO 2: Geniş Arama (Q2) - Süreç ve Sentez Sorusu
    #run_test("Kişisel verilerin silinmesi ve yok edilmesi konusunda birden fazla döküman ne diyor")
    #run_test("Kişisel verilerin silinmesi belgesinde verilerin yok edilmesiyle ilgili ne diyor?")

    #run_test("kvkkya göre açık rıza nedir?")
    #run_test("belgelere göre açık rıza nedir?")