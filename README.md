# ⚖️ Türkçe Hukuk Asistanı (Agentic RAG)

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-1C3C3C?style=for-the-badge&logo=langchain)](https://www.langchain.com/)
[![Hugging Face](https://img.shields.io/badge/Embeddings-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface)](https://huggingface.co/)
[![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-8E75B2?style=for-the-badge&logo=google-gemini&logoColor=white)](https://deepmind.google/technologies/gemini/)

> **"Sadece arama yapmaz; sorunun türünü anlar, strateji belirler, cevabı oluşturur."**

Bu proje, Türkçe hukuki dokümanlar üzerinde çalışan, **Multi-Agent (Çoklu Ajan)** mimarisine sahip, bir **Agentic RAG** sistemidir. Standart "Soru-Cevap" botlarının ötesine geçerek, bir hukuk danışmanının muhakeme süreçlerini simüle etmeyi hedefler.

## Proje Hedefi, Kapsam ve Senaryolar

Geleneksel RAG sistemleri genellikle "tek tip" bir yaklaşım sergiler: Soruyu al, vektör veritabanında ara, bulduğunu LLM'e ver. Ancak hukuki süreçler bu kadar doğrusal değildir. Bu proje, **kullanıcı niyetini (user intent)** temel alarak dinamik bir akış sunar.

Sistem, aşağıdaki üç temel senaryoyu birbirinden ayırır ve her biri için optimize edilmiş farklı bir alt akış (sub-graph) çalıştırır:

### 1. Tekil Bilgi ve Tanım Soruları (Q1 - Precision Focus)
* **Senaryo:** Kullanıcı spesifik bir maddenin tanımını veya tek bir dokümanda geçen net bir bilgiyi sorar.
* **Örnek:** *"KVKK'ya göre 'Veri Sorumlusu' kimdir?"* veya *"Sözleşmenin 4. maddesindeki fesih süresi nedir?"*
* **Strateji:** Sistem, geniş bir okuma yapmak yerine "Nokta Atışı" (Needle in a haystack) stratejisini uygular. Hedef, en yüksek benzerlik skoruna sahip 1-2 paragrafı bulmaktır.

### 2. Sentez ve Karşılaştırma Soruları (Q2 - Recall Focus)
* **Senaryo:** Kullanıcı, birden fazla dokümanın taranmasını, bilgilerin birleştirilmesini veya karşılaştırılmasını gerektiren kompleks sorular sorar.
* **Örnek:** *"Bu konuda İş Kanunu ve Borçlar Kanunu arasındaki farklar nelerdir?"* veya *"Elimizdeki tüm sözleşmelerde 'Mücbir Sebep' maddesi ne şekilde tanımlanmıştır?"*
* **Strateji:** Sistem "Geniş Arama" moduna geçer. Daha fazla doküman parçası (chunk) getirilir, gerekirse dokümanlar arası bağlam korunarak bir sentez (synthesis) yanıtı oluşturulur.

### 3. Genel Sohbet ve Özetleme (Q3 - Efficiency Focus)
* **Senaryo:** Kullanıcı dokümanlardan bağımsız bir soru sorabilir, selamlaşabilir veya mevcut doküman setinin genel bir özetini isteyebilir.
* **Örnek:** *"Merhaba, nasılsın?"* veya *"Yüklenen dokümanların genel konusu nedir?"*
* **Strateji:** Vektör veritabanında maliyetli ve gereksiz bir arama yapılmaz. Sistem doğrudan LLM'in kendi bilgi birikimini veya dokümanların önceden hazırlanmış meta-özetlerini kullanır.

---

## Mimari Detayları ve Ajan Yapısı

Proje, **LangGraph** kütüphanesi kullanılarak bir **"State Machine" (Durum Makinesi)** olarak kurgulanmıştır. Bu yapı, ajanların birbirine iş devretmesine, durum (state) paylaşmasına ve döngüsel (cyclic) işlemler yapmasına olanak tanır.

Mimarideki temel bileşenler şunlardır:

### 1. Supervisor Agent (Yönetici & Router)
Sistemin giriş kapısıdır. Gelen soruyu semantik olarak analiz eder ve bir sınıflandırma (classification) yapar. Bu ajan bir cevap üretmez, sadece trafiği yönlendirir.
* **Görevi:** Sorunun Q1, Q2 veya Q3 kategorisine girdiğini belirlemek.
* **Karar Mekanizması:** LLM'e sunulan özel bir prompt ile sorunun niyetini (Intent Detection) tespit eder.

### 2. Analyzer Agent (Analist & Stratejist)
Doküman analizi gerektiğinde devreye girer. Sadece arama yapmaz, "nasıl arama yapılacağını" planlar.
* **Query Expansion (Sorgu Genişletme):** Kullanıcının sorusunu, veritabanında daha iyi sonuç verecek hukuki terimlerle yeniden yazar veya alternatif sorgular üretir.
* **Tool Seçimi:** Sorunun derinliğine göre aşağıdaki araçlardan hangisinin kullanılacağına karar verir:
    * **🎯 Nokta Atışı Aracı (Point Search Tool):** `top_k=3` gibi dar bir pencerede yüksek kesinlikli arama yapar.
    * **🌐 Geniş Arama Aracı (Broad Search Tool):** `top_k=10` veya üzeri geniş bir pencerede arama yapar ve gerekirse MMR (Maximal Marginal Relevance) algoritması ile çeşitliliği artırır.

### 3. Kalite Kontrol (Grader & Self-Correction Loop)
Sistemin "Zekası" buradadır. Standart RAG sistemlerinde olmayan "Oto-Kontrol" mekanizmasını işletir.
* **Relevance Check (Alaka Kontrolü):** Araçlardan dönen doküman parçalarının, kullanıcının sorusuyla gerçekten alakalı olup olmadığını puanlar.
* **Hallucination Check (Halüsinasyon Kontrolü):** Üretilen cevabın, sadece ve sadece bulunan dokümanlara dayanıp dayanmadığını kontrol eder.
* **Döngü (Loop) Mekanizması:** Eğer Grader, bulunan dokümanları yetersiz bulursa veya cevabın uydurma olduğunu tespit ederse akışı sonlandırmaz. **Analyzer Agent**'a geri bildirim (feedback) göndererek: *"Bulduğun dokümanlar soruyla alakasız, lütfen sorguyu değiştir ve tekrar ara"* komutunu verir. Bu döngü, doğru bilgi bulunana veya deneme hakkı bitene kadar devam eder.

---

## İnteraktif Akış Şeması (Mermaid)

Aşağıdaki diyagram, sistemin karar ağaçlarını, ajanlar arası geçişleri ve hata durumunda devreye giren geri bildirim döngülerini detaylıca göstermektedir:

```mermaid
graph TD
    %% --- Node Tanımları ---
    User(["👤 Kullanıcı Sorusu"])
    Supervisor{"🚦 Supervisor Agent"}
    SummaryTool["📝 Genel Özet Aracı"]
    End(["🚀 Nihai Cevap"])

    %% --- RAG Agent Grubu ---
    subgraph AE ["🚀 Rag Agent: RAG VE CEVAP ÜRETİMİ"]
        direction TB
        AnalystNode("🤖 Analizer")
        SearchRouter{"❓ Soru Tipi?"}
        PointSearch["🎯 Nokta Atışı Aracı"]
        BroadSearch["🌐 Geniş Arama Aracı"]
        Control{"🧐 Kalite Kontrol"}
        FinalWrite["✍️ Cevap Üretimi"]
    end

    %% --- Bağlantılar ---
    User --> Supervisor
    Supervisor -- Doküman Analizi --> AnalystNode
    Supervisor -- Özet Gerekli (Q3) --> SummaryTool

    AnalystNode --> SearchRouter
    SearchRouter -- Tekil Bilgi (Q1) --> PointSearch
    SearchRouter -- Çoklu Doküman (Q2) --> BroadSearch

    PointSearch --> Control
    BroadSearch --> Control

    Control -- Tamam --> FinalWrite

    SummaryTool --> End
    FinalWrite --> End

    %% --- Stil Tanımları (GitHub Standartlarına Uygun) ---
    classDef startStop fill:#2d3436,stroke:#000,stroke-width:2px,color:#fff
    classDef supervisor fill:#ff9f43,stroke:#e67e22,stroke-width:3px,color:#000
    classDef analyst fill:#0984e3,stroke:#074e83,stroke-width:2px,color:#fff
    classDef tool fill:#f5f6fa,stroke:#7f8c8d,stroke-width:2px,stroke-dasharray: 5 5,color:#000
    classDef decision fill:#fdcb6e,stroke:#f39c12,stroke-width:2px,color:#000
    classDef container fill:#E3F2FD,stroke:#2980b9,stroke-width:2px,color:#000

    %% --- Stilleri Uygula ---
    class User,End startStop
    class Supervisor supervisor
    class AnalystNode analyst
    class SearchRouter,Control decision
    class PointSearch,BroadSearch,SummaryTool tool
    class AE container

## 🚀 Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda (lokal ortamda) çalıştırmak için aşağıdaki adımları sırasıyla izleyin.

### Ön Hazırlık
* Bilgisayarınızda **Python 3.13** veya üzeri yüklü olmalıdır.
* Google AI Studio üzerinden alınmış bir **Gemini API Key** gereklidir.

### 1. Repoyu Klonlayın
Projeyi bilgisayarınıza indirin ve proje dizinine girin:

```bash
git clone [https://github.com/beyzanurozcelik/hukuk_asistani.git]
cd hukuk_asistani

### 2. Sanal Ortam Oluşturun
python -m venv venv
.\venv\Scripts\activate

### 3. Kütüphaneleri Yükleyin
pip install -r requirements.txt

### 4. .env dosyası oluşturun
GOOGLE_API_KEY ve GEMINI_MODEL_NAME içermeli.

### 5. FastAPI appini çalıştırmak için
Komutu çalıştırın: uvicorn main:app --reload

### 6. Streamlit Arayüzünü Çalıştırmak için
Komutu çalıştırın: streamlit run frontend.py

