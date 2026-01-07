import streamlit as st
import requests
import time

# --- AYARLAR ---
API_URL = "http://127.0.0.1:8000/chat"
st.set_page_config(
    page_title="Hukuk Asistanı",
    page_icon="⚖️",
    layout="centered"
)

# --- BAŞLIK VE AÇIKLAMA ---
st.title("⚖️ AI Hukuk Asistanı")
st.markdown("Supervisor mimarisi ile çalışan **RAG** ve **Özetleme** asistanı.")

# --- CSS STİLLERİ (Opsiyonel: Görünümü Güzelleştirme) ---
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
    }
    .info-box {
        font-size: 0.8rem;
        color: #666;
        background-color: #f0f2f6;
        padding: 5px 10px;
        border-radius: 5px;
        margin-top: 5px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE (Sohbet Geçmişi) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- GEÇMİŞ MESAJLARI EKRANA YAZDIR ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Eğer varsa teknik detayları (meta-data) da gösterelim
        if "metadata" in message:
            meta = message["metadata"]
            st.markdown(
                f"""
                <div class='info-box'>
                🧭 <b>Rota:</b> {meta.get('route')} | 
                ⏱️ <b>Süre:</b> {meta.get('time')} sn
                </div>
                """, 
                unsafe_allow_html=True
            )

# --- KULLANICI GİRDİSİ ---
if prompt := st.chat_input("Hukuki sorunuzu buraya yazın..."):
    
    # 1. Kullanıcı mesajını ekrana bas ve geçmişe ekle
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Asistanın cevabını bekle
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # "Düşünüyor..." efekti
        with st.status("🔍 Sistem analiz yapıyor...", expanded=True) as status:
            try:
                start_time = time.time()
                
                # API'ye istek at
                response = requests.post(API_URL, json={"question": prompt})
                
                if response.status_code == 200:
                    data = response.json()
                    
                    answer = data.get("response", "Cevap yok.")
                    route = data.get("route", "Bilinmiyor")
                    rag_decision = data.get("rag_decision")
                    elapsed = data.get("elapsed_time", 0)
                    
                    # Durum çubuğunu güncelle
                    status.update(label=f"✅ İşlem Tamamlandı (Rota: {route})", state="complete", expanded=False)
                    
                    # Cevabı yazdır
                    message_placeholder.markdown(answer)
                    
                    # Altına teknik bilgi kutucuğu ekle
                    detail_text = f"🧭 **Rota:** `{route}`"
                    if rag_decision:
                        detail_text += f" | 🔍 **Analiz:** `{rag_decision}`"
                    detail_text += f" | ⏱️ **Süre:** `{elapsed} sn`"
                    
                    st.caption(detail_text)
                    
                    # 3. Asistan mesajını geçmişe kaydet
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "metadata": {"route": route, "time": elapsed}
                    })
                    
                else:
                    status.update(label="❌ Hata oluştu", state="error")
                    error_msg = f"API Hatası: {response.status_code}"
                    message_placeholder.error(error_msg)
            
            except Exception as e:
                status.update(label="❌ Bağlantı Hatası", state="error")
                message_placeholder.error(f"Backend'e bağlanılamadı. API çalışıyor mu? \n\nHata: {e}")