import streamlit as st
import pandas as pd
import time
from pymongo import MongoClient
import plotly.express as px

# 1. Sayfa Ayarları
st.set_page_config(
    page_title="Data Analytics Dashboard",
    layout="wide",
    page_icon="📊"
)

# 2. Veritabanı Bağlantısı
@st.cache_resource
def init_connection():
    return MongoClient("mongodb://mongodb:27017/")

try:
    client = init_connection()
    db = client['sales_db']
except Exception as e:
    st.error(f"Veritabanına bağlanılamadı: {e}")
    st.stop()

# 3. Başlık
st.title("📊 Gerçek Zamanlı Satış Analitiği Sistemi")
st.markdown("""
Bu dashboard, **Kafka**, **Spark Streaming** ve **MongoDB** kullanılarak oluşturulan veri boru hattının (pipeline) canlı sonuçlarını gösterir.
Model, mağaza ve kategori bazlı **Satış Adedi (Unit Sales)** tahmini yapmaktadır.
* **Akış:** `Producer` -> `Kafka` -> `Spark ML` -> `MongoDB` -> `Dashboard`
""")
st.divider()

# 4. Yer Tutucular
col1, col2 = st.columns([1, 3])

with col1:
    metric_ph = st.empty()
    status_ph = st.empty()

with col2:
    chart_ph = st.empty()

st.subheader("📥 Gelen Son Ham Veriler (Kafka Stream)")
raw_data_ph = st.empty()

# 5. Ana Döngü
while True:
    try:
        # A) Verileri Çek
        predictions = list(db.predictions.find().sort('_id', -1).limit(100))
        raw_data = list(db.raw_data.find().sort('_id', -1).limit(10))

        # --- GÖRSELLEŞTİRME ---
        if predictions:
            df = pd.DataFrame(predictions)
            df = df.iloc[::-1] # Eskiden yeniye sırala

            # 1. Metrik Güncelleme
            latest_pred = df.iloc[-1]['predicted_sales']
            latest_family = df.iloc[-1]['family']
            
            is_promo = "🔥 İndirim Var" if df.iloc[-1]['onpromotion'] == 1 else "Standart Fiyat"

            metric_ph.metric(
                label=f"Son Tahmin ({latest_family})", 
                value=f"{latest_pred:.0f} Adet", 
                delta=is_promo
            )
            
            # 2. Grafik Güncelleme (HATA DÜZELTİLDİ)
            # 'x' karmaşasını önlemek için 'sira' adında gerçek bir sütun ekliyoruz
            df['sira'] = range(len(df))

            fig = px.line(
                df, 
                x='sira',  # Artık doğrudan sütun adını kullanıyoruz, hata çıkmaz
                y='predicted_sales', 
                title='Son 100 İşlemin Satış Miktarı Tahmini',
                labels={'sira': 'Akış Sırası', 'predicted_sales': 'Tahmini Satış (Adet)'},
                # MOUSE İLE ÜZERİNE GELİNCE GÖRÜNECEKLER
                hover_data={
                    'sira': False, # Sıra numarasını gizle, gerek yok
                    'family': True,
                    'store_nbr': True,
                    'onpromotion': True,
                    'predicted_sales': ':.0f'
                }
            )
            
            fig.update_traces(mode="lines+markers")
            
            # Benzersiz ID (Hatayı önler)
            unique_key = f"chart_{time.time()}"
            chart_ph.plotly_chart(fig, use_container_width=True, key=unique_key)
            
            status_ph.success("✅ Sistem Çalışıyor: ML Modeli Satış Adedi Tahmin Ediyor.")
        else:
            status_ph.warning("⏳ Veri bekleniyor... (Spark henüz veri yazmadı)")

        # 3. Tablo
        if raw_data:
            raw_df = pd.DataFrame(raw_data)
            if '_id' in raw_df.columns:
                raw_df = raw_df.drop('_id', axis=1)
            
            with raw_data_ph.container():
                st.dataframe(raw_df, hide_index=True, use_container_width=True)
        
    except Exception as e:
        status_ph.error(f"Bir hata oluştu: {e}")

    time.sleep(1)