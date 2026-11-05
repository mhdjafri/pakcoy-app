import os
import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

# 1. KONTROL ENVIRONMENT (Wajib paling atas sebelum import TF/ST)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Monitoring Pakcoy", layout="wide")
st.title("🌱 Dashboard Realtime & Analisis AI Pakcoy")

BASE = Path(__file__).parent
DATA, MODELS = BASE / "data", BASE / "models"
DB = DATA / "app.db"
IMG = 224

# ---------------------------------------------------------------- Database & Init Data
def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    con = db()
    con.executescript("""
    CREATE TABLE IF NOT EXISTS sensor (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, dap INTEGER, jam INTEGER, moisture REAL, temp REAL);
    """)
    if con.execute("SELECT COUNT(*) FROM sensor").fetchone()[0] == 0:
        xlsx = next(DATA.glob("*.xlsx"), None)
        if xlsx is not None:
            d = pd.read_excel(xlsx)
            d["jam"] = d["Time"].astype(str).str[:2].astype(int)
            d = d.sort_values(["Day", "jam"])
            tgl_mulai = datetime(2026, 1, 1)
            rows = [((tgl_mulai + timedelta(days=int(day) - 1, hours=int(jam))).isoformat(),
                     int(dap), int(jam), float(kel), float(suhu))
                    for day, dap, jam, kel, suhu in zip(d["Day"], d["DAP"], d["jam"],
                                                        d["Soil Moisture (%)"], d["Temperature (°C)"])]
            con.executemany("INSERT INTO sensor (ts,dap,jam,moisture,temp) VALUES (?,?,?,?,?)", rows)
            con.commit()
    con.close()

init_db()

# ---------------------------------------------------------------- Caching Model AI
@st.cache_resource
def load_models():
    """Memuat semua konfigurasi dan model Keras sekali saja ke dalam cache"""
    tmp = {}
    tmp["cfg"] = json.load(open(MODELS / "config.json"))
    tmp["lstm"] = tf.keras.models.load_model(MODELS / "lstm_sensor.keras")
    tmp["cnn"] = tf.keras.models.load_model(MODELS / "cnn_tahap_pakcoy.keras")
    
    sc = json.load(open(MODELS / "scaler_sensor.json"))
    tmp["smin"] = np.array(sc["data_min"], dtype="float32")
    tmp["smax"] = np.array(sc["data_max"], dtype="float32")
    
    uk = pd.read_csv(MODELS / "ukuran_referensi.csv", sep=None, engine="python")
    uk.columns = [c.strip().lower() for c in uk.columns]
    for c in ("dap", "lebar_daun_cm", "tinggi_cm"):
        uk[c] = uk[c].astype(str).str.replace(",", ".", regex=False).astype(float)
    tmp["uk"] = uk
    tmp["k_lebar"] = np.poly1d(np.polyfit(uk["dap"], uk["lebar_daun_cm"], 2))
    tmp["k_tinggi"] = np.poly1d(np.polyfit(uk["dap"], uk["tinggi_cm"], 2))
    return tmp

try:
    m = load_models()
except Exception as e:
    st.error(f"❌ Gagal memuat model AI: {e}")
    st.stop()

# ---------------------------------------------------------------- State Kontrol Replay
if "posisi_kursor" not in st.session_state:
    st.session_state.posisi_kursor = 40

con = db()
total_data = con.execute("SELECT COUNT(*) FROM sensor").fetchone()[0]
con.close()

# Otomatis maju 1 indeks setiap halaman memuat ulang (Simulasi Live)
if st.session_state.posisi_kursor < total_data:
    st.session_state.posisi_kursor += 1
else:
    st.session_state.posisi_kursor = 40 

def ambil_data_live(posisi, limit=40):
    con = db()
    rows = con.execute("SELECT * FROM sensor WHERE id <= ? ORDER BY id DESC LIMIT ?", (posisi, limit)).fetchall()
    con.close()
    return pd.DataFrame([dict(r) for r in reversed(rows)])

df_live = ambil_data_live(st.session_state.posisi_kursor)

# ---------------------------------------------------------------- ANTARMUKA TAB STREAMLIT
tab1, tab2 = st.tabs(["📊 Realtime Telemetry & Forecasting", "📷 Analisis Citra CNN"])

with tab1:
    if not df_live.empty:
        baris_terakhir = df_live.iloc[-1]
        
        # Ringkasan KPI Atas
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Kelembaban Tanah (Moisture)", f"{baris_terakhir['moisture']}%")
        kpi2.metric("Suhu Lingkungan", f"{baris_terakhir['temp']} °C")
        kpi3.metric("Usia Tanaman (DAP)", f"{int(baris_terakhir['dap'])} Hari")

        # Grafik Realtime
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.subheader("💧 Tren Kelembaban Tanah")
            st.line_chart(df_live.set_index('ts')[['moisture']], height=280)
        with col_g2:
            st.subheader("🌡️ Tren Suhu Lingkungan")
            st.line_chart(df_live.set_index('ts')[['temp']], height=280, color="#FF4B4B")
            
        # Estimasi Ukuran Berdasarkan Usia (DAP) saat ini
        st.subheader("📐 Estimasi Dimensi Pakcoy Saat Ini")
        uk_ref = m["uk"]
        d = float(np.clip(baris_terakhir['dap'], uk_ref["dap"].min(), uk_ref["dap"].max()))
        lebar_est = round(float(m["k_lebar"](d)), 1)
        tinggi_est = round(float(m["k_tinggi"](d)), 1)
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Proyeksi Lebar Daun", f"{lebar_est} cm")
        col_m2.metric("Proyeksi Tinggi Batang", f"{tinggi_est} cm")
    else:
        st.warning("Menunggu data sensor masuk...")

with tab2:
    st.header("Upload Citra untuk Klasifikasi CNN")
    file_gambar = st.file_uploader("Pilih file gambar pakcoy...", type=["jpg", "jpeg", "png"])
    
    if file_gambar is not None:
        image = Image.open(file_gambar)
        st.image(image, caption="Gambar yang berhasil diunggah", width=350)
        
        with st.spinner("Model CNN sedang menganalisis gambar..."):
            # Pemrosesan gambar & Prediksi CNN
            im = ImageOps.exif_transpose(image).convert("RGB").resize((IMG, IMG))
            arr_img = np.array(im, dtype="float32")[None]
            pr = m["cnn"].predict(arr_img, verbose=0)[0]
            
            kelas_prediksi = m["cfg"]["kelas"][int(pr.argmax())]
            keyakinan = float(pr.max()) * 100
        
        # Tampilkan Hasil Analisis Gambar
        res1, res2 = st.columns(2)
        res1.success(f"**Hasil Klasifikasi Tahap:** {kelas_prediksi}")
        res2.info(f"**Tingkat Keyakinan (Confidence):** {keyakinan:.2f}%")

# ---------------------------------------------------------------- Detak Refresh Otomatis
st.caption(f"Status: Mode Replay Aktif (Indeks: {st.session_state.posisi_kursor}/{total_data}). Halaman me-refresh otomatis tiap 5 detik.")
time.sleep(5)
st.rerun()
