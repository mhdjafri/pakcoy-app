import os
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

# 1. WAJIB DI BARIS PALING ATAS (Sebelum import TF dan ST)
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
st.title("🌱 Aplikasi Monitoring & Prediksi Pakcoy")

BASE = Path(__file__).parent
MODELS, DATA = BASE / "models", BASE / "data"
DB = DATA / "app.db"
IMG = 224

# ---------------------------------------------------------------- Database & Model
def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con

@st.cache_resource
def load_models():
    """Memuat model sekali saja dan disimpan di cache Streamlit"""
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

# Load model secara aman
try:
    m = load_models()
    st.success("✅ Semua model AI berhasil dimuat!")
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# ---------------------------------------------------------------- UI Aplikasi Streamlit
tab1, tab2 = st.tabs(["📊 Data Sensor & Prediksi", "📷 Analisis Citra Pakcoy"])

with tab1:
    st.header("Monitoring Sensor")
    # Tampilkan input manual atau ambil dari database SQLite Anda
    dap_input = st.number_input("Masukkan Usia Tanaman (DAP):", min_value=1, max_value=60, value=10)
    
    # Estimasi Ukuran
    uk = m["uk"]
    d = float(np.clip(dap_input, uk["dap"].min(), uk["dap"].max()))
    lebar = round(float(m["k_lebar"](d)), 1)
    tinggi = round(float(m["k_tinggi"](d)), 1)
    
    col1, col2 = st.columns(2)
    col1.metric("Estimasi Lebar Daun", f"{lebar} cm")
    col2.metric("Estimasi Tinggi Tanaman", f"{tinggi} cm")

with tab2:
    st.header("Analisis Foto Pakcoy")
    file_gambar = st.file_uploader("Unggah foto pakcoy Anda", type=["jpg", "jpeg", "png"])
    
    if file_gambar is not None:
        image = Image.open(file_gambar)
        st.image(image, caption="Gambar yang diunggah", use_container_width=True)
        
        # Klasifikasi Gambar via CNN
        im = ImageOps.exif_transpose(image).convert("RGB").resize((IMG, IMG))
        pr = m["cnn"].predict(np.array(im, dtype="float32")[None], verbose=0)[0]
        kelas_prediksi = m["cfg"]["kelas"][int(pr.argmax())]
        keyakinan = float(pr.max()) * 100
        
        st.metric("Hasil Tahap CNN", kelas_prediksi)
        st.metric("Tingkat Keyakinan AI", f"{keyakinan:.2f}%")
