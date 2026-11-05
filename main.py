import os
import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

# 1. KONTROL ENVIRONMENT (Wajib paling atas)
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
st.title("🌱 Dashboard Realtime & Prediksi Pakcoy")

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
    """Memastikan database terisi dari file Excel jika kosong"""
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

# ---------------------------------------------------------------- State Kontrol Replay
# Menyimpan posisi index data simulasi saat ini di memory Streamlit
if "posisi_kursor" not in st.session_state:
    st.session_state.posisi_kursor = 40  # Mulai menampilkan dari 40 data awal

# Dapatkan total baris data yang tersedia
con = db()
total_data = con.execute("SELECT COUNT(*) FROM sensor").fetchone()[0]
con.close()

# Otomatis maju 1 baris setiap halaman dimuat ulang (Simulasi Realtime)
if st.session_state.posisi_kursor < total_data:
    st.session_state.posisi_kursor += 1
else:
    st.session_state.posisi_kursor = 40  # Reset ke awal jika data habis (Looping)

# Ambil data terbaru batas index saat ini
def ambil_data_live(posisi, limit=40):
    con = db()
    rows = con.execute("SELECT * FROM sensor WHERE id <= ? ORDER BY id DESC LIMIT ?", (posisi, limit)).fetchall()
    con.close()
    return pd.DataFrame([dict(r) for r in reversed(rows)])

df_live = ambil_data_live(st.session_state.posisi_kursor)

# ---------------------------------------------------------------- Tampilan Widget Utama
if not df_live.empty:
    baris_terakhir = df_live.iloc[-1]
    
    # Ringkasan Angka Indikator Atas
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Kelembaban Tanah (Moisture)", f"{baris_terakhir['moisture']}%")
    kpi2.metric("Suhu Lingkungan", f"{baris_terakhir['temp']} °C")
    kpi3.metric("Usia Tanaman (DAP)", f"{int(baris_terakhir['dap'])} Hari")

    # Layout Grafik Realtime Bergerak
    col_grafik_moisture, col_grafik_temp = st.columns(2)
    
    with col_grafik_moisture:
        st.subheader("💧 Grafik Kelembaban Tanah Realtime")
        # Menggunakan kolom 'ts' atau indeks sebagai penanda waktu X
        st.line_chart(df_live.set_index('ts')[['moisture']], height=300)

    with col_grafik_temp:
        st.subheader("🌡️ Grafik Suhu Lingkungan Realtime")
        st.line_chart(df_live.set_index('ts')[['temp']], height=300, color="#FF4B4B")
else:
    st.warning("Data sensor tidak ditemukan di database.")

# ---------------------------------------------------------------- Kontrol Interval Replay
# Teks info debug posisi simulasi
st.caption(f"Simulasi Replay aktif: Menampilkan data ke-{st.session_state.posisi_kursor} dari total {total_data} data sensor. Diperbarui otomatis tiap 5 detik.")

# Delay 5 detik lalu picu Streamlit untuk me-refresh halaman guna memajukan data
time.sleep(5)
st.rerun()
