"""
Backend Monitoring Pakcoy (FastAPI)
Jalankan:  uvicorn main:app --reload      lalu buka http://127.0.0.1:8000
Mode DEMO (default): data sensor dari Excel diputar ulang satu pembacaan per permintaan.
Mode sensor asli   : set DEMO_MODE=0 dan kirim data dengan POST /api/sensor
"""
import os, json, sqlite3, threading
from datetime import datetime, timedelta
from pathlib import Path

# Menyembunyikan GPU dari TensorFlow dan menonaktifkan optimasi XLA CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

import streamlit as st
import tensorflow as tf  # Impor TF ditaruh setelah os.environ


import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image, ImageOps

BASE = Path(__file__).parent
MODELS, DATA, UPLOADS, STATIC = BASE / "models", BASE / "data", BASE / "uploads", BASE / "static"
DB = DATA / "app.db"
DEMO = os.getenv("DEMO_MODE", "1") == "1"
TANGGAL_MULAI_DATA = datetime(2026, 1, 1)      # samakan dengan Cell 3 di Colab
IMG = 224
lock = threading.Lock()

app = FastAPI(title="Monitoring Pakcoy")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOADS), name="uploads")

# ---------------------------------------------------------------- database
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
    CREATE TABLE IF NOT EXISTS riwayat (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        waktu TEXT, dap INTEGER, tahap_cnn TEXT, keyakinan REAL,
        suhu_terakhir REAL, kelembaban_terakhir REAL,
        lebar_daun_cm REAL, tinggi_cm REAL, prediksi_json TEXT, nama_gambar TEXT);
    """)
    if con.execute("SELECT COUNT(*) FROM sensor").fetchone()[0] == 0:
        xlsx = next(DATA.glob("*.xlsx"), None)
        if xlsx is not None:
            d = pd.read_excel(xlsx)
            d["jam"] = d["Time"].astype(str).str[:2].astype(int)
            d = d.sort_values(["Day", "jam"])
            rows = [((TANGGAL_MULAI_DATA + timedelta(days=int(day) - 1, hours=int(jam))).isoformat(),
                     int(dap), int(jam), float(kel), float(suhu))
                    for day, dap, jam, kel, suhu in zip(d["Day"], d["DAP"], d["jam"],
                                                        d["Soil Moisture (%)"], d["Temperature (°C)"])]
            con.executemany("INSERT INTO sensor (ts,dap,jam,moisture,temp) VALUES (?,?,?,?,?)", rows)
            con.commit()
    con.close()

init_db()
cursor = {"pos": 60}      # posisi pemutaran ulang (mode demo)

# ---------------------------------------------------------------- model (dimuat saat pertama dipakai)
_M = {}
_load_lock = threading.Lock()

def _baca_ukuran():
    uk = pd.read_csv(MODELS / "ukuran_referensi.csv", sep=None, engine="python")   # aman untuk ; atau ,
    uk.columns = [c.strip().lower() for c in uk.columns]
    for c in ("dap", "lebar_daun_cm", "tinggi_cm"):
        uk[c] = uk[c].astype(str).str.replace(",", ".", regex=False).astype(float)
    return uk

def models():
    """Muat semua model sekali. Hanya dianggap berhasil kalau SEMUA langkah sukses."""
    if "ok" not in _M:
        with _load_lock:
            if "ok" not in _M:
                import tensorflow as tf
                tmp = {}
                tmp["cfg"] = json.load(open(MODELS / "config.json"))
                tmp["lstm"] = tf.keras.models.load_model(MODELS / "lstm_sensor.keras")
                tmp["cnn"] = tf.keras.models.load_model(MODELS / "cnn_tahap_pakcoy.keras")
                sc = json.load(open(MODELS / "scaler_sensor.json"))
                tmp["smin"] = np.array(sc["data_min"], dtype="float32")
                tmp["smax"] = np.array(sc["data_max"], dtype="float32")
                uk = _baca_ukuran()
                tmp["uk"] = uk
                tmp["k_lebar"] = np.poly1d(np.polyfit(uk["dap"], uk["lebar_daun_cm"], 2))
                tmp["k_tinggi"] = np.poly1d(np.polyfit(uk["dap"], uk["tinggi_cm"], 2))
                tmp["ok"] = True
                _M.update(tmp)
    return _M

@app.on_event("startup")
def muat_model_saat_start():
    """Muat model saat server menyala supaya error (jika ada) langsung tampil di terminal."""
    try:
        models()
        print(">>> Model berhasil dimuat.")
    except Exception:
        import traceback
        print(">>> GAGAL MEMUAT MODEL. Error asli:")
        traceback.print_exc()

# ---------------------------------------------------------------- fungsi inti
def baris_terbaru(n):
    """n pembacaan terakhir (mode demo: sampai posisi kursor)."""
    con = db()
    if DEMO:
        rows = con.execute("SELECT * FROM sensor WHERE id<=? ORDER BY id DESC LIMIT ?",
                           (cursor["pos"], n)).fetchall()
    else:
        rows = con.execute("SELECT * FROM sensor ORDER BY id DESC LIMIT ?", (n,)).fetchall()
    con.close()
    return [dict(r) for r in reversed(rows)]

def prediksi_4_hari():
    m = models()
    cfg = m["cfg"]; L, H = cfg["lookback"], cfg["horizon"]
    rows = baris_terbaru(L)
    if len(rows) < L:
        raise HTTPException(400, f"Butuh minimal {L} pembacaan sensor, baru ada {len(rows)}")
    d = pd.DataFrame(rows)
    d["jam_sin"] = np.sin(2 * np.pi * d["jam"] / 24)
    d["jam_cos"] = np.cos(2 * np.pi * d["jam"] / 24)
    smin, smax = m["smin"], m["smax"]
    arr = (d[cfg["fitur"]].values.astype("float32") - smin) / (smax - smin)
    with lock:
        p = m["lstm"].predict(arr[None], verbose=0)[0]
    p = p * (smax[:2] - smin[:2]) + smin[:2]
    return [{"hari_ke": i // 3 + 1, "jam": [8, 12, 16][i % 3],
             "moisture": round(float(p[i, 0]), 2), "temp": round(float(p[i, 1]), 2)}
            for i in range(H)]

def estimasi_ukuran(dap):
    m = models(); uk = m["uk"]
    d = float(np.clip(dap, uk["dap"].min(), uk["dap"].max()))
    return {"dap": int(dap),
            "lebar_daun_cm": round(float(m["k_lebar"](d)), 1),
            "tinggi_cm": round(float(m["k_tinggi"](d)), 1)}

def klasifikasi_gambar(path):
    m = models()
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB").resize((IMG, IMG))
    with lock:
        pr = m["cnn"].predict(np.array(im, dtype="float32")[None], verbose=0)[0]
    return m["cfg"]["kelas"][int(pr.argmax())], float(pr.max())

# ---------------------------------------------------------------- endpoint
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")

@app.get("/api/sensor/live")
def sensor_live(n: int = 40):
    """Dipanggil dasbor tiap beberapa detik. Mode demo: kursor maju 1 pembacaan."""
    if DEMO:
        total = db().execute("SELECT COUNT(*) FROM sensor").fetchone()[0]
        cursor["pos"] = cursor["pos"] + 1 if cursor["pos"] < total else 60
    rows = baris_terbaru(n)
    return {"demo": DEMO, "data": rows}

class SensorIn(BaseModel):
    moisture: float
    temp: float
    dap: int | None = None

@app.post("/api/sensor")
def sensor_masuk(s: SensorIn):
    """Untuk sensor asli (mis. ESP32): kirim JSON {"moisture": 55, "temp": 27.1, "dap": 22}."""
    now = datetime.now()
    con = db()
    last = con.execute("SELECT dap FROM sensor ORDER BY id DESC LIMIT 1").fetchone()
    dap = s.dap if s.dap is not None else (last["dap"] if last else 10)
    con.execute("INSERT INTO sensor (ts,dap,jam,moisture,temp) VALUES (?,?,?,?,?)",
                (now.isoformat(timespec="seconds"), dap, now.hour, s.moisture, s.temp))
    con.commit(); con.close()
    return {"ok": True}

@app.get("/api/prediksi")
def api_prediksi():
    return {"prediksi": prediksi_4_hari()}

@app.get("/api/ukuran")
def api_ukuran(dap: int):
    return estimasi_ukuran(dap)

@app.post("/api/analisis")
async def api_analisis(dap: int = Form(...), gambar: UploadFile = File(...)):
    ext = Path(gambar.filename).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        raise HTTPException(400, "Format gambar harus jpg/jpeg/png")
    nama = f"{datetime.now():%Y%m%d_%H%M%S}{ext}"
    path = UPLOADS / nama
    path.write_bytes(await gambar.read())

    tahap, conf = klasifikasi_gambar(path)
    pred = prediksi_4_hari()
    ukr = estimasi_ukuran(dap)
    terakhir = baris_terbaru(1)[0]
    hasil = {"waktu": datetime.now().isoformat(timespec="seconds"), "dap": dap,
             "tahap": tahap, "keyakinan": round(conf, 3),
             "suhu_terakhir": terakhir["temp"], "kelembaban_terakhir": terakhir["moisture"],
             "lebar_daun_cm": ukr["lebar_daun_cm"], "tinggi_cm": ukr["tinggi_cm"],
             "prediksi_4_hari": pred, "nama_gambar": nama}
    con = db()
    con.execute("""INSERT INTO riwayat (waktu,dap,tahap_cnn,keyakinan,suhu_terakhir,kelembaban_terakhir,
                   lebar_daun_cm,tinggi_cm,prediksi_json,nama_gambar) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (hasil["waktu"], dap, tahap, hasil["keyakinan"], hasil["suhu_terakhir"],
                 hasil["kelembaban_terakhir"], ukr["lebar_daun_cm"], ukr["tinggi_cm"],
                 json.dumps(pred), nama))
    con.commit(); con.close()
    return hasil

@app.get("/api/riwayat")
def api_riwayat(limit: int = 20):
    con = db()
    rows = con.execute("SELECT id,waktu,dap,tahap_cnn,keyakinan,suhu_terakhir,kelembaban_terakhir,"
                       "lebar_daun_cm,tinggi_cm,nama_gambar FROM riwayat ORDER BY id DESC LIMIT ?",
                       (limit,)).fetchall()
    con.close()
    return [dict(r) for r in rows]