# Website Monitoring Pakcoy (siap jalan)

Model hasil training Anda sudah ada di folder `models/`.

## Cara menjalankan
**Windows:** klik dua kali `jalankan.bat`
(pertama kali akan memasang library, butuh internet dan beberapa menit; browser terbuka otomatis).

**Mac/Linux:** `bash jalankan.sh`, lalu buka http://127.0.0.1:8000

Butuh Python 3.10 - 3.13 (https://www.python.org/downloads/, centang "Add Python to PATH" saat install di Windows).

## Mode
- Demo (default): data Excel diputar ulang, grafik bergerak tiap 3 detik.
- Sensor asli: set `DEMO_MODE=0`, sensor kirim `POST /api/sensor` {"moisture":55,"temp":27.1,"dap":22}.

## Yang perlu diganti
- `models/ukuran_referensi.csv`: angka CONTOH, ganti dengan hasil ukur nyata (dap,lebar_daun_cm,tinggi_cm).
- `TANGGAL_MULAI_DATA` di main.py: tanggal asli data hari ke-1.

## Endpoint
GET /api/sensor/live, POST /api/sensor, GET /api/prediksi, GET /api/ukuran?dap=, POST /api/analisis, GET /api/riwayat
