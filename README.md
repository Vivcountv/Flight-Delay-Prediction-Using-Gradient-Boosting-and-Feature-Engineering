# ✈️ Prediksi Keterlambatan Penerbangan (Streamlit App)

Ini adalah proyek Ujian Tengah Semester (UTS) Mata Kuliah Pembelajaran Mesin yang bertujuan untuk memprediksi keterlambatan penerbangan sipil di Amerika Serikat.

Model ini menggunakan **LightGBM (Gradient Boosting)** yang dilatih pada 1 juta baris data penerbangan dari tahun 2023. Keakuratan model ditingkatkan secara signifikan melalui **Feature Engineering** cerdas yang menggabungkan data cuaca harian (di bandara asal dan tujuan), fitur kalender (hari libur), dan penanganan *data leakage* (dengan mempertahankan fitur status seperti `Delay_NAS` dan `Delay_LastAircraft`).

Aplikasi ini dibangun dan di-deploy sepenuhnya menggunakan **Streamlit** dan Python.

## 🚀 Demo Aplikasi Web

**Link Deployment Live:** `[MASUKKAN LINK STREAMLIT.APP ANDA DI SINI]`

Aplikasi ini memungkinkan pengguna memasukkan detail penerbangan dan menerima prediksi instan serta estimasi waktu kedatangan (ETA).

| Prediksi Tepat Waktu | Prediksi Terlambat (dengan Rincian Data) |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/Vivcountv/Flight-Delay-Prediction-Using-Gradient-Boosting-and-Feature-Engineering/main/image_dd201c.png" alt="Tampilan Prediksi Tepat Waktu" width="400"> | <img src="https://raw.githubusercontent.com/Vivcountv/Flight-Delay-Prediction-Using-Gradient-Boosting-and-Feature-Engineering/main/image_dd1fbd.png" alt="Tampilan Prediksi Terlambat" width="400"> |

*(Catatan: Gambar akan muncul setelah Anda mengunggah file `image_dd201c.png` dan `image_dd1fbd.png` ke GitHub)*

## 📊 Hasil Model (Performa Final)

Model dievaluasi pada **202.163 data uji** dan mencapai hasil yang sangat kuat, membuktikan bahwa model ini **stabil dan tidak overfitting**.

* **Skor AUC:** **0.9988** (vs 0.9988 pada data validasi)
* **Akurasi:** **98.48%**
* **Recall (Kelas Terlambat):** **97%** (Model berhasil mengidentifikasi 97% dari semua penerbangan yang *sebenarnya* akan terlambat).
* **Precision (Kelas Terlambat):** **95%** (Saat model memprediksi "Terlambat", 95% prediksinya benar).

### Fitur Paling Penting

Analisis *Feature Importance* (**Langkah 18**) menunjukkan bahwa fitur status (`Delay_LastAircraft`, `Dep_Delay`, `Delay_NAS`) adalah prediktor terkuat, diikuti oleh fitur cuaca (`origin_wspd`) dan kalender (`Is_Near_Holiday`).

<img src="https://raw.githubusercontent.com/Vivcountv/Flight-Delay-Prediction-Using-Gradient-Boosting-and-Feature-Engineering/main/image_86b920.png" alt="Grafik Feature Importance" width="600">

---

## 🛠️ Tumpukan Teknologi (Tech Stack)

| Kategori | Teknologi |
| :--- | :--- |
| **Bahasa** | Python 3.11 |
| **Data Science** | Pandas, Numpy, Scikit-learn, LightGBM, Joblib |
| **Aplikasi Web** | Streamlit, Lucide-React |
| **Lingkungan** | Google Colab (Pelatihan), VS Code (Development) |
| **Deployment** | GitHub, Streamlit Community Cloud |

---

## 🚦 Menjalankan Proyek Secara Lokal

1.  **Unduh Repository:**
    ```bash
    git clone [https://github.com/Vivcountv/Flight-Delay-Prediction-Using-Gradient-Boosting-and-Feature-Engineering.git](https://github.com/Vivcountv/Flight-Delay-Prediction-Using-Gradient-Boosting-and-Feature-Engineering.git)
    cd Flight-Delay-Prediction-Using-Gradient-Boosting-and-Feature-Engineering
    ```
2.  **(Opsional) Buat Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # (atau .\venv\Scripts\Activate.ps1 di Windows)
    ```
3.  **Instal Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Jalankan Aplikasi Streamlit:**
    ```bash
    streamlit run app.py
    ```
5.  Buka `http://localhost:8501` di browser Anda.

---

## 📁 Struktur File Proyek