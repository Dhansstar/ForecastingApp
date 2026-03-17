# 🚀 DemandSense AI: Forecasting App Interface
**Interactive Streamlit Deployment for Time-Series Market Prediction**

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces)
[![Status](https://img.shields.io/badge/Status-Live--Deployment-green?style=for-the-badge)](https://huggingface.co/spaces/Dhansstar/ForeacastingApp)

</div>

---

## 📌 1. Project Overview
Aplikasi ini merupakan *deployment interface* resmi dari framework **DemandSense AI**. Dirancang menggunakan **Streamlit**, aplikasi ini memungkinkan pengguna untuk mengeksplorasi wawasan pasar dan melakukan prediksi permintaan inventaris secara *real-time* menggunakan model **Hybrid RNN + XGBoost Ensemble**.

### Key Modules:
* **📉 Exploratory Data Analysis (EDA):** Visualisasi tren penjualan historis, distribusi geografis pesanan, dan faktor pendorong *demand*.
* **🔮 Inventory Forecasting:** Modul prediksi yang memungkinkan pengguna memilih kategori produk spesifik untuk mendapatkan estimasi kuantitas terjual dalam 1 bulan ke depan.

---

## 📁 2. Repository Architecture & Filepath
Struktur repositori ini difokuskan pada efisiensi *cloud deployment*:

```text
.
├── src/
│   ├── app.py              # Main entry point for Streamlit application
│   ├── prediction.py       # Logic for loading Hybrid Models and running inference
│   └── data_loader.py      # Utility for fetching preprocessed datasets
├── requirements.txt        # Manifest of Python dependencies (TensorFlow, XGBoost, etc.)
└── README.md               # Technical documentation for deployment

```

---

## 🔬 3. Technical Implementation & System Architecture

### 🏗️ A. Interface & Model Inference Logic
Gue ngerancang aplikasi ini untuk menjembatani model Deep Learning yang berat dengan antarmuka yang responsif:
* **Hybrid Model Loading:** Menggunakan `prediction.py` untuk memuat arsitektur **RNN (TensorFlow)** sebagai ekstraktor fitur temporal dan **XGBoost** sebagai estimator akhir secara efisien.
* **Real-time Pipeline:** Aplikasi melakukan transformasi input user secara instan menggunakan *joblib artifacts* (Scalers & Encoders) sebelum diumpankan ke model ensemble untuk menjaga akurasi prediksi.
* **Dynamic Visualization:** Integrasi grafik interaktif yang memungkinkan stakeholder melakukan filter data per kategori produk (Kitchen, Home, etc.) untuk melihat tren fluktuasi secara detail.

### ⚙️ B. System Requirements & Environment
Aplikasi ini berjalan di atas Python 3.11 dengan optimasi library sebagai berikut:
* **Predictive Engines:** `tensorflow` untuk RNN logic dan `xgboost` untuk akurasi volume.
* **Data Processing:** `pandas` & `numpy` untuk manipulasi data time-series sebelum inferensi.
* **Frontend:** `streamlit` sebagai framework UI utama yang ringan namun kuat untuk kebutuhan Data Science.

---

## 🚀 4. How to Run Locally & Deployment
Untuk menjalankan **ForecastingApp** di mesin lokal Anda atau melakukan deployment ulang:

1. **Clone & Setup Environment:**
   ```bash
   git clone [https://github.com/username/forecasting-app.git](https://github.com/username/forecasting-app.git)
   cd forecasting-app
   pip install -r requirements.txt
   ```
2. **Execute Application:**
   ```bash
   streamlit run src/app.py
   ```
3. **Deployment Strategy:** Aplikasi ini sepenuhnya dioptimalkan untuk **HuggingFace Spaces** menggunakan Docker runtime. Pastikan seluruh model artifacts (`.h5` dan `.joblib`) telah terunggah ke dalam direktori yang tepat agar *inference engine* dapat memuat model secara otomatis saat inisialisasi.

---

## 🏁 5. Conclusion & Strategic Impact
**ForecastingApp** mentransformasi framework **DemandSense AI** yang kompleks menjadi instrumen pengambilan keputusan praktis bagi manajemen e-commerce. Dengan antarmuka interaktif ini, stakeholder dapat memitigasi risiko *overstock* dan *stockout* secara presisi.

Integrasi arsitektur **Hybrid RNN-XGBoost** di balik UI yang intuitif menjamin bahwa prediksi yang dihasilkan didasarkan pada pola data temporal yang mendalam, namun tetap memberikan kemudahan akses bagi para pengambil kebijakan strategis.

### 🔗 Quick Links
* **Live Deployment:** [Streamlit.app - DemandSense AI App](https://forecastingapp-myproject.streamlit.app/)
* **Main Project Repo:** [DemandSense AI Full Framework](https://github.com/Dhansstar/ForecastingApp)
* **Scientific Reference:** [Hybrid RNN-XGBoost Journal Methodology](https://jcasc.com/index.php/jcasc/article/view/3736)

---
**Author:** Risyadhana Syaifuddin | Data Science & Analytics Practitioner 2026  