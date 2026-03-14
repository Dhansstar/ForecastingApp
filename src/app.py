import streamlit as st
import os
import sys

# Tambahkan path agar bisa baca file di dalam src jika dipanggil dari luar
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import eda
import prediction

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="DemandSense AI Dashboard",
    page_icon="📦",
    layout="wide"
)

# --- CSS CUSTOM ---
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    [data-testid="stSidebar"] { background-color: #1e293b; }
    .stMetric { background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🚀 DemandSense AI")
page = st.sidebar.selectbox("Pilih Modul:", ["Dashboard EDA", "AI Demand Forecasting"])

st.sidebar.divider()
st.sidebar.info("Gunakan modul EDA untuk analisis historis dan Demand Forecasting untuk prediksi stok.")

# --- ROUTING ---
if page == "Dashboard EDA":
    eda.run()
else:
    prediction.run()