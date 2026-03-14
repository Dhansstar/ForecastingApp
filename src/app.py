import streamlit as st
import os
import sys
# Remove 'from src'
import eda
import prediction

# Menambahkan folder src ke path agar bisa di-import
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src import eda, prediction

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="DemandSense AI Dashboard",
    page_icon="📦",
    layout="wide"
)

# --- CSS CUSTOM UNTUK TAMPILAN ---
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .stButton>button { width: 100%; border-radius: 8px; background-color: #3b82f6; color: white; }
    .animate-header { font-weight: bold; color: #3b82f6; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🚀 DemandSense Menu")
page = st.sidebar.selectbox("Pilih Modul:", ["Dashboard EDA", "AI Demand Forecasting"])

st.sidebar.divider()
st.sidebar.info("Gunakan modul EDA untuk analisis historis dan Demand Forecasting untuk prediksi stok mendatang.")

# --- ROUTING ---
if page == "Dashboard EDA":
    eda.run()
else:
    prediction.run()