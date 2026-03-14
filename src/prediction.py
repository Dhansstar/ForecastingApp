import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

def run():
    st.markdown('<h2 class="animate-header">🔮 AI Demand Forecasting</h2>', unsafe_allow_html=True)
    
    BASE_DIR = os.path.dirname(__file__)

    @st.cache_resource
    def load_assets():
        # Load model & pickle dari folder src
        fe = load_model(os.path.join(BASE_DIR, 'feature_extractor.keras'))
        with open(os.path.join(BASE_DIR, 'xgb_volume.pkl'), 'rb') as f:
            xgb_vol = pickle.load(f)
        with open(os.path.join(BASE_DIR, 'xgb_mape.pkl'), 'rb') as f:
            xgb_mape = pickle.load(f)
        with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(BASE_DIR, 'encoder.pkl'), 'rb') as f:
            encoder = pickle.load(f)
        return fe, xgb_vol, xgb_mape, scaler, encoder

    fe, xgb_vol, xgb_mape, scaler, encoder = load_assets()

    def process_inference(df_raw, category_name):
        df = df_raw.copy()
        df['Waktu Pesanan Dibuat'] = pd.to_datetime(df['Waktu Pesanan Dibuat'])
        df = df.sort_values('Waktu Pesanan Dibuat')
        
        # Preprocessing sesuai notebook
        df['Net_Sales'] = (df['Jumlah'] - df['Returned Quantity']).clip(lower=0)
        df['lag_1'] = np.log1p(df['Net_Sales'].shift(1).fillna(0))
        df['lag_7'] = np.log1p(df['Net_Sales'].shift(7).fillna(0))
        df['lag_28'] = np.log1p(df['Net_Sales'].shift(28).fillna(0))
        df['ma_7'] = np.log1p(df['Net_Sales'].rolling(7).mean().fillna(0))
        df['day_sin'] = np.sin(2 * np.pi * df['Waktu Pesanan Dibuat'].dt.day / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['Waktu Pesanan Dibuat'].dt.day / 31)
        
        num_cols = ['day_sin', 'day_cos', 'lag_1', 'lag_7', 'lag_28', 'ma_7']
        df[num_cols] = scaler.transform(df[num_cols])
        
        kat_encoded = encoder.transform([[category_name]])
        kat_cols = [f"Cat_{c}" for c in encoder.categories_[0]]
        for i, col in enumerate(kat_cols):
            df[col] = kat_encoded[0][i]
            
        features = num_cols + kat_cols
        X_input = df[features].tail(30).values.reshape(1, 30, len(features))
        
        # Inference Stage
        extracted_feats = fe.predict(X_input, verbose=0)
        p_vol = np.maximum(xgb_vol.predict(extracted_feats), 0)[0]
        p_mape = np.maximum(xgb_mape.predict(extracted_feats), 0)[0]
        
        # Logika Ensemble Hybrid
        if category_name in ['Kitchen', 'Home']:
            res = (0.8 * p_vol) + (0.2 * p_mape)
        else:
            res = min(p_vol, p_mape)
            
        return int(np.ceil(res))

    if st.button("🚀 Jalankan Kalkulasi Prediksi"):
        csv_files = {
            "Bathroom": "forecast_bathroom_data.csv",
            "Home": "forecast_home_data.csv",
            "Kitchen": "forecast_kitchen_data.csv",
            "Storage": "forecast_storage_data.csv",
            "Tools": "forecast_tools_data.csv",
            "Other": "forecast_other_data.csv"
        }
        
        results = []
        for kat, file_name in csv_files.items():
            path = os.path.join(BASE_DIR, file_name)
            if os.path.exists(path):
                raw = pd.read_csv(path)
                pred = process_inference(raw, kat)
                results.append({"Kategori": kat, "Prediksi Unit Besok": pred})
        
        st.table(pd.DataFrame(results))