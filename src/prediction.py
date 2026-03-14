import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import glob
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- 1. CORE FUNCTIONS (Sesuai Notebook) ---

def generate_long_term_forecast(report_df, test_meta, horizons=[3, 6]):
    long_term_data = []
    for kat, data in test_meta.items():
        # Baseline dari hasil 1 bulan
        row_kat = report_df[report_df['Kategori'] == kat]
        if row_kat.empty: continue
        
        mae_base = row_kat['MAE (Daily)'].values[0]
        total_pred_1m = row_kat['Predicted Total'].values[0]
        vol_acc_base = float(str(row_kat['Vol Acc (%)'].values[0]).replace('%',''))

        for month in horizons:
            error_multiplier = 1 + (month * 0.15)
            pred_total = int(total_pred_1m * month)
            
            # Simulasi metrik jangka panjang
            mae_sim = round(mae_base * error_multiplier, 2)
            mape_sim = min(20 + (month * 2), 45) # Simulasi degradasi akurasi
            vol_acc_sim = max(0, vol_acc_base - (month * 3))
            
            safe_perc = min(mape_sim + (month * 1.5), 40)
            safety_stock = np.ceil(pred_total * (safe_perc / 100))
            
            long_term_data.append({
                "Kategori": kat,
                "Horizon": f"{month} Months",
                "MAE (Sim)": mae_sim,
                "MAPE (%)": f"{mape_sim:.2f}%",
                "Vol Acc (%)": f"{vol_acc_sim:.2f}%",
                "Est. Demand": pred_total,
                "Safety Stock": int(safety_stock),
                "Total Procurement": int(pred_total + safety_stock)
            })
    return pd.DataFrame(long_term_data)

def run():
    st.title("🚀 DemandSense AI: Pro-Level Forecasting")
    st.markdown("---")

    # --- 2. DATA LOADING & PREPROCESSING ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(os.path.join(current_dir, "forecast_*_data.csv"))
    
    if not files:
        st.error("CSV files not found in src folder!")
        return

    all_dfs = []
    for f in files:
        df = pd.read_csv(f)
        fb = os.path.splitext(os.path.basename(f))[0]
        df['Kategori'] = fb.replace('forecast_', '').replace('_data', '').capitalize()
        df['Waktu Pesanan Dibuat'] = pd.to_datetime(df['Waktu Pesanan Dibuat'])
        df['Net_Sales'] = (df['Jumlah'] - df['Returned Quantity']).clip(lower=0)
        all_dfs.append(df)
    
    full_df = pd.concat(all_dfs, ignore_index=True).sort_values('Waktu Pesanan Dibuat')

    # --- 3. INFERENCE ENGINE (SIMULATED FOR UI) ---
    # Di sini aslinya lo load model_rnn.h5 dan xgb_vol.json
    
    categories = full_df['Kategori'].unique()
    test_meta = {}
    final_report_list = []

    with st.spinner("Calculating Optimized Ensemble Forecasts..."):
        for kat in categories:
            temp = full_df[full_df['Kategori'] == kat].tail(30)
            y_act = temp['Net_Sales'].values
            
            # Mock Inference sesuai logic Notebook lo
            # P_V (Volume) & P_M (MAPE)
            p_v = np.random.normal(np.mean(y_act), np.std(y_act), 30).clip(min=0)
            p_m = np.random.normal(np.mean(y_act), np.std(y_act)*0.8, 30).clip(min=0)
            
            if kat in ['Kitchen', 'Home']:
                preds = (0.8 * p_v) + (0.2 * p_m)
            else:
                preds = np.minimum(p_v, p_m) # Konservatif
            
            preds = np.ceil(np.clip(preds, 0, np.max(y_act)*1.6))
            
            total_act = np.sum(y_act)
            total_pred = np.sum(preds)
            vol_acc = (1 - (abs(total_act - total_pred) / (total_act + 1e-7))) * 100
            mape_30d = (abs(total_act - total_pred) / (total_act + 1e-7)) * 100
            safety_stock = np.ceil(total_pred * (min(mape_30d, 25) / 100))

            final_report_list.append({
                "Kategori": kat,
                "MAE (Daily)": round(np.mean(np.abs(y_act - preds)), 2),
                "MAPE 30D (%)": f"{mape_30d:.2f}%",
                "Vol Acc (%)": f"{max(0, vol_acc):.2f}%",
                "Actual Total": int(total_act),
                "Predicted Total": int(total_pred),
                "Safety Stock": int(safety_stock),
                "Total Rec.": int(total_pred + safety_stock),
                "preds_series": preds
            })
            test_meta[kat] = {'y_act': y_act, 'X_test': None} # Metadata minimal

    report_df = pd.DataFrame(final_report_list)

    # --- 4. VISUALIZATION: SHORT TERM DASHBOARD ---
    st.subheader("📊 30-Day Accuracy Analysis: Actual vs Recommendation")
    
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Bar(x=report_df['Kategori'], y=report_df['Actual Total'], name='Actual Demand', marker_color='#1f77b4'))
    fig_acc.add_trace(go.Bar(x=report_df['Kategori'], y=report_df['Total Rec.'], name='Total Recommendation', marker_color='#ff7f0e'))
    
    fig_acc.update_layout(barmode='group', template='plotly_white', height=500)
    
    # Add Acc Annotations
    for i, row in report_df.iterrows():
        fig_acc.add_annotation(x=row['Kategori'], y=max(row['Actual Total'], row['Total Rec.']) * 1.05,
                               text=f"<b>Acc: {row['Vol Acc (%)']}</b>", showarrow=False, font=dict(size=10))
    st.plotly_chart(fig_acc, use_container_width=True)

    # --- 5. LONG TERM FORECASTING ---
    st.markdown("---")
    st.subheader("🔮 Strategic Long-Term Forecasting (3 & 6 Months)")
    
    long_term_df = generate_long_term_forecast(report_df, test_meta)
    
    # Faceted Bar Chart
    fig_lt_bar = px.bar(long_term_df, x="Kategori", y=["Est. Demand", "Safety Stock"], 
                        facet_col="Horizon", barmode="group",
                        color_discrete_sequence=['#3498db', '#e74c3c'])
    fig_lt_bar.update_layout(template="plotly_white", height=450)
    st.plotly_chart(fig_lt_bar, use_container_width=True)

    # --- 6. DETAILED COMPREHENSIVE REPORT ---
    st.markdown("---")
    target_kat = st.selectbox("Select Category for Detailed Planning:", categories)
    target_hz = st.radio("Select Planning Horizon:", ["3 Months", "6 Months"], horizontal=True)
    
    row_data = long_term_df[(long_term_df['Kategori'] == target_kat) & (long_term_df['Horizon'] == target_hz)].iloc[0]
    
    # Metrics Table
    fig_tbl = go.Figure(data=[go.Table(
        header=dict(values=["Metric", "Value"], fill_color='#2c3e50', font=dict(color='white')),
        cells=dict(values=[["MAPE (%)", "MAE (Units)", "Vol Acc (%)", "Total Procurement"], 
                           [row_data['MAPE (%)'], row_data['MAE (Sim)'], row_data['Vol Acc (%)'], row_data['Total Procurement']]])
    )])
    fig_tbl.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_tbl, use_container_width=True)

    # Simulation Chart (Daily Noise)
    num_days = int(target_hz.split(' ')[0]) * 30
    avg_d = row_data['Est. Demand'] / num_days
    sim_daily = np.random.normal(avg_d, avg_d * 0.2, num_days).clip(min=0)
    
    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(y=sim_daily, name="Daily Forecast", line=dict(color='#3498db', dash='dash')))
    fig_sim.add_hrect(y0=avg_d * 0.8, y1=avg_d * 1.2, fillcolor="green", opacity=0.1, layer="below", line_width=0, name="Safe Zone")
    fig_sim.update_layout(title=f"Simulated Daily Demand - {target_kat} ({target_hz})", template="plotly_white")
    st.plotly_chart(fig_sim, use_container_width=True)

    # --- 7. RSI MOMENTUM (Pilihan Tambahan) ---
    if st.checkbox("Show RSI Momentum Analysis"):
        y_vals = report_df[report_df['Kategori'] == target_kat]['preds_series'].values[0]
        delta = pd.Series(y_vals).diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = 100 - (100 / (1 + (gain / (loss + 1e-7))))
        
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(y=rsi, name="RSI", line=dict(color='purple')))
        fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
        fig_rsi.update_layout(height=300, title="RSI Momentum Predictor", template="plotly_white")
        st.plotly_chart(fig_rsi, use_container_width=True)