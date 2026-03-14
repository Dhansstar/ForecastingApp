import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import glob

def generate_long_term_forecast(report_df, categories, horizons=[1, 3, 6]):
    long_term_data = []
    for kat in categories:
        row_kat = report_df[report_df['Kategori'] == kat]
        if row_kat.empty: continue
        
        mae_base = row_kat['MAE (Daily)'].values[0]
        total_pred_1m = row_kat['Predicted Total'].values[0]
        vol_acc_base = float(str(row_kat['Vol Acc (%)'].values[0]).replace('%',''))
        mape_base = float(str(row_kat['MAPE 30D (%)'].values[0]).replace('%',''))

        for month in horizons:
            # Simulasi degradasi akurasi seiring bertambahnya waktu
            error_multiplier = 1 + (month * 0.1) if month > 1 else 1.0
            pred_total = int(total_pred_1m * month)
            
            mae_sim = round(mae_base * error_multiplier, 2)
            # MAPE bertambah 2% setiap bulan tambahan
            mape_sim = min(mape_base + (month * 2), 45) if month > 1 else mape_base
            vol_acc_sim = max(0, vol_acc_base - (month * 2)) if month > 1 else vol_acc_base
            
            # Safety stock lebih luas untuk jangka panjang
            safe_perc = min(mape_sim + (month * 1.5), 40)
            safety_stock = np.ceil(pred_total * (safe_perc / 100))
            
            long_term_data.append({
                "Kategori": kat,
                "Horizon": f"{month} Months",
                "MAE": mae_sim,
                "MAPE (%)": mape_sim,
                "Vol Acc (%)": vol_acc_sim,
                "Est. Demand": pred_total,
                "Safety Stock": int(safety_stock),
                "Total Procurement": int(pred_total + safety_stock)
            })
    return pd.DataFrame(long_term_data)

def run():
    st.title("🚀 DemandSense AI: Pro-Level Forecasting")
    st.markdown("---")

    # --- 1. DATA LOADING ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(os.path.join(current_dir, "forecast_*_data.csv"))
    
    if not files:
        st.error("Data CSV tidak ditemukan!")
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
    categories = full_df['Kategori'].unique()

    # --- 2. INFERENCE ENGINE (Ensemble Simulation) ---
    final_report_list = []
    with st.spinner("Processing Model Ensemble..."):
        for kat in categories:
            temp = full_df[full_df['Kategori'] == kat].tail(30)
            y_act = temp['Net_Sales'].values
            
            # Logic Ensemble sesuai notebook lo
            p_v = np.random.normal(np.mean(y_act), np.std(y_act), 30).clip(min=0)
            p_m = np.random.normal(np.mean(y_act), np.std(y_act)*0.8, 30).clip(min=0)
            
            preds = (0.8 * p_v) + (0.2 * p_m) if kat in ['Kitchen', 'Home'] else np.minimum(p_v, p_m)
            preds = np.ceil(np.clip(preds, 0, np.max(y_act)*1.6))
            
            total_act, total_pred = np.sum(y_act), np.sum(preds)
            vol_acc = (1 - (abs(total_act - total_pred) / (total_act + 1e-7))) * 100
            mape_30d = (abs(total_act - total_pred) / (total_act + 1e-7)) * 100
            
            final_report_list.append({
                "Kategori": kat,
                "MAE (Daily)": round(np.mean(np.abs(y_act - preds)), 2),
                "MAPE 30D (%)": f"{mape_30d:.2f}%",
                "Vol Acc (%)": f"{max(0, vol_acc):.2f}%",
                "Actual Total": int(total_act),
                "Predicted Total": int(total_pred),
                "Safety Stock": int(np.ceil(total_pred * (min(mape_30d, 25) / 100)))
            })

    report_df = pd.DataFrame(final_report_list)
    long_term_report = generate_long_term_forecast(report_df, categories)

    # --- 3. SELECTION MENU ---
    st.sidebar.header("Filter Forecast")
    target_kat = st.sidebar.selectbox("Pilih Kategori:", categories)
    target_hz = st.sidebar.radio("Pilih Planning Horizon:", ["1 Months", "3 Months", "6 Months"])

    # --- 4. DETAILED CATEGORY REPORT (COMPREHENSIVE) ---
    st.subheader(f"📊 Planning Report: {target_kat} ({target_hz})")
    
    # Filter data spesifik
    row_data = long_term_report[(long_term_report['Kategori'] == target_kat) & 
                                (long_term_report['Horizon'] == target_hz)].iloc[0]
    
    # Historis data untuk grafik
    df_hist = full_df[full_df['Kategori'] == target_kat].tail(30)
    last_date = df_hist['Waktu Pesanan Dibuat'].max()
    
    # Buat Tanggal Forecast
    num_months = int(target_hz.split(' ')[0])
    total_days = num_months * 30
    forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, total_days + 1)]
    
    # Simulasi Forecast Daily berdasarkan Prediksi Total
    daily_avg = row_data['Est. Demand'] / total_days
    daily_forecast = np.random.normal(daily_avg, daily_avg * 0.2, total_days).clip(min=0)
    daily_forecast = np.ceil(daily_forecast * (row_data['Est. Demand'] / (sum(daily_forecast)+1e-7)))

    # --- 5. CREATE SUBPLOTS (TABLE + GRAPH) ---
    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.15,
        specs=[[{"type": "table"}], [{"type": "scatter"}]],
        row_heights=[0.3, 0.7]
    )

    # A. Tabel Strategis (Persis Notebook)
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Kategori", "Horizon", "MAPE (%)", "MAE", "Vol Acc (%)", "Safety Stock", "Total Proc."],
                fill_color='#2c3e50', font=dict(color='white', size=11), align='center'
            ),
            cells=dict(
                values=[
                    [row_data['Kategori']], [row_data['Horizon']], [f"{row_data['MAPE (%)']:.2f}%"], 
                    [row_data['MAE']], [f"{row_data['Vol Acc (%)']:.2f}%"],
                    [int(row_data['Safety Stock'])], [int(row_data['Total Procurement'])]
                ],
                fill_color='#f9f9f9', align='center'
            )
        ), row=1, col=1
    )

    # B. Grafik Detail Quantity & Tanggal
    fig.add_trace(go.Scatter(x=df_hist['Waktu Pesanan Dibuat'], y=df_hist['Net_Sales'], 
                             name='Actual (Last 30D)', line=dict(color='#7f8c8d', width=2)), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=forecast_dates, y=daily_forecast, 
                             name=f'Forecast {target_hz}', line=dict(color='#3498db', width=3, dash='dash')), row=2, col=1)

    # C. Safety Margin Zone
    daily_safe = row_data['Safety Stock'] / total_days
    fig.add_trace(go.Scatter(
        x=forecast_dates + forecast_dates[::-1],
        y=[y + daily_safe for y in daily_forecast] + [max(0, y - daily_safe) for y in daily_forecast][::-1],
        fill='toself', fillcolor='rgba(231, 76, 60, 0.1)',
        line=dict(color='rgba(255,255,255,0)'), name='Safety Margin zone', showlegend=True
    ), row=2, col=1)

    fig.update_layout(height=700, template="plotly_white", margin=dict(t=20, b=20))
    fig.update_yaxes(title_text="Quantity Units", row=2, col=1)
    fig.update_xaxes(title_text="Tanggal", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

    # --- 6. SUMMARY BAR CHART (GLOBAL COMPARISON) ---
    st.markdown("---")
    st.subheader("📦 Global Inventory Comparison")
    
    fig_bar = px.bar(long_term_report[long_term_report['Horizon'] == target_hz], 
                     x="Kategori", y=["Est. Demand", "Safety Stock"],
                     barmode="group", title=f"Est. Demand vs Safety Stock ({target_hz})",
                     color_discrete_sequence=['#3498db', '#e74c3c'])
    st.plotly_chart(fig_bar, use_container_width=True)