import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import glob

# --- 1. CORE LOGIC: LONG TERM SIMULATION ---
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
            # Simulasi degradasi akurasi (15% per bulan tambahan)
            error_multiplier = 1 + (month * 0.15) if month > 1 else 1.0
            pred_total = int(total_pred_1m * month)
            
            mae_sim = round(mae_base * error_multiplier, 2)
            mape_sim = min(mape_base + (month * 2), 45) if month > 1 else mape_base
            vol_acc_sim = max(0, vol_acc_base - (month * 3)) if month > 1 else vol_acc_base
            
            # Safety stock: Prediksi * (MAPE + Risk Factor)
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

    # --- 2. DATA LOADING ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Pastikan file CSV lo ada di folder yang sama atau sesuaikan path-nya
    files = glob.glob(os.path.join(current_dir, "forecast_*_data.csv"))
    
    if not files:
        st.error("❌ Data CSV tidak ditemukan di folder src!")
        return

    all_dfs = []
    for f in files:
        df = pd.read_csv(f)
        fb = os.path.splitext(os.path.basename(f))[0]
        # Mapping nama kategori biar rapi
        kat_name = fb.replace('forecast_', '').replace('_data', '').capitalize()
        df['Kategori'] = kat_name
        df['Waktu Pesanan Dibuat'] = pd.to_datetime(df['Waktu Pesanan Dibuat'])
        df['Net_Sales'] = (df['Jumlah'] - df['Returned Quantity']).clip(lower=0)
        all_dfs.append(df)
    
    full_df = pd.concat(all_dfs, ignore_index=True).sort_values('Waktu Pesanan Dibuat')
    categories = full_df['Kategori'].unique()

    # --- 3. ENSEMBLE INFERENCE (Sesuai Logic Notebook) ---
    final_report_list = []
    with st.spinner("Calculating Optimized Ensemble Forecasts..."):
        for kat in categories:
            temp = full_df[full_df['Kategori'] == kat].tail(30)
            y_act = temp['Net_Sales'].values
            
            # Simulasi Model: P_V (Volume) & P_M (MAPE)
            # Gantilah dengan model.predict() lo yang asli jika sudah siap
            p_v = np.random.normal(np.mean(y_act), np.std(y_act), 30).clip(min=0)
            p_m = np.random.normal(np.mean(y_act), np.std(y_act)*0.8, 30).clip(min=0)
            
            # Optimized Ensemble Logic
            if kat in ['Kitchen', 'Home']:
                preds = (0.8 * p_v) + (0.2 * p_m)
            else:
                preds = np.minimum(p_v, p_m) 
            
            preds = np.ceil(np.clip(preds, 0, np.max(y_act)*1.6))
            
            # Metrics
            total_act = np.sum(y_act)
            total_pred = np.sum(preds)
            vol_acc = (1 - (abs(total_act - total_pred) / (total_act + 1e-7))) * 100
            mape_30d = (abs(total_act - total_pred) / (total_act + 1e-7)) * 100
            
            final_report_list.append({
                "Kategori": kat,
                "MAE (Daily)": round(np.mean(np.abs(y_act - preds)), 2),
                "MAPE 30D (%)": f"{mape_30d:.2f}%",
                "Vol Acc (%)": f"{max(0, vol_acc):.2f}%",
                "Actual Total": int(total_act),
                "Predicted Total": int(total_pred)
            })

    report_df = pd.DataFrame(final_report_list)
    long_term_report = generate_long_term_forecast(report_df, categories)

    # --- 4. SELECTION MENU (SIDEBAR) ---
    st.sidebar.header("🕹️ Control Panel")
    target_kat = st.sidebar.selectbox("Pilih Kategori Produk:", categories)
    target_hz = st.sidebar.radio("Pilih Horizon Perencanaan:", ["1 Months", "3 Months", "6 Months"])

    # --- 5. DETAILED DASHBOARD (TABLE + GRAPH) ---
    st.subheader(f"📊 Detailed Forecast: {target_kat} ({target_hz})")
    
    # Filter Row Data
    row_data = long_term_report[(long_term_report['Kategori'] == target_kat) & 
                                (long_term_report['Horizon'] == target_hz)].iloc[0]
    
    # Data Historis & Tanggal Forecast
    df_hist = full_df[full_df['Kategori'] == target_kat].tail(30)
    last_date = df_hist['Waktu Pesanan Dibuat'].max()
    num_months = int(target_hz.split(' ')[0])
    total_days = num_months * 30
    forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, total_days + 1)]
    
    # Simulasi Daily Forecast (Quantity per hari)
    daily_avg = row_data['Est. Demand'] / total_days
    daily_forecast = np.random.normal(daily_avg, daily_avg * 0.2, total_days).clip(min=0)
    daily_forecast = np.ceil(daily_forecast * (row_data['Est. Demand'] / (sum(daily_forecast)+1e-7)))

    # CREATE SUBPLOTS
    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.12,
        specs=[[{"type": "table"}], [{"type": "scatter"}]],
        row_heights=[0.3, 0.7]
    )

    # A. TABEL STRATEGIS (FIXED COLOR)
    fig.add_trace(
        go.Table(
            header=dict(
                values=["<b>Kategori</b>", "<b>Horizon</b>", "<b>MAPE (%)</b>", "<b>MAE</b>", "<b>Vol Acc (%)</b>", "<b>Safety Stock</b>", "<b>Total Proc.</b>"],
                fill_color='#2c3e50', font=dict(color='white', size=11), align='center'
            ),
            cells=dict(
                values=[
                    [f"<b>{row_data['Kategori']}</b>"], [row_data['Horizon']], [f"{row_data['MAPE (%)']:.2f}%"], 
                    [row_data['MAE']], [f"{row_data['Vol Acc (%)']:.2f}%"],
                    [int(row_data['Safety Stock'])], [int(row_data['Total Procurement'])]
                ],
                fill_color='#f8f9fa', font=dict(color='#1f2c39', size=11), align='center'
            )
        ), row=1, col=1
    )

    # B. GRAFIK QUANTITY & TANGGAL
    # Historis
    fig.add_trace(go.Scatter(x=df_hist['Waktu Pesanan Dibuat'], y=df_hist['Net_Sales'], 
                             name='Actual (Last 30D)', line=dict(color='#7f8c8d', width=2)), row=2, col=1)
    # Forecast
    fig.add_trace(go.Scatter(x=forecast_dates, y=daily_forecast, 
                             name=f'Forecast {target_hz}', line=dict(color='#3498db', width=3, dash='dash')), row=2, col=1)
    # Safety Margin
    daily_safe = row_data['Safety Stock'] / total_days
    fig.add_trace(go.Scatter(
        x=forecast_dates + forecast_dates[::-1],
        y=[y + daily_safe for y in daily_forecast] + [max(0, y - daily_safe) for y in daily_forecast][::-1],
        fill='toself', fillcolor='rgba(231, 76, 60, 0.1)',
        line=dict(color='rgba(255,255,255,0)'), name='Safety Stock Margin', showlegend=True
    ), row=2, col=1)

    # Styling Layout
    fig.update_layout(height=750, template="plotly_white", showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="Quantity (Units)", row=2, col=1)
    fig.update_xaxes(title_text="Tanggal / Timeline", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

    # --- 6. GLOBAL COMPARISON ---
    st.markdown("---")
    st.subheader(f"📊 Global Stock Distribution ({target_hz})")
    
    current_hz_data = long_term_report[long_term_report['Horizon'] == target_hz]
    fig_bar = px.bar(current_hz_data, x="Kategori", y=["Est. Demand", "Safety Stock"],
                     barmode="group", color_discrete_sequence=['#3498db', '#e74c3c'])
    st.plotly_chart(fig_bar, use_container_width=True)

# Run apps
if __name__ == "__main__":
    run()