import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import glob
from datetime import timedelta

# --- 1. CORE FUNCTIONS (Logic Perencanaan Jangka Panjang) ---

def generate_long_term_forecast(report_df, categories, horizons=[1, 3, 6]):
    """Menghasilkan simulasi perencanaan stok untuk 1, 3, dan 6 bulan."""
    long_term_data = []
    for kat in categories:
        # Ambil baseline metrik dari hasil evaluasi 1 bulan
        row_kat = report_df[report_df['Kategori'] == kat]
        if row_kat.empty: continue
        
        # Ekstrak data baseline
        mae_base = row_kat['MAE (Daily)'].values[0]
        total_pred_1m = row_kat['Predicted Total'].values[0]
        vol_acc_base = float(str(row_kat['Vol Acc (%)'].values[0]).replace('%',''))
        mape_base = float(str(row_kat['MAPE 30D (%)'].values[0]).replace('%',''))

        for month in horizons:
            # Simulasi degradasi akurasi seiring bertambahnya horizon waktu
            error_multiplier = 1 + (month * 0.15) if month > 1 else 1.0
            pred_total = int(total_pred_1m * month)
            
            # Perhitungan metrik simulasi
            mae_sim = round(mae_base * error_multiplier, 2)
            mape_sim = min(mape_base + (month * 2), 45) if month > 1 else mape_base
            vol_acc_sim = max(0, vol_acc_base - (month * 3)) if month > 1 else vol_acc_base
            
            # Perhitungan Safety Stock yang lebih agresif untuk jangka panjang
            # Rumus: Prediksi * (MAPE simulasi + Faktor Risiko Tambahan)
            safe_perc = min(mape_sim + (month * 1.5), 40) # Cap safety stock di 40%
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

    # --- 2. DATA LOADING & PREPROCESSING (Logic Sesuai Notebook) ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    files = glob.glob(os.path.join(current_dir, "forecast_*_data.csv"))
    
    if not files:
        st.error("❌ Data CSV tidak ditemukan di folder src!")
        return

    all_dfs = []
    for f in files:
        df = pd.read_csv(f)
        fb = os.path.splitext(os.path.basename(f))[0]
        # Mapping nama kategori biar rapi di dashboard
        kat_name = fb.replace('forecast_', '').replace('_data', '').capitalize()
        df['Kategori'] = kat_name
        df['Waktu Pesanan Dibuat'] = pd.to_datetime(df['Waktu Pesanan Dibuat'])
        # Preprocessing Sesuai Notebook
        df['Net_Sales'] = (df['Jumlah'] - df['Returned Quantity']).clip(lower=0)
        # Hitung Moving Average untuk visualisasi tambahan
        df['ma_7'] = df['Net_Sales'].rolling(7).mean().fillna(0)
        all_dfs.append(df)
    
    # Gabungkan data dan urutkan berdasarkan waktu
    full_df = pd.concat(all_dfs, ignore_index=True).sort_values('Waktu Pesanan Dibuat')
    categories = full_df['Kategori'].unique()

    # --- 3. ENSEMBLE INFERENCE (Mirroring Logic Notebook) ---
    final_report_list = []
    
    # Wadah untuk menyimpan series harian prediksi agar bisa ditampilkan di grafik
    all_daily_preds = {}

    with st.spinner("Calculating Optimized Ensemble Forecasts..."):
        for kat in categories:
            # Ambil 30 hari terakhir sebagai input model
            temp = full_df[full_df['Kategori'] == kat].tail(30)
            y_act = temp['Net_Sales'].values
            
            # --- SIMULASI MODEL (Hybrid RNN + XGBoost) ---
            # Gantilah bagian ini dengan pemanggilan model lo yang asli:
            # P_V = model_vol.predict(features)
            # P_M = model_mape.predict(features)
            
            # Simulasi Model Volume (P_V) & Model MAPE (P_M)
            p_v = np.random.normal(np.mean(y_act), np.std(y_act), 30).clip(min=0)
            p_m = np.random.normal(np.mean(y_act), np.std(y_act)*0.8, 30).clip(min=0)
            
            # --- OPTIMIZED ENSEMBLE LOGIC ---
            if kat in ['Kitchen', 'Home']:
                # Bobot lebih berat ke Model Volume untuk kategori besar
                preds_raw = (0.8 * p_v) + (0.2 * p_m)
            else:
                # Pendekatan konservatif (minimal) untuk kategori kecil
                preds_raw = np.minimum(p_v, p_m) 
            
            # --- ADAPTIVE GUARDRAILS ---
            # Terapkan ceil agar angka stok bulat, dan clip agar tidak overshoot
            preds = np.ceil(np.clip(preds_raw, 0, np.max(y_act)*1.6))
            
            # Simpan series untuk grafik
            all_daily_preds[kat] = preds
            
            # --- HITUNG METRIK AGREGASI ---
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

    # Siapkan dataframe untuk laporan evaluasi
    report_df = pd.DataFrame(final_report_list)
    # Jalankan simulasi jangka panjang
    long_term_report = generate_long_term_forecast(report_df, categories)

    # --- 4. SELECTION MENU (SIDEBAR) ---
    st.sidebar.header("🕹️ Control Panel")
    target_kat = st.sidebar.selectbox("Pilih Kategori Produk:", categories)
    target_hz = st.sidebar.radio("Pilih Horizon Perencanaan:", ["1 Months", "3 Months", "6 Months"])

    # --- 5. DETAILED DASHBOARD (TABLE + GRAPH) ---
    st.subheader(f"📊 Detailed Planning: {target_kat} ({target_hz})")
    
    # Filter data perencanaan yang sesuai pilihan
    row_data = long_term_report[(long_term_report['Kategori'] == target_kat) & 
                                (long_term_report['Horizon'] == target_hz)].iloc[0]
    
    # --- PENYIAPAN DATA GRAFIK (Menyatukan Garis) ---
    # A. Ambil Data Historis (Solid Line)
    df_hist = full_df[full_df['Kategori'] == target_kat].tail(30)
    last_date = df_hist['Waktu Pesanan Dibuat'].max()
    
    # B. Buat Timeline Masa Depan
    num_months = int(target_hz.split(' ')[0])
    total_days = num_months * 30
    forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, total_days + 1)]
    
    # C. Siapkan Series Prediksi Harian
    if target_hz == "1 Months":
        # Gunakan hasil proyeksi model yang sudah dihitung di Loop Ensemble
        daily_forecast_raw = all_daily_preds[target_kat]
    else:
        # Gunakan baseline harian dari Prediksi Total yang disimulasikan
        daily_avg_base = row_data['Est. Demand'] / total_days
        # Tambahkan noise acak (volatilitas) agar terlihat alami di grafik
        daily_noise = np.random.normal(0, daily_avg_base * 0.15, total_days)
        daily_forecast_raw = np.ceil((daily_avg_base + daily_noise).clip(min=0))
        
    # **POIN PENTING: Menyatukan Titik**
    # Series proyeksi dimulai dari hari *setelah* data historis berakhir.
    # Garis historis dan proyeksi akan 'tersambung' secara visual karena data proyeksi
    # ditempatkan tepat setelah tanggal terakhir historis.
    
    # --- BUAT SUBPLOTS ---
    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.12,
        specs=[[{"type": "table"}], [{"type": "scatter"}]],
        row_heights=[0.3, 0.7]
    )

    # A. TABEL STRATEGIS (FIXED COLOR: Teks Gelap, Background Terang)
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
                # WARNA TEXT DI SINI SEKARANG HITAM (#1f2c39)
                fill_color='#f8f9fa', font=dict(color='#1f2c39', size=11), align='center'
            )
        ), row=1, col=1
    )

    # B. GRAFIK QUANTITY & TANGGAL DETIL
    # Historis: Sumbu X Tanggal, Sumbu Y Quantity
    fig.add_trace(go.Scatter(x=df_hist['Waktu Pesanan Dibuat'], y=df_hist['Net_Sales'], 
                             name='Actual (Last 30D)', line=dict(color='#7f8c8d', width=2)), row=2, col=1)
    
    # Tambahkan Moving Average (ma_7) untuk melihat tren historis
    fig.add_trace(go.Scatter(x=df_hist['Waktu Pesanan Dibuat'], y=df_hist['ma_7'], 
                             name='MA 7-Days (Tren)', line=dict(color='#2ecc71', width=1, dash='dot')), row=2, col=1)

    # Proyeksi: Pake warna biru dan garis putus-putus
    fig.add_trace(go.Scatter(x=forecast_dates, y=daily_forecast_raw, 
                             name=f'Forecast {target_hz}', line=dict(color='#3498db', width=3, dash='dash')), row=2, col=1)
    
    # Tambahkan Garis Pemisah "Hari Ini" (Cut-off Line)
    fig.add_vline(x=last_date, line_width=2, line_dash="solid", line_color="#2ecc71", row=2, col=1)

    # C. SAFETY MARGIN ZONE (PENTING!)
    # Area transparan di sekitar garis prediksi untuk menunjukkan margin kesalahan
    daily_safe_margin = row_data['Safety Stock'] / total_days
    fig.add_trace(go.Scatter(
        x=forecast_dates + forecast_dates[::-1],
        y=[y + daily_safe_margin for y in daily_forecast_raw] + [max(0, y - daily_safe_margin) for y in daily_forecast_raw][::-1],
        fill='toself', fillcolor='rgba(231, 76, 60, 0.1)',
        line=dict(color='rgba(255,255,255,0)'), name='Safety Stock Margin zone', showlegend=True
    ), row=2, col=1)

    # Styling Layout Dashboard
    fig.update_layout(height=750, template="plotly_white", showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="Quantity (Units)", row=2, col=1)
    fig.update_xaxes(title_text="Timeline Perencanaan (Tanggal)", row=2, col=1)
    
    # Tampilkan dashboard
    st.plotly_chart(fig, use_container_width=True)

    # --- 6. SUMMARY BAR CHART (GLOBAL COMPARISON) ---
    st.markdown("---")
    st.subheader(f"📦 Perbandingan Pengadaan Global ({target_hz})")
    
    current_hz_data = long_term_report[long_term_report['Horizon'] == target_hz]
    fig_bar = px.bar(current_hz_data, x="Kategori", y=["Est. Demand", "Safety Stock"],
                     barmode="group", color_discrete_sequence=['#3498db', '#e74c3c'])
    fig_bar.update_layout(template="plotly_white")
    st.plotly_chart(fig_bar, use_container_width=True)

# Menjalankan aplikasi
if __name__ == "__main__":
    run()