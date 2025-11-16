"""
market_foresight.py

Lightweight Streamlit page that exposes a `main()` function and uses
the `MarketForesight` class implemented in `market_model.py`.

This file was corrupted and has been replaced with a minimal, safe UI
that supports sample-data generation, CSV/Excel upload, selecting a
fish type, running ARIMA predictions (via MarketForesight), showing
results, plotting, and exporting CSV results.
"""

import io
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np # Ditambahkan untuk dummy class

# Coba impor, jika gagal (saat dijalankan sendiri), buat class dummy
try:
    from market_model import MarketForesight
except ImportError:
    # Buat class placeholder jika market_model.py tidak ditemukan
    # Ini agar file-nya tetap bisa 'dijalankan' untuk testing
    st.warning("Mode testing: market_model.py tidak ditemukan. Menggunakan placeholder.")
    class MarketForesight:
        def __init__(self):
            self.data = None
            self.processed_data = None
        def create_sample_data(self, n_months=36):
            dates = pd.date_range(end=pd.Timestamp.now(), periods=n_months, freq='MS')
            data = []
            for fish in ['Cakalang', 'Kembung', 'Tongkol']:
                price = np.random.randint(15000, 30000)
                for date in dates:
                    price += np.random.randint(-1000, 1000)
                    data.append([date, fish, price])
            return pd.DataFrame(data, columns=['Tanggal', 'Jenis_Ikan', 'Harga_Per_Kg'])
        def preprocess_data(self, fish_type):
            if self.data is not None:
                df_fish = self.data[self.data['Jenis_Ikan'] == fish_type].copy()
                df_fish['Tanggal'] = pd.to_datetime(df_fish['Tanggal'])
                df_fish = df_fish.set_index('Tanggal').resample('MS')['Harga_Per_Kg'].mean().fillna(method='ffill')
                self.processed_data = df_fish.to_frame()
        def train_arima(self): return "Model (dummy) dilatih"
        def predict_future(self, periods, method="arima"):
            if self.processed_data is None or self.processed_data.empty:
                return pd.DataFrame()
            last_date = self.processed_data.index.max()
            pred_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='MS')
            last_price = self.processed_data['Harga_Per_Kg'].iloc[-1]
            preds = []
            for i in range(periods):
                price = last_price + np.random.randint(-500, 500) * (i+1)
                preds.append({
                    "Tanggal": pred_dates[i],
                    "Prediksi": price,
                    "Batas_Bawah": price - 1000,
                    "Batas_Atas": price + 1000
                })
            return pd.DataFrame(preds)
        def generate_alerts(self, preds, threshold_percent, method='arima'):
            return pd.DataFrame()


def _ensure_forecaster():
    if "market_forecaster" not in st.session_state:
        st.session_state["market_forecaster"] = MarketForesight()
    return st.session_state["market_forecaster"]


def _plot_predictions(df, hist_data=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot historical data if available
    if hist_data is not None and not hist_data.empty:
        # Ganti nama kolom jika perlu, sesuaikan dengan 'Harga_Per_Kg'
        if 'Harga_Per_Kg' in hist_data.columns:
            hist_values_col = 'Harga_Per_Kg'
        else:
            # Fallback jika nama kolom berbeda
            hist_values_col = hist_data.columns[0] 
            
        hist_dates = hist_data.index
        hist_values = hist_data[hist_values_col]
        ax.plot(hist_dates, hist_values, 'b-', label='Data Historis', alpha=0.7)
    
    # Plot predictions with confidence intervals if available
    pred_dates = pd.to_datetime(df["Tanggal"])
    ax.plot(pred_dates, df["Prediksi"], 'r-', marker='o', label='Prediksi ARIMA')
    
    if "Batas_Bawah" in df.columns and "Batas_Atas" in df.columns:
        ax.fill_between(pred_dates, 
                        df["Batas_Bawah"], 
                        df["Batas_Atas"], 
                        color='red', 
                        alpha=0.1, 
                        label='Interval Kepercayaan')
    
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga (Rp)")
    ax.set_title("Prediksi Harga ARIMA")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def main():
    # --- PERBAIKAN 1: Hapus st.set_page_config() ---
    # st.set_page_config(page_title="Market Foresight", layout="wide") # <-- HAPUS
    
    # --- PERBAIKAN 2: Ganti Judul ---
    st.markdown('<h1 class="dashboard-title">Prediksi Harga Ikan (ARIMA)</h1>', unsafe_allow_html=True)
    
    # --- PERBAIKAN 3: Bungkus Info Expander ---
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    # Informasi model
    with st.expander("ℹ️ Informasi Model ARIMA"):
        st.markdown("""
        **Model ARIMA (Autoregressive Integrated Moving Average)**
        - Metode statistik untuk analisis dan prediksi data time series
        - Mempertimbangkan tren, musiman, dan pola historis
        - Komponen:
            - AR (Autoregressive): Menggunakan nilai historis
            - I (Integrated): Transformasi diferensiasi untuk stasionaritas
            - MA (Moving Average): Memperhitungkan error prediksi sebelumnya
        """)
    st.markdown('</div>', unsafe_allow_html=True) # Tutup container info

    forecaster = _ensure_forecaster()
    
    # --- Bagian Sidebar (Biarkan, sudah benar) ---
    st.sidebar.header("Data & Pengaturan")
    uploaded = st.sidebar.file_uploader("Upload CSV atau Excel (opsional)", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            forecaster.data = df
            st.sidebar.success(f"Data dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
        except Exception as e:
            st.sidebar.error(f"Gagal membaca file: {e}")

    if st.sidebar.button("Buat Data Sampel (36 bulan)"):
        sample = forecaster.create_sample_data(n_months=36)
        forecaster.data = sample
        st.sidebar.success("Data sampel dibuat dan dimuat ke session.")
    # --- Akhir Bagian Sidebar ---

    if forecaster.data is None:
        st.info("Tidak ada data. Upload file atau buat data sampel dari sidebar.")
        return

    # Select fish types available in the data
    if "Jenis_Ikan" in forecaster.data.columns:
        fish_options = sorted(forecaster.data["Jenis_Ikan"].dropna().unique())
    else:
        st.error("Kolom 'Jenis_Ikan' tidak ditemukan di data. Pastikan format file sesuai.")
        return

    # --- PERBAIKAN 4: Bungkus Form Input ---
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("⚙️ Pengaturan Prediksi")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_fish = st.selectbox("Pilih Jenis Ikan", fish_options)
    with col2:
        periods = st.number_input("Periode prediksi (bulan)", min_value=1, max_value=60, value=6)
    with col3:
        threshold = st.number_input("Ambang alert (%) untuk perubahan", min_value=1, max_value=100, value=10)

    run_button = st.button("Jalankan Prediksi ARIMA", type="primary")
    st.markdown('</div>', unsafe_allow_html=True) # Tutup container input

    # --- PERBAIKAN 5: Bungkus Blok Hasil ---
    if run_button:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        with st.spinner("Memproses prediksi..."):
            try:
                # Prepare data and generate predictions
                forecaster.preprocess_data(fish_type=selected_fish)
                if forecaster.processed_data is None or forecaster.processed_data.empty:
                    st.error(f"Tidak ada data historis untuk ikan '{selected_fish}'. Pilih ikan lain.")
                    st.markdown('</div>', unsafe_allow_html=True) # Tutup container jika error
                    return

                _ = forecaster.train_arima()
                # Generate predictions
                preds = forecaster.predict_future(periods=periods, method="arima")

                if preds.empty:
                    st.error("Gagal menghasilkan prediksi. Cek data input.")
                else:
                    st.subheader("Hasil Prediksi")
                    preds_display = preds.copy()
                    preds_display["Tanggal"] = pd.to_datetime(preds_display["Tanggal"]).dt.strftime("%Y-%m-%d")
                    st.dataframe(preds_display)

                    # Plot both historical and predicted data
                    fig = _plot_predictions(preds, forecaster.processed_data)
                    st.pyplot(fig)

                    alerts = forecaster.generate_alerts(preds, threshold_percent=threshold, method='arima')
                    if not alerts.empty:
                        st.warning("Perhatian: Terdapat alert berikut berdasarkan ambang yang ditetapkan:")
                        st.dataframe(alerts)
                    else:
                        st.success("Tidak ada alert signifikan pada prediksi.")

                    csv_buf = io.StringIO()
                    preds.to_csv(csv_buf, index=False)
                    st.download_button("Download Hasil Prediksi (CSV)", data=csv_buf.getvalue(), file_name="prediksi_harga.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True) # Tutup container hasil


if __name__ == "__main__":
    main()