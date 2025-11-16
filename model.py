import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import os # Diperlukan untuk mengecek path

def main():
    # --- PERBAIKAN 1: Ganti Judul ---
    st.markdown('<h1 class="dashboard-title">🎣 Model Prediksi Sistem Stok Ikan</h1>', unsafe_allow_html=True)
    st.write("Prediksi CPUE dan Produksi Total berdasarkan data penangkapan ikan di Karangantu.")

    # --- PERBAIKAN 2: Bungkus Blok Loading Data ---
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("1. Memuat Data Latih")
    
    # --- PERBAIKAN 3: Error Kritis Path File ---
    # Path lokal D:\... diganti ke path relatif
    # dan pd.read_excel diganti ke pd.read_csv sesuai file list
    
    file_path = "DATA/Data penangkapan ikan karangantu.xlsx"
    
    # -----------------------------
    # 1️⃣ Load Data
    # -----------------------------
    try:
        if not os.path.exists(file_path):
            st.error(f"File data tidak ditemukan di path: {file_path}")
            st.info("Pastikan file 'Data penangkapan ikan karangantu.xlsx - Sheet1.csv' ada di dalam folder 'DATA'.")
            st.markdown('</div>', unsafe_allow_html=True) # Tutup container jika error
            st.stop()
            
        # Ganti dari read_excel ke read_csv
        df = pd.read_csv(file_path)
        st.success(f"Data '{file_path}' berhasil dimuat.")
        
    except Exception as e:
        st.error(f"Gagal memuat file CSV: {e}")
        st.markdown('</div>', unsafe_allow_html=True) # Tutup container jika error
        st.stop() 
    
    st.markdown('</div>', unsafe_allow_html=True) # Tutup container loading data

    # -----------------------------
    # 2️⃣ Pembersihan & Pra-pemrosesan Data (berjalan di latar)
    # -----------------------------
    for col in ['Produksi_total (kg)', 'Effort (trip)']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
        else:
            st.warning(f"Kolom '{col}' tidak ditemukan di data. Menggunakan nilai default.")
            # Buat kolom dummy jika tidak ada agar kode tidak error
            if col == 'Produksi_total (kg)': df[col] = 1000
            if col == 'Effort (trip)': df[col] = 100

    if 'CPUE' not in df.columns:
        st.error("Kolom 'CPUE' tidak ada di data. Model tidak bisa dilatih.")
        st.stop()
        
    df_cleaned = df.dropna(subset=['CPUE']).copy()

    # Tangani outlier CPUE (IQR method)
    Q1 = df_cleaned['CPUE'].quantile(0.25)
    Q3 = df_cleaned['CPUE'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_cleaned['CPUE'] = np.where(
        df_cleaned['CPUE'] > upper_bound, upper_bound,
        np.where(df_cleaned['CPUE'] < lower_bound, lower_bound, df_cleaned['CPUE'])
    )

    # One-hot encoding kolom "Alat tangkap"
    if 'Alat tangkap' not in df_cleaned.columns:
        st.error("Kolom 'Alat tangkap' tidak ada di data. Model tidak bisa dilatih.")
        st.stop()
        
    df_encoded = pd.get_dummies(df_cleaned, columns=['Alat tangkap'])
    features_encoded = ['Effort (trip)'] + [col for col in df_encoded.columns if 'Alat tangkap' in col]
    
    X = df_encoded[features_encoded]
    y = df_encoded['CPUE']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    training_columns = X_train.columns

    # -----------------------------
    # 3️⃣ Pelatihan Model Ridge Regression (di latar belakang)
    # -----------------------------
    best_params_ridge = {"alpha": 1.0} 
    model = Ridge(**best_params_ridge)
    model.fit(X_train, y_train)

    # -----------------------------
    # 4️⃣ Input Form Streamlit
    # --- PERBAIKAN 4: Bungkus Blok Form Input ---
    # -----------------------------
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("🧮 Masukkan Parameter Prediksi")

    col1, col2 = st.columns(2)
    with col1:
        tahun = st.number_input("Tahun", min_value=2000, max_value=2100, value=2024)
        effort = st.number_input("Effort (trip)", min_value=0.0, value=100.0, step=10.0)
    with col2:
        alat_tangkap = st.selectbox("Alat tangkap", df_cleaned['Alat tangkap'].unique())

    # -----------------------------
    # 5️⃣ Tombol Prediksi
    # -----------------------------
    if st.button("🔍 Prediksi CPUE & Produksi"):
        input_df = pd.DataFrame({
            'Tahun': [tahun],
            'Alat tangkap': [alat_tangkap],
            'Effort (trip)': [effort]
        })

        input_encoded = pd.get_dummies(input_df, columns=['Alat tangkap'])
        processed_input = input_encoded.reindex(columns=training_columns, fill_value=0)
        
        # Pastikan kolom 'Effort (trip)' ada
        if 'Effort (trip)' not in processed_input.columns:
            processed_input['Effort (trip)'] = effort
            
        # Pastikan urutan kolom sama
        processed_input = processed_input[training_columns]

        predicted_cpue = model.predict(processed_input)[0]
        predicted_production = predicted_cpue * effort

        st.success(f"**Prediksi CPUE:** {predicted_cpue:.2f}")
        st.info(f"**Prediksi Produksi Total (kg):** {predicted_production:.2f}")

    # -----------------------------
    # 6️⃣ Info tambahan
    # -----------------------------
    st.caption("Model Ridge Regression dilatih ulang dari data penangkapan ikan Karangantu.")
    
    st.markdown('</div>', unsafe_allow_html=True) # Tutup container form