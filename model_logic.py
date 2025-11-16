import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import os
import plotly.express as px
import plotly.io as pio

DATA_PATH = "DATA/Data penangkapan ikan karangantu.xlsx"

def load_and_train_model():
    """
    Melatih model Ridge dan mengembalikan komponen yang diperlukan.
    """
    global training_columns # Simpan kolom untuk prediksi
    model_ridge = None
    training_columns = []
    alat_tangkap_list = []
    
    try:
        if not os.path.exists(DATA_PATH):
            print(f"Warning: File {DATA_PATH} tidak ditemukan.")
            return None, [], []

        df = pd.read_excel(DATA_PATH) # <-- Perbaikan dari .csv ke .excel
        
        # Preprocessing (sama seperti kode Kakak)
        for col in ['Produksi_total (kg)', 'Effort (trip)']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = 1000 if 'Produksi' in col else 100

        if 'CPUE' in df.columns and 'Alat tangkap' in df.columns:
            df_cleaned = df.dropna(subset=['CPUE']).copy()
            
            alat_tangkap_list = df_cleaned['Alat tangkap'].unique().tolist()
            
            df_encoded = pd.get_dummies(df_cleaned, columns=['Alat tangkap'])
            features = ['Effort (trip)'] + [c for c in df_encoded.columns if 'Alat tangkap' in c]
            
            # Pastikan semua fitur ada
            if 'Effort (trip)' not in df_encoded.columns:
                 df_encoded['Effort (trip)'] = 100 # Default jika hilang

            X = df_encoded[features]
            y = df_encoded['CPUE']
            
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
            training_columns = X_train.columns
            
            model_ridge = Ridge(alpha=1.0)
            model_ridge.fit(X_train, y_train)
            print("Model berhasil dilatih!")
            
    except Exception as e:
        print(f"Error training model: {e}")
        
    return model_ridge, training_columns, alat_tangkap_list

def predict_cpue(model, columns, effort, alat):
    """
    Membuat prediksi berdasarkan input user.
    """
    try:
        # Prepare input dataframe
        input_df = pd.DataFrame({
            'Alat tangkap': [alat],
            'Effort (trip)': [effort]
        })
        
        input_encoded = pd.get_dummies(input_df, columns=['Alat tangkap'])
        processed = input_encoded.reindex(columns=columns, fill_value=0)
        
        # Pastikan Effort (trip) ada dan di posisi yang benar
        if 'Effort (trip)' in columns:
            processed['Effort (trip)'] = effort
        
        # Reorder jika perlu untuk memastikan urutan sama
        processed = processed[columns] 
        
        pred_cpue = model.predict(processed)[0]
        pred_prod = pred_cpue * effort
        
        return {
            'cpue': round(pred_cpue, 2),
            'produksi': round(pred_prod, 2)
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

def get_dashboard_summary():
    """
    Mengambil data ringkasan untuk dashboard.
    """
    try:
        df = pd.read_excel(DATA_PATH)
        total_prod = df['Produksi_total (kg)'].sum()
        avg_cpue = df['CPUE'].mean()
        total_trip = df['Effort (trip)'].sum()
        return {
            'total_produksi': f"{total_prod:,.0f} kg",
            'avg_cpue': f"{avg_cpue:,.2f}",
            'total_trip': f"{total_trip:,.0f} trip"
        }
    except:
        return {
            'total_produksi': 'N/A',
            'avg_cpue': 'N/A',
            'total_trip': 'N/A'
        }

def get_analysis_plots():
    """
    Membuat plot Plotly untuk halaman analisis dan mengembalikannya sebagai JSON.
    """
    try:
        df = pd.read_excel(DATA_PATH)
        
        # Plot 1: Produksi vs Effort
        fig1 = px.scatter(df, x='Effort (trip)', y='Produksi_total (kg)', 
                          color='Alat tangkap', title='Produksi vs Effort berdasarkan Alat Tangkap',
                          template='plotly_dark')
        
        # Plot 2: Rata-rata CPUE per Alat Tangkap
        df_grouped = df.groupby('Alat tangkap')['CPUE'].mean().reset_index()
        fig2 = px.bar(df_grouped, x='Alat tangkap', y='CPUE', 
                      title='Rata-rata CPUE per Alat Tangkap',
                      template='plotly_dark')
        
        # Ubah ke JSON
        plot1_json = pio.to_json(fig1)
        plot2_json = pio.to_json(fig2)
        
        # Data tabel
        df_head = df.head().to_html(classes='data-table', border=0)
        
        return plot1_json, plot2_json, df_head
    
    except Exception as e:
        print(f"Error creating plots: {e}")
        return None, None, "<p>Gagal memuat data.</p>"