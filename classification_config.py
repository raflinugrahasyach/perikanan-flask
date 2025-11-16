import flask
from flask import Flask, render_template, request, flash, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import os
import io
import base64
import matplotlib
matplotlib.use('Agg') # WAJIB
import matplotlib.pyplot as plt
from scipy import stats
from io import StringIO, BytesIO
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
# import tensorflow as tf # <--- DIHAPUS
import tflite_runtime.interpreter as tflite # <--- IMPORT BARU
import joblib
from werkzeug.utils import secure_filename
import threading
import firebase_admin
from firebase_admin import credentials, db

# --- Konfigurasi Upload ---
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
# -------------------------

# --- KELAS CUSTOM TF LAMA DIHAPUS ---
# class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
#     ...
# (Seluruh kelas ini dihapus karena tflite-runtime tidak membutuhkannya)
# --- AKHIR PENGHAPUSAN ---

try:
    from classification_config import (
        MODEL_PATHS, IMAGE_SETTINGS, CONFIDENCE_THRESHOLDS, 
        SPECIES_INFO, CONSERVATION_STATUS, UI_CONFIG
    )
except ImportError:
    # (Semua placeholder config Anda tetap sama)
    print("Warning: classification_config.py tidak ditemukan. Menggunakan placeholder.")
    MODEL_PATHS = {'species_model': 'Model_Klasifikasi/model1.tflite', 'conservation_model': 'Model_Klasifikasi/model2.tflite', 'species_labels': 'Model_Klasifikasi/labels1.txt', 'conservation_labels': 'Model_Klasifikasi/labels2.txt'}
    IMAGE_SETTINGS = {'target_size': (224, 224), 'color_mode': 'RGB', 'normalize': True, 'data_type': 'float32'}
    CONFIDENCE_THRESHOLDS = {'high': 0.8, 'medium': 0.5}
    SPECIES_INFO = {'Fish': {'description': 'Informasi dummy', 'habitat': 'Dummy', 'importance': 'Dummy'}}
    CONSERVATION_STATUS = {'Masih Lestari': {'color': '#2ecc71', 'icon': '🟢', 'urgency': 'SUSTAINABLE', 'description': 'Dummy', 'actions': ['Dummy'], 'legal_basis': 'Tersedia'}, 'Dilindungi': {'color': '#e74c3c', 'icon': '🔴', 'urgency': 'CRITICAL', 'description': 'Dummy', 'actions': ['Dummy'], 'legal_basis': 'Dummy'}, 'Bukan Biota Laut': {'color': '#95a5a6', 'icon': '⚪', 'urgency': 'NOT_APPLICABLE', 'description': 'Dummy', 'actions': ['Dummy'], 'legal_basis': 'Dummy'}}
    UI_CONFIG = {'supported_formats': ['png', 'jpg', 'jpeg', 'bmp']}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey_dev_12345'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Tetap pertahankan batas 16MB

# --- Inisialisasi Firebase Admin SDK ---
try:
    if not firebase_admin._apps:
        # Pastikan file 'firebase-service-account.json' ada di folder utama
        cred = credentials.Certificate('firebase-service-account.json') 
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://perikanan-73237-default-rtdb.firebaseio.com' # GANTI JIKA URL ANDA BEDA
        })
    print("Koneksi Firebase Admin SDK berhasil.")
except Exception as e:
    print(f"Error inisialisasi Firebase Admin: {e}")
    print("PERINGATAN: Fitur Peta tidak akan berfungsi.")
# --- Akhir Inisialisasi Firebase ---


# --- (Semua fungsi helper /analysis, /dashboard, /market TETAP SAMA) ---
# ... (def parse_analysis_file ... )
# ... (def dash_load_and_clean_data ... )
# ... (class MarketForesight ... )
# (Kode-kode ini tidak perlu diubah, jadi saya sembunyikan agar ringkas)
def parse_analysis_file(uploaded_file):
    try:
        content = uploaded_file.read().decode('utf-8')
        df = pd.read_csv(StringIO(content))
        required_columns = ['Tahun', 'Alat_Tangkap', 'Produksi', 'Upaya']
        if not all(col in df.columns for col in required_columns):
            return None, "File harus memiliki kolom: Tahun, Alat_Tangkap, Produksi, Upaya"
        gears = sorted(df['Alat_Tangkap'].unique().tolist())
        years = sorted(df['Tahun'].unique().tolist())
        template = pd.DataFrame(index=pd.MultiIndex.from_product([years, gears], names=['Tahun', 'Alat_Tangkap'])).reset_index()
        df_prod = pd.merge(template, df[['Tahun', 'Alat_Tangkap', 'Produksi']], on=['Tahun', 'Alat_Tangkap'], how='left').fillna(0)
        df_effort = pd.merge(template, df[['Tahun', 'Alat_Tangkap', 'Upaya']], on=['Tahun', 'Alat_Tangkap'], how='left').fillna(0)
        df_production = df_prod.pivot(index='Tahun', columns='Alat_Tangkap', values='Produksi').reset_index()
        df_effort = df_effort.pivot(index='Tahun', columns='Alat_Tangkap', values='Upaya').reset_index()
        df_production['Jumlah'] = df_production[gears].sum(axis=1)
        df_effort['Jumlah'] = df_effort[gears].sum(axis=1)
        return {'production': df_production, 'effort': df_effort, 'gears': gears, 'years': years, 'display_names': [g.replace('_', ' ') for g in gears]}, "Success"
    except Exception as e:
        return None, f"Error membaca file: {str(e)}"
def hitung_cpue(p, u, g): return pd.DataFrame([{'Tahun': y, **{gr: (p.loc[p['Tahun'] == y, gr].values[0] / u.loc[u['Tahun'] == y, gr].values[0]) if u.loc[u['Tahun'] == y, gr].values[0] > 0 else 0 for gr in g}, 'Jumlah': sum([(p.loc[p['Tahun'] == y, gr].values[0] / u.loc[u['Tahun'] == y, gr].values[0]) if u.loc[u['Tahun'] == y, gr].values[0] > 0 else 0 for gr in g])} for y in p['Tahun'].values])
def hitung_fpi_per_tahun(c, g, s): return pd.DataFrame([{'Tahun': y, **{gr: (c.loc[c['Tahun'] == y, gr].values[0] / c.loc[c['Tahun'] == y, s].values[0]) if c.loc[c['Tahun'] == y, s].values[0] > 0 else 0 for gr in g}, 'Jumlah': sum([(c.loc[c['Tahun'] == y, gr].values[0] / c.loc[c['Tahun'] == y, s].values[0]) if c.loc[c['Tahun'] == y, s].values[0] > 0 else 0 for gr in g])} for y in c['Tahun'].values])
def hitung_upaya_standar(u, f, g): return pd.DataFrame([{'Tahun': y, **{gr: u.loc[u['Tahun'] == y, gr].values[0] * f.loc[f['Tahun'] == y, gr].values[0] for gr in g}, 'Jumlah': sum([u.loc[u['Tahun'] == y, gr].values[0] * f.loc[f['Tahun'] == y, gr].values[0] for gr in g])} for y in u['Tahun'].values])
def hitung_cpue_standar(p, s): return pd.DataFrame([{'Tahun': y, 'CPUE_Standar_Total': (p.loc[p['Tahun'] == y, 'Jumlah'].values[0] / s.loc[s['Tahun'] == y, 'Jumlah'].values[0]) if s.loc[s['Tahun'] == y, 'Jumlah'].values[0] > 0 else 0, 'Ln_CPUE': np.log((p.loc[p['Tahun'] == y, 'Jumlah'].values[0] / s.loc[s['Tahun'] == y, 'Jumlah'].values[0])) if s.loc[s['Tahun'] == y, 'Jumlah'].values[0] > 0 and p.loc[p['Tahun'] == y, 'Jumlah'].values[0] > 0 else 0} for y in p['Tahun'].values])
def analisis_msy_schaefer(s, c):
    if len(s) < 2: return None
    slope, intercept, r, p, std_err = stats.linregress(s, c); a=intercept; b=slope
    if b >= 0: return {'success': False, 'error': 'Slope (b) harus negatif'}
    F_MSY = -a / (2 * b); C_MSY = -(a ** 2) / (4 * b)
    return {'a': a, 'b': b, 'r_squared': r**2, 'p_value': p, 'std_err': std_err, 'F_MSY': F_MSY, 'C_MSY': C_MSY, 'success': True}
def plot_to_base64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')
def create_plot_cpue_vs_effort(e, c, r):
    fig, ax = plt.subplots(figsize=(8, 5)); ax.scatter(e, c, s=80, label='Data Aktual')
    if r and r.get('success'):
        er = np.linspace(min(e), max(e) * 1.2, 100); cp = r['a'] + r['b'] * er
        ax.plot(er, cp, 'red', linewidth=2, label='Garis Regresi (Schaefer)'); ax.axvline(r['F_MSY'], color='green', linestyle='--', label=f'F_MSY ({r["F_MSY"]:.0f})')
    ax.set_title('CPUE vs Effort (Model Schaefer)'); ax.set_xlabel('Upaya (Effort)'); ax.set_ylabel('CPUE')
    ax.legend(); ax.grid(True, linestyle='--', alpha=0.6); return plot_to_base64(fig)
def create_plot_production_vs_effort(e, p, r):
    fig, ax = plt.subplots(figsize=(8, 5)); ax.scatter(e, p, s=80, label='Data Aktual')
    if r and r.get('success'):
        erp = np.linspace(0, max(e) * 1.5, 100); cp = r['a'] * erp + r['b'] * (erp ** 2)
        ax.plot(erp, cp, 'purple', linewidth=3, label='Kurva Produksi'); ax.axvline(r['F_MSY'], color='green', linestyle='--', label=f'F_MSY ({r["F_MSY"]:.0f})'); ax.axhline(r['C_MSY'], color='orange', linestyle='--', label=f'C_MSY ({r["C_MSY"]:.0f})')
    ax.set_title('Produksi vs Effort'); ax.set_xlabel('Upaya (Effort)'); ax.set_ylabel('Produksi (Catch)')
    ax.legend(); ax.grid(True, linestyle='--', alpha=0.6); return plot_to_base64(fig)
def dash_load_and_clean_data(uploaded_file):
    content = uploaded_file.read().decode('utf-8')
    try:
        df = pd.read_csv(StringIO(content), sep='\t')
        if len(df.columns) < 2: df = pd.read_csv(StringIO(content))
    except: df = pd.read_csv(StringIO(content))
    df.columns = df.columns.str.strip()
    df['Volume Produksi (kg)'] = pd.to_numeric(df['Volume Produksi (kg)'].astype(str).replace(r'[\.,]', '', regex=True).replace(['-', '', ' ', 'nan'], '0'), errors='coerce').fillna(0)
    df['Jenis Ikan'] = df['Jenis Ikan'].astype(str).str.strip().str.replace('"', '')
    df = df.drop_duplicates(subset=['Tahun', 'Bulan', 'Jenis Ikan'])
    bulan_order = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
    df['Bulan_Num'] = df['Bulan'].map({b: i+1 for i, b in enumerate(bulan_order)})
    df['Bulan_Num'] = df['Bulan_Num'].fillna(1)
    df['Tanggal'] = pd.to_datetime(df['Tahun'].astype(str) + '-' + df['Bulan_Num'].astype(int).astype(str) + '-01', format='%Y-%m-%d', errors='coerce')
    df = df.dropna(subset=['Tanggal'])
    return df
def dash_create_kpi_cards(df):
    if df.empty: return {}
    total_produksi = df['Volume Produksi (kg)'].sum()
    monthly_sum = df.groupby(['Tahun', 'Bulan'])['Volume Produksi (kg)'].sum()
    rata_bulanan = monthly_sum.mean() if len(monthly_sum) > 0 else 0
    jumlah_jenis = df['Jenis Ikan'].nunique()
    yearly_sum = df.groupby('Tahun')['Volume Produksi (kg)'].sum()
    tahun_terbaik = "N/A"
    if len(yearly_sum) > 0 and yearly_sum.max() > 0:
        tahun_terbaik = yearly_sum.idxmax()
    return {"total_produksi_m": f"{total_produksi/1_000_000:.2f}M", "rata_bulanan_k": f"{rata_bulanan/1000:.1f}K", "jumlah_jenis": f"{jumlah_jenis}", "tahun_terbaik": f"{tahun_terbaik}"}
def plot_to_html(fig): return fig.to_html(full_html=False, include_plotlyjs='cdn')
def dash_plot_trend_tahunan(df):
    if df.empty: return None
    df_bulanan = df.groupby('Tanggal')['Volume Produksi (kg)'].sum().reset_index()
    fig = px.line(df_bulanan, x='Tanggal', y='Volume Produksi (kg)', title='📈 Tren Produksi Ikan Bulanan')
    fig.update_traces(line_color='#1E88E5', line_width=3); fig.update_layout(hovermode='x unified', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=400); return plot_to_html(fig)
def dash_plot_top_species(df, n=10):
    if df.empty: return None
    top_species = df.groupby('Jenis Ikan')['Volume Produksi (kg)'].sum().nlargest(n).reset_index()
    if top_species.empty: return None
    fig = px.bar(top_species, x='Volume Produksi (kg)', y='Jenis Ikan', orientation='h', title=f'🏆 Top {n} Jenis Ikan', color='Volume Produksi (kg)', color_continuous_scale='Blues')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500, showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'); return plot_to_html(fig)
class MarketForesight:
    def __init__(self): self.data = None; self.processed_data = None; self.model = None
    def load_data_from_path(self, file_path):
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.txt'):
                self.data = pd.read_csv(file_path, sep='\t')
            elif file_path.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(file_path)
            else:
                raise ValueError("Format file tidak didukung")
        except Exception as e:
            print(f"Error loading market data from path: {e}"); raise e
    def get_fish_options(self):
        if self.data is None or "Jenis_Ikan" not in self.data.columns: return []
        return sorted(self.data["Jenis_Ikan"].dropna().unique())
    def preprocess_data(self, fish_type):
        if self.data is None: return False
        df_fish = self.data[self.data['Jenis_Ikan'] == fish_type].copy()
        if 'Nilai_Produksi_Rp' not in df_fish.columns or 'Volume_Produksi_Kg' not in df_fish.columns: 
            return False
        df_fish['Nilai_Produksi_Rp'] = pd.to_numeric(df_fish['Nilai_Produksi_Rp'].astype(str).replace(r'[\.,]', '', regex=True), errors='coerce')
        df_fish['Volume_Produksi_Kg'] = pd.to_numeric(df_fish['Volume_Produksi_Kg'].astype(str).replace(r'[\.,]', '', regex=True), errors='coerce')
        df_fish = df_fish.dropna(subset=['Nilai_Produksi_Rp', 'Volume_Produksi_Kg'])
        df_fish = df_fish[df_fish['Volume_Produksi_Kg'] != 0]
        df_fish['Harga_Per_Kg'] = df_fish['Nilai_Produksi_Rp'] / df_fish['Volume_Produksi_Kg']
        df_fish['Tanggal'] = pd.to_datetime(df_fish['Tanggal']); df_fish = df_fish.set_index('Tanggal').resample('MS')['Harga_Per_Kg'].mean().ffill()
        self.processed_data = df_fish.to_frame(); return True
    def train_arima(self): print("Menjalankan training ARIMA (placeholder)...")
    def predict_future(self, periods):
        if self.processed_data is None or self.processed_data.empty: return pd.DataFrame()
        last_date = self.processed_data.index.max(); pred_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='MS')
        last_price = self.processed_data['Harga_Per_Kg'].iloc[-1]; preds = []
        for i in range(periods):
            price = last_price + np.random.randint(-500, 500) * (i+1)
            preds.append({"Tanggal": pred_dates[i], "Prediksi": price, "Batas_Bawah": price - 1000, "Batas_Atas": price + 1000})
        return pd.DataFrame(preds)


# --- MULAI PERUBAHAN BESAR ---
# Helper 4: Fungsi untuk /classification (SEKARANG MENGGUNAKAN TFLITE)
class MarineClassifier:
    def __init__(self): 
        # Interpreter TFLite
        self.interpreter1 = None
        self.interpreter2 = None
        # Label
        self.labels1 = []
        self.labels2 = []
        # Detail Input/Output TFLite
        self.input_details1 = None
        self.output_details1 = None
        self.input_details2 = None
        self.output_details2 = None
    
    def load_models_lazy(self):
        if self.interpreter1 and self.interpreter2: 
            return True
        
        print("--- MEMUAT MODEL TFLITE (Lazy Load) ---")
        try:
            m1_path = MODEL_PATHS['species_model']
            m2_path = MODEL_PATHS['conservation_model']
            l1_path = MODEL_PATHS['species_labels']
            l2_path = MODEL_PATHS['conservation_labels']

            if not all(os.path.exists(p) for p in [m1_path, m2_path, l1_path, l2_path]):
                print(f"Warning: File model TFLite tidak ditemukan. Cek path: {m1_path}, {m2_path}")
                return False
            
            # Muat label (tetap sama)
            with open(l1_path, "r", encoding='utf-8') as f: self.labels1 = [line.strip() for line in f.readlines()]
            with open(l2_path, "r", encoding='utf-8') as f: self.labels2 = [line.strip() for line in f.readlines()]
            
            # --- Perubahan Logika Pemuatan ---
            # Tidak perlu custom_objects
            # Muat model 1
            self.interpreter1 = tflite.Interpreter(model_path=m1_path)
            self.interpreter1.allocate_tensors()
            self.input_details1 = self.interpreter1.get_input_details()
            self.output_details1 = self.interpreter1.get_output_details()

            # Muat model 2
            self.interpreter2 = tflite.Interpreter(model_path=m2_path)
            self.interpreter2.allocate_tensors()
            self.input_details2 = self.interpreter2.get_input_details()
            self.output_details2 = self.interpreter2.get_output_details()
            
            print("Model TFLite berhasil dimuat.")
            return True
            
        except Exception as e: 
            print(f"Error loading TFLite models: {e}"); 
            return False
            
    def preprocess_image(self, image):
        # Fungsi ini sebagian besar tetap sama
        # Pastikan ukuran dan tipe data SAMA PERSIS dengan saat konversi
        image = image.resize((224, 224))
        if image.mode != 'RGB': image = image.convert('RGB')
        
        image_array = np.array(image, dtype=np.float32) / 255.0 # Konversi ke float32
        
        # TFLite butuh (1, 224, 224, 3)
        return np.expand_dims(image_array, axis=0)
    
    def predict(self, image):
        if not self.load_models_lazy(): 
            return None, None
        
        # Pastikan gambar diproses dengan tipe data yang benar (float32)
        processed_image = self.preprocess_image(image)
        if processed_image is None: return None, None
        
        try:
            # --- Perubahan Logika Prediksi ---
            
            # Prediksi Model 1
            self.interpreter1.set_tensor(self.input_details1[0]['index'], processed_image)
            self.interpreter1.invoke()
            pred1 = self.interpreter1.get_tensor(self.output_details1[0]['index'])

            # Prediksi Model 2
            self.interpreter2.set_tensor(self.input_details2[0]['index'], processed_image)
            self.interpreter2.invoke()
            pred2 = self.interpreter2.get_tensor(self.output_details2[0]['index'])

        except Exception as e:
            print(f"Error during TFLite invoke(): {e}")
            return None, None
            
        # Logika sisa (argmax, dll) tetap sama
        c1_idx = np.argmax(pred1[0]); c2_idx = np.argmax(pred2[0])
        r1 = {'class': self.labels1[c1_idx] if c1_idx < len(self.labels1) else "Unknown", 'confidence': float(pred1[0][c1_idx])}
        r2 = {'class': self.labels2[c2_idx] if c2_idx < len(self.labels2) else "Unknown", 'confidence': float(pred2[0][c2_idx])}
        return r1, r2

# --- AKHIR PERUBAHAN BESAR ---


def get_confidence_color(confidence):
    if confidence > CONFIDENCE_THRESHOLDS['high']: return "🟢"
    elif confidence > CONFIDENCE_THRESHOLDS['medium']: return "🟡"
    else: return "🔴"
def get_conservation_info(species_class, conservation_class):
    info = {}; species_name = species_class.split()[-1] if species_class else ""
    info['species'] = SPECIES_INFO.get(species_name, {'description': "N/A", 'habitat': "N/A", 'importance': "N/A"})
    conservation_name = conservation_class.split()[-1] if conservation_class else ""
    info['conservation'] = CONSERVATION_STATUS.get(conservation_name, {"color": "#95a5a6", "icon": "❓", "urgency": "UNKNOWN", "description": "N/A", "actions": [], "legal_basis": "N/A"})
    return info

# --- Singleton Pattern (Tetap sama) ---
classifier_lock = threading.Lock()
_global_classifier = None

def get_classifier():
    global _global_classifier
    with classifier_lock:
        if _global_classifier is None:
            print("--- Menginisialisasi Singleton Classifier (TFLite) ---")
            _global_classifier = MarineClassifier()
    return _global_classifier

# --- Logika /model (Ridge) (Tetap sama) ---
DATA_PATH_MODEL = "DATA/Data penangkapan ikan karangantu.xlsx"
model_ridge = None; training_columns = None; alat_tangkap_list = []
def train_model():
    global model_ridge, training_columns, alat_tangkap_list
    try:
        alt_path = "DATA/Data penangkapan ikan karangantu.xlsx - Sheet1.csv"
        df = None
        if os.path.exists(DATA_PATH_MODEL):
            try: df = pd.read_excel(DATA_PATH_MODEL)
            except: df = pd.read_csv(DATA_PATH_MODEL)
        elif os.path.exists(alt_path):
            df = pd.read_csv(alt_path); print(f"Warning: Menggunakan file {alt_path}")
        else:
            print(f"Warning: File data untuk model tidak ditemukan."); return
        if 'CPUE' in df.columns and 'Alat tangkap' in df.columns:
            df_cleaned = df.dropna(subset=['CPUE']).copy()
            alat_tangkap_list = df_cleaned['Alat tangkap'].unique().tolist()
            df_encoded = pd.get_dummies(df_cleaned, columns=['Alat tangkap'])
            features = ['Effort (trip)'] + [c for c in df_encoded.columns if 'Alat tangkap' in c]
            X = df_encoded[features]; y = df_encoded['CPUE']
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
            training_columns = X_train.columns
            model_ridge = Ridge(alpha=1.0); model_ridge.fit(X_train, y_train)
            print("Model (/model) berhasil dilatih!")
        else:
            print("Warning: Kolom 'CPUE' atau 'Alat tangkap' tidak ada. Model /model tidak dapat dilatih.")
    except Exception as e: print(f"Error training model: {e}")
train_model()


# --- ROUTES (Semua rute tetap sama) ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('dashboard.html', error="Tidak ada file yang di-upload.")
        file = request.files['file']
        if file.filename == '':
            return render_template('dashboard.html', error="Tidak ada file yang dipilih.")
        if file:
            try:
                df = dash_load_and_clean_data(file)
                kpi_data = dash_create_kpi_cards(df)
                plot_tren_html = dash_plot_trend_tahunan(df)
                plot_top_species_html = dash_plot_top_species(df)
                return render_template('dashboard.html', kpi=kpi_data, plot_tren=plot_tren_html, plot_komposisi=plot_top_species_html)
            except Exception as e:
                return render_template('dashboard.html', error=f"Gagal memproses file: {e}")
    return render_template('dashboard.html')

@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('analysis.html', error="Tidak ada file yang di-upload.")
        file = request.files['file']
        if file.filename == '':
            return render_template('analysis.html', error="Tidak ada file yang dipilih.")
        if file:
            try:
                data, message = parse_analysis_file(file)
                if data is None: return render_template('analysis.html', error=message)
                df_production=data['production']; df_effort=data['effort']; gears_list=data['gears']
                display_names_list=data['display_names']; years_list=data['years']; standard_gear=gears_list[0] 
                df_cpue = hitung_cpue(df_production, df_effort, gears_list)
                df_fpi = hitung_fpi_per_tahun(df_cpue, gears_list, standard_gear)
                df_standard_effort = hitung_upaya_standar(df_effort, df_fpi, gears_list)
                df_standard_cpue = hitung_cpue_standar(df_production, df_standard_effort)
                effort_values = df_standard_effort['Jumlah'].values; cpue_values = df_standard_cpue['CPUE_Standar_Total'].values
                production_values = df_production['Jumlah'].values
                results = analisis_msy_schaefer(effort_values, cpue_values)
                plot_cpue_effort_b64 = create_plot_cpue_vs_effort(effort_values, cpue_values, results)
                plot_prod_effort_b64 = create_plot_production_vs_effort(effort_values, production_values, results)
                df_cpue_display = df_cpue.rename(columns=dict(zip(gears_list, display_names_list)))
                cpue_table_html = df_cpue_display.to_html(classes='dataframe', border=0, index=False)
                return render_template('analysis.html', msy_results=results, plot_cpue_effort=plot_cpue_effort_b64, plot_prod_effort=plot_prod_effort_b64, cpue_table=cpue_table_html)
            except Exception as e:
                return render_template('analysis.html', error=str(e))
    return render_template('analysis.html')

@app.route('/model', methods=['GET', 'POST'])
def predict_model():
    prediction = None; inputs = None
    if request.method == 'POST':
        try:
            tahun = float(request.form.get('tahun')); effort = float(request.form.get('effort')); alat = request.form.get('alat_tangkap')
            inputs = {'tahun': tahun, 'effort': effort, 'alat_tangkap': alat}
            if model_ridge:
                input_df = pd.DataFrame({'Tahun': [tahun], 'Alat tangkap': [alat], 'Effort (trip)': [effort]})
                input_encoded = pd.get_dummies(input_df, columns=['Alat tangkap'])
                processed = input_encoded.reindex(columns=training_columns, fill_value=0)
                if 'Effort (trip)' not in processed.columns: 
                    processed['Effort (trip)'] = effort
                processed = processed[training_columns]
                pred_cpue = model_ridge.predict(processed)[0]
                pred_prod = pred_cpue * effort
                prediction = {'cpue': pred_cpue, 'produksi': pred_prod}
            else: 
                print("Model /model tidak dilatih, prediksi dibatalkan.")
                prediction = {'cpue': 0, 'produksi': 0}
        except Exception as e: 
            print(f"Prediction error: {e}")
            prediction = {'cpue': 0, 'produksi': 0}
    return render_template('model.html', alat_list=alat_tangkap_list, prediction=prediction, inputs=inputs)

@app.route('/market', methods=['GET', 'POST'])
def market():
    fish_options = flask.session.get('fish_options', [])
    plot_html = None
    selected_fish = None
    error = None
    uploaded_filename = flask.session.get('market_file', None) 

    try:
        if request.method == 'POST':
            action = request.form.get('action')
            file = request.files.get('file')

            if action == 'upload':
                if not file or file.filename == '':
                    error = "Tidak ada file yang dipilih."
                else:
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(file_path) 
                    
                    forecaster = MarketForesight()
                    forecaster.load_data_from_path(file_path) 
                    fish_options = forecaster.get_fish_options()
                    
                    flask.session['market_file'] = filename 
                    flask.session['fish_options'] = fish_options
                    
                    if not fish_options:
                        error = "Gagal memuat data. Periksa format file: 'Jenis_Ikan' tidak ditemukan."
            
            elif action == 'analyze':
                if not uploaded_filename:
                    error = "Sesi data tidak ditemukan. Harap upload file terlebih dahulu."
                else:
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)
                    if not os.path.exists(file_path):
                        error = "File data tidak ditemukan di server. Harap upload ulang."
                        flask.session.pop('market_file', None)
                        flask.session.pop('fish_options', None)
                    else:
                        forecaster = MarketForesight()
                        forecaster.load_data_from_path(file_path)
                        fish_options = forecaster.get_fish_options() 
                        
                        selected_fish = request.form.get('fish_type')
                        periods = int(request.form.get('periods', 6))
                        
                        if not selected_fish:
                            error = "Pilih jenis ikan."
                        elif not forecaster.preprocess_data(fish_type=selected_fish):
                            error = "Gagal memproses data. Periksa kolom file (butuh 'Nilai_Produksi_Rp' dan 'Volume_Produksi_Kg')."
                        else:
                            forecaster.train_arima() 
                            preds = forecaster.predict_future(periods=periods)
                            
                            if preds.empty:
                                error = "Gagal menghasilkan prediksi."
                            else:
                                fig, ax = plt.subplots(figsize=(8, 5))
                                ax.plot(forecaster.processed_data.index, forecaster.processed_data['Harga_Per_Kg'], 'b-', label='Data Historis', alpha=0.7)
                                pred_dates = pd.to_datetime(preds["Tanggal"])
                                ax.plot(pred_dates, preds["Prediksi"], 'r-', marker='o', label='Prediksi ARIMA')
                                ax.fill_between(pred_dates, preds["Batas_Bawah"], preds["Batas_Atas"], color='red', alpha=0.1, label='Interval Kepercayaan')
                                ax.set_title(f'Prediksi Harga: {selected_fish}'); ax.set_xlabel('Tanggal'); ax.set_ylabel('Harga (Rp)')
                                ax.legend(); ax.grid(True, linestyle='--', alpha=0.6); plt.xticks(rotation=45); plt.tight_layout()
                                plot_html = plot_to_base64(fig)

    except Exception as e:
        error = str(e)
        print(f"Error di /market POST: {e}")
        flask.session.pop('market_file', None)
        flask.session.pop('fish_options', None)
        fish_options = []

    return render_template('market.html', 
                           plot_harga=plot_html, 
                           fish_options=fish_options, 
                           selected_fish=selected_fish,
                           error=error)

@app.route('/classification', methods=['GET', 'POST'])
def classification():
    # Panggil 'get_classifier' untuk mendapatkan instance TFLite
    classifier = get_classifier()
    
    if request.method == 'POST':
        image = None
        error = None

        try:
            # Logika Input Ganda (Tetap sama)
            if 'file' in request.files and request.files['file'].filename != '':
                file = request.files['file']
                image = Image.open(file.stream)
                print("Menerima gambar dari UPLOAD FILE")

            elif 'image_data' in request.form:
                img_data_url = request.form.get('image_data')
                header, encoded_data = img_data_url.split(',', 1)
                image_bytes = base64.b64decode(encoded_data)
                image = Image.open(BytesIO(image_bytes))
                print("Menerima gambar dari WEBCAM (Base64)")

            else:
                error = "Tidak ada file yang di-upload atau gambar yang diambil."
                return render_template('classification.html', error=error)
            
            if image:
                buf = BytesIO()
                image.save(buf, format='PNG')
                img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                
                # Prediksi (sekarang menggunakan TFLite)
                result1, result2 = classifier.predict(image) 
                
                if result1 is None or result2 is None:
                    error = "Model klasifikasi TFLite gagal prediksi. Periksa log server."
                    return render_template('classification.html', error=error)
                
                info = get_conservation_info(result1['class'], result2['class'])
                
                return render_template(
                    'classification.html',
                    img_data=img_b64,
                    result1=result1,
                    result2=result2,
                    info=info,
                    get_confidence_color=get_confidence_color 
                )
            
        except Exception as e:
            print(f"Error di /classification: {e}")
            error = f"Gagal memproses gambar: {e}"
            return render_template('classification.html', error=error)
            
    return render_template('classification.html')

@app.route('/game')
def game():
    return render_template('game.html')

# --- RUTE BARU UNTUK FITUR PETA ---
@app.route('/peta')
def peta_viewer():
    """Menampilkan halaman utama peta pelacakan."""
    return render_template('peta_viewer.html')

@app.route('/peta/share')
def peta_share_location():
    """Menampilkan halaman untuk berbagi lokasi."""
    return render_template('peta_share_location.html')
# --- AKHIR RUTE PETA ---

if __name__ == '__main__':
    # Tetap gunakan use_reloader=False untuk stabilitas
    app.run(debug=True, use_reloader=False, port=5000)