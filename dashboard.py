import streamlit as st
from plotly.subplots import make_subplots
from io import StringIO
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- FUNGSI HELPER (Tidak Berubah) ---
@st.cache_data
def load_and_clean_data(uploaded_file=None):
    if uploaded_file is not None:
        try:
            content = uploaded_file.getvalue().decode('utf-8')
            df = pd.read_csv(StringIO(content), sep='\t')
        except:
            uploaded_file.seek(0)
            content = uploaded_file.getvalue().decode('utf-8')
            df = pd.read_csv(StringIO(content))
    else:
        return pd.DataFrame()
    
    df.columns = df.columns.str.strip()
    df['Volume Produksi (kg)'] = pd.to_numeric(
        df['Volume Produksi (kg)'].astype(str).replace(['-', '', ' ', 'nan'], '0'), 
        errors='coerce'
    ).fillna(0)
    df['Jenis Ikan'] = df['Jenis Ikan'].astype(str).str.strip().str.replace('"', '')
    df = df.drop_duplicates(subset=['Tahun', 'Bulan', 'Jenis Ikan'])
    bulan_order = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
                   'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
    df['Bulan_Num'] = df['Bulan'].map({b: i+1 for i, b in enumerate(bulan_order)})
    df['Bulan_Num'] = df['Bulan_Num'].fillna(1)
    try:
        df['Tanggal'] = pd.to_datetime(
            df['Tahun'].astype(str) + '-' + df['Bulan_Num'].astype(int).astype(str) + '-01',
            format='%Y-%m-%d',
            errors='coerce'
        )
    except:
        df['Tanggal'] = pd.to_datetime('2020-01-01')
    df = df.dropna(subset=['Tanggal'])
    return df

def create_kpi_cards(df, col1, col2, col3, col4):
    if df.empty:
        st.info("Unggah data untuk melihat KPI.")
        return
    total_produksi = df['Volume Produksi (kg)'].sum()
    monthly_sum = df.groupby(['Tahun', 'Bulan'])['Volume Produksi (kg)'].sum()
    rata_bulanan = monthly_sum.mean() if len(monthly_sum) > 0 else 0
    jumlah_jenis = df['Jenis Ikan'].nunique()
    yearly_sum = df.groupby('Tahun')['Volume Produksi (kg)'].sum()
    if len(yearly_sum) > 0 and yearly_sum.max() > 0:
        tahun_terbaik = yearly_sum.idxmax()
    else:
        tahun_terbaik = "N/A"
    with col1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Total Produksi</div><div class="kpi-value">{total_produksi/1_000_000:.2f}M kg</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Rata-rata/Bulan</div><div class="kpi-value">{rata_bulanan/1000:.1f}K kg</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Jenis Ikan</div><div class="kpi-value">{jumlah_jenis}</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Tahun Terbaik</div><div class="kpi-value">{tahun_terbaik}</div></div>""", unsafe_allow_html=True)

def plot_trend_tahunan(df):
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    if df.empty:
        st.info("Data kosong untuk plot trend.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    df_bulanan = df.groupby('Tanggal')['Volume Produksi (kg)'].sum().reset_index()
    fig = px.line(df_bulanan, x='Tanggal', y='Volume Produksi (kg)', title='📈 Tren Produksi Ikan Bulanan (2020-2024)', labels={'Volume Produksi (kg)': 'Volume Produksi (kg)', 'Tanggal': 'Periode'})
    fig.update_traces(line_color='#1E88E5', line_width=3)
    fig.update_layout(hovermode='x unified', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=400)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def plot_top_species(df, n=10):
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    if df.empty:
        st.info("Tidak ada data untuk ditampilkan")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    top_species = df.groupby('Jenis Ikan')['Volume Produksi (kg)'].sum().nlargest(n).reset_index()
    if top_species.empty:
        st.info("Tidak ada data untuk ditampilkan")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    fig = px.bar(top_species, x='Volume Produksi (kg)', y='Jenis Ikan', orientation='h', title=f'🏆 Top {n} Jenis Ikan dengan Produksi Tertinggi', color='Volume Produksi (kg)', color_continuous_scale='Blues')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500, showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def plot_heatmap_bulanan(df):
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    if df.empty:
        st.info("Tidak ada data untuk ditampilkan")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    pivot_data = df.groupby(['Tahun', 'Bulan'])['Volume Produksi (kg)'].sum().reset_index()
    pivot_table = pivot_data.pivot(index='Bulan', columns='Tahun', values='Volume Produksi (kg)')
    bulan_order = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
    pivot_table = pivot_table.reindex(bulan_order)
    fig = px.imshow(pivot_table.T, labels=dict(x="Bulan", y="Tahun", color="Volume (kg)"), title='🔥 Heatmap Produksi Bulanan per Tahun', color_continuous_scale='YlOrRd', aspect='auto')
    fig.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def plot_comparison_yearly(df):
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    if df.empty:
        st.info("Tidak ada data untuk ditampilkan")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    yearly_prod = df.groupby('Tahun')['Volume Produksi (kg)'].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=yearly_prod['Tahun'], y=yearly_prod['Volume Produksi (kg)'], text=yearly_prod['Volume Produksi (kg)'].apply(lambda x: f'{x/1000:.0f}K'), textposition='auto', marker_color='#764ba2'))
    fig.update_layout(title='📊 Perbandingan Produksi Tahunan', xaxis_title='Tahun', yaxis_title='Volume Produksi (kg)', height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# Main App
def main():
    st.markdown('<h1 class="dashboard-title">🐟 Visualisasi Analisis Produksi Ikan</h1>', 
                unsafe_allow_html=True)
    
    # --- PERBAIKAN: Pindahkan Sidebar ke Konten Utama ---
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    
    st.subheader("📁 Upload & Filter Data")
    
    uploaded_file = st.file_uploader(
        "Upload file CSV/TXT (format: Tahun, Bulan, Jenis Ikan, Volume)",
        type=['csv', 'txt']
    )
    
    selected_years, selected_months, selected_species = [], [], []
    df = pd.DataFrame()

    if uploaded_file is not None:
        with st.spinner('Memproses data...'):
            df = load_and_clean_data(uploaded_file)
    
    if df is not None and not df.empty:
        st.markdown("---")
        st.subheader("🔧 Filter Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tahun_options = sorted(df['Tahun'].unique())
            selected_years = st.multiselect(
                "Pilih Tahun", options=tahun_options, default=tahun_options
            )
        
        with col2:
            bulan_options = df['Bulan'].unique().tolist()
            selected_months = st.multiselect(
                "Pilih Bulan", options=bulan_options, default=bulan_options
            )
        
        with col3:
            jenis_options = sorted(df['Jenis Ikan'].unique())
            selected_species = st.multiselect(
                "Pilih Jenis Ikan", options=jenis_options, default=jenis_options[:20] if len(jenis_options) > 20 else jenis_options
            )
    
    st.markdown('</div>', unsafe_allow_html=True)
    # --- Akhir Perbaikan ---
    
    if uploaded_file is not None:
        if df is not None and not df.empty:
            
            df_filtered = df[
                (df['Tahun'].isin(selected_years)) &
                (df['Bulan'].isin(selected_months)) &
                (df['Jenis Ikan'].isin(selected_species))
            ]
            
            if df_filtered.empty:
                st.warning("⚠️ Tidak ada data yang sesuai dengan filter. Silakan ubah filter.")
            else:
                col1, col2, col3, col4 = st.columns(4)
                create_kpi_cards(df_filtered, col1, col2, col3, col4)
                
                col1, col2 = st.columns(2)
                with col1:
                    plot_trend_tahunan(df_filtered)
                with col2:
                    plot_comparison_yearly(df_filtered)
                
                col1, col2 = st.columns(2)
                with col1:
                    plot_top_species(df_filtered, 10)
                with col2:
                    plot_heatmap_bulanan(df_filtered)
                
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.subheader("📋 Data Detail")
                
                df_display = df_filtered.groupby(['Tahun', 'Bulan', 'Jenis Ikan'])['Volume Produksi (kg)'].sum().reset_index()
                df_display = df_display.sort_values(['Tahun', 'Volume Produksi (kg)'], 
                                                    ascending=[False, False])
                
                st.dataframe(
                    df_display[['Tahun', 'Bulan', 'Jenis Ikan', 'Volume Produksi (kg)']],
                    use_container_width=True,
                    height=400
                )
                
                csv = df_display.to_csv(index=False)
                st.download_button(
                    label="📥 Download Data (CSV)",
                    data=csv,
                    file_name="produksi_ikan_filtered.csv",
                    mime="text/csv"
                )
                st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.error("❌ Gagal memproses file. Pastikan format file sesuai.")
    
    else:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.info("👈 Silakan upload file data di atas untuk memulai analisis.")
        st.markdown("""
        ### 📖 Cara Menggunakan Dashboard:
        1. **Upload Data**: Klik tombol "Browse files" di atas.
        2. **Filter Data**: Gunakan filter yang muncul setelah data di-upload.
        ...
        """)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()