import streamlit as st

def main(ICONS):
    """
    Render Halaman Utama (Landing Page) yang identik dengan
    file referensi Index.tsx.
    """
    
    # --- 1. HERO SECTION (Area Abu-abu "Selamat Datang") ---
    st.markdown(
        f"""
        <section class="custom-hero">
            <div class="custom-hero-content">
                <div style="margin-bottom: 1.5rem;">
                    <img src="{ICONS['fish_hero']}" width="64" height="64">
                </div>
                <h1>Selamat Datang di FishStatik</h1>
                <p>Platform Analisis Perikanan Modern dengan Teknologi AI dan Data Science</p>
            </div>
        </section>
        """,
        unsafe_allow_html=True
    )
    
    # --- 2. FEATURE CARDS (Kotak-kotak) ---
    st.markdown(
        f"""
        <div class="feature-card-grid">
            
            <div class="feature-card">
                <span class="feature-card-badge badge-active">Active</span>
                <div class="feature-card-content">
                    <div class="feature-card-icon">
                        <div class="feature-card-icon-bg">
                            <img src="{ICONS['cpue']}" width="32" height="32" style="stroke: white;">
                        </div>
                    </div>
                    <h3>CPUE & MSY Analysis</h3>
                    <p>Analisis Catch Per Unit Effort dan Maximum Sustainable Yield menggunakan Model Schaefer.</p>
                </div>
            </div>

            <div class="feature-card">
                <span class="feature-card-badge badge-development">Development</span>
                <div class="feature-card-content">
                    <div class="feature-card-icon">
                        <div class="feature-card-icon-bg">
                            <img src="{ICONS['stock']}" width="32" height="32" style="stroke: white;">
                        </div>
                    </div>
                    <h3>Stock Prediction</h3>
                    <p>Model prediksi stok ikan berbasis ARIMA dan machine learning untuk perencanaan berkelanjutan.</p>
                </div>
            </div>

            <div class="feature-card">
                <span class="feature-card-badge badge-beta">Beta</span>
                <div class="feature-card-content">
                    <div class="feature-card-icon">
                        <div class="feature-card-icon-bg">
                            <img src="{ICONS['price']}" width="32" height="32" style="stroke: white;">
                        </div>
                    </div>
                    <h3>Market Foresight</h3>
                    <p>Prediksi harga ikan dengan AI untuk optimasi profit dan analisis pasar real-time.</p>
                </div>
            </div>

            <div class="feature-card">
                <span class="feature-card-badge badge-active">Active</span>
                <div class="feature-card-content">
                    <div class="feature-card-icon">
                        <div class="feature-card-icon-bg">
                            <img src="{ICONS['viz']}" width="32" height="32" style="stroke: white;">
                        </div>
                    </div>
                    <h3>Data Visualization</h3>
                    <p>Visualisasi data perikanan interaktif dengan dashboard analitik komprehensif.</p>
                </div>
            </div>

            <div class="feature-card">
                <span class="feature-card-badge badge-beta">Beta</span>
                <div class="feature-card-content">
                    <div class="feature-card-icon">
                        <div class="feature-card-icon-bg">
                            <img src="{ICONS['classify']}" width="32" height="32" style="stroke: white;">
                        </div>
                    </div>
                    <h3>Fish Classification</h3>
                    <p>Sistem klasifikasi spesies ikan menggunakan computer vision dan deep learning.</p>
                </div>
            </div>

            <div class="feature-card">
                <span class="feature-card-badge badge-beta">Beta</span>
                <div class="feature-card-content">
                    <div class="feature-card-icon">
                        <div class="feature-card-icon-bg">
                            <img src="{ICONS['sim']}" width="32" height="32" style="stroke: white;">
                        </div>
                    </div>
                    <h3>Simulation Tools</h3>
                    <p>Simulasi skenario perikanan untuk pengambilan keputusan berbasis data.</p>
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    # Dummy ICONS untuk testing file ini secara mandiri
    DUMMY_ICONS = { "fish_header": "", "fish_hero": "", "cpue": "", "stock": "", "price": "", "viz": "", "classify": "", "sim": "" }
    main(DUMMY_ICONS)