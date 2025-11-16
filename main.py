import warnings
import streamlit as st
from streamlit.components.v1 import html
import base64 # Untuk ikon

# PANGGIL HANYA SEKALI
st.set_page_config(
    page_title="Analisis Perikanan - Fish Statik",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="collapsed" # Sembunyikan sidebar
)

# Load custom CSS (file ini akan menyembunyikan sidebar)
try:
    with open('pages.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.error("File 'pages.css' tidak ditemukan. Desain tidak akan dimuat.")

# --- Ikon (dikonversi ke base64 agar tidak perlu file) ---
ICONS = {
    "fish_header": "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImhzbCh2YXIoLS1vY2Vhbi1ibHVlKSkiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cGF0aCBkPSJNMjIgMTAuNDhjMC0uOS0xLjItMS40LTItMS40LTYgMC02IDIuMy02IDQuNnMwIDQuNiA2IDQuNiAxLjctLjUgMi0xLjRjMS44LTYuMy0uNCA5LjYtNi40IDkuNi04IDAtOC00LTgtOC44cy0uOS04LjggOC04LjhDMjIuMi45IDIwLjIgMy4yIDIyIDEwLjQ4WiIvPjxwYXRoIGQ9Ik0xOCAyYy41LTIuNSAzIDIuMyAzIDUuNUMyMSA5LjggMTkuNSA5IDE5IDgiLz48L3N2Zz4=",
    "fish_hero": "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI2NCIgaGVpZ2h0PSI2NCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9ImhzbCh2YXIoLS1vY2Vhbi1ibHVlKSkiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yMiAxMC40OGMwLS45LTEuMi0xLjQtMi0xLjQtNiAwLTYgMi4zLTYgNC42czAgNC42IDYgNC42IDEuNy0uNSAyLTEuNGMxLjgtNi4zLS40IDkuNi02LjQgOS42LTggMC04LTQtOC04LjhzLS45LTguOCA4LTguOEMyMi4yLjkgMjAuMiAzLjIgMjIgMTAuNDhaIi8+PHBhdGggZD0iTTE4IDJjLjUtMi41IDMgMi4zIDMgNS41QzIxIDkuOCAxOS41IDkgMTkgOCIvPjwvc3ZnPg==",
    "cpue": "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMiIgaGVpZ2h0PSIzMiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9InJnYmEoMjU1LDI1NSwyNTUsMC45KSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5bGluZSBwb2ludHM9IjIzIDYgMjMgMTYgMSAxNiIvPjxwb2x5bGluZSBwb2ludHM9IjE3IDYgMjMgNiAxOSAxMCIvPjwvc3ZnPg==",
    "stock": "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMiIgaGVpZ2h0PSIzMiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9InJnYmEoMjU1LDI1NSwyNTUsMC45KSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik05LjUgNy41UzIwIDcuNSAyMSA3Ii8+PHBhdGggZD0iTTE3IDEyLjVDMTcgMTIuNSAyMSAxNyAyMSAxMiIvPjxwYXRoIGQ9Ik0xMiA4YzAtMS41LTMtMi0zLTUiLz48cGF0aCBkPSJNMTEgNnMzLTUgNy0yIDIgOC0zIDciLz48cGF0aCBkPSJNMTEgMTJjMC0xLjUgMC0yIDItMyBzMy44LTMgMy44LTMuNSIvPjxwYXRoIGQ9Ik0zIDEyYzAtMiA0IDMgNSAyIi8+PHBhdGggZD0iTTYgOHMyLTMgMy01Ii8+PHBhdGggZD0iTTUgMTJjMC0yLTMgMC0zIDIuNSIvPjxwYXRoIGQ9Ik0xMSAxNmMxLjItMS4zIDQtMS4zIDUtMSAwIC41IDIuOCAyLjggMi44IDUiLz48cGF0aCBkPSJNMTMgMTIuNXMtMiAzLTUgMyIvPjxwYXRoIGQ9Ik0xIDlzMyAzIDYgNCIvPjxwYXRoIGQ9Ik0yIDE2YzAgNi4xIDMuMyA2LjkgNSA1LjkiLz48cGF0aCBkPSJNMjIgMTlzLTMuNS0zLTctNCIvPjxwYXRoIGQ9Ik0yMCAxMmMwIDAtMy41LTIuNS0zLjUgMiIvPjxwYXRoIGQ9Ik0yMiA5Yy0uOC0xLjMtNS41LTEuNS01LjUgMiIvPjwvc3ZnPg==",
    "price": "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMiIgaGVpZ2h0PSIzMiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9InJnYmEoMjU1LDI1NSwyNTUsMC45KSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxsaW5lIHgxPSIxMiIgeDI9IjEyIiB5MT0iMSIgeTI9IjIzIi8+PHBhdGggZD0iTTE3IDVIMy41YTMuNSAzLjUgMCAwIDAgMCA3aDdhNC41IDQuNSAwIDAgMSAwIDlINyIvPjwvc3ZnPg==",
    "viz": "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMiIgaGVpZ2h0PSIzMiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9InJnYmEoMjU1LDI1NSwyNTUsMC45KSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0zIDZ2MTRoLjAxIi8+PHBhdGggZD0iTTcgNnYxNGgiLz48cGF0aCBkPSJNMTkgNnYxNGgiLz48cGF0aCBkPSJNMTUgNnYxNGgiLz48cGF0aCBkPSJNMTEgNnYxNGgiLz48L3N2Zz4=",
    "classify": "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMiIgaGVpZ2h0PSIzMiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9InJnYmEoMjU1LDI1NSwyNTUsMC45KSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0yMiAxMC40OGMwLS45LTEuMi0xLjQtMi0xLjQtNiAwLTYgMi4zLTYgNC42czAgNC42IDYgNC42IDEuNy0uNSAyLTEuNGMxLjgtNi4zLS40IDkuNi02LjQgOS42LTggMC04LTQtOC04LjhzLS45LTguOCA4LTguOEMyMi4yLjkgMjAuMiAzLjIgMjIgMTAuNDhaIi8+PHBhdGggZD0iTTE4IDJjLjUtMi41IDMgMi4zIDMgNS41QzIxIDkuOCAxOS41IDkgMTkgOCIvPjwvc3ZnPg==",
    "sim": "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMiIgaGVpZ2h0PSIzMiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9InJnYmEoMjU1LDI1NSwyNTUsMC45KSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwb2x5Z29uIHBvaW50cz0iMTMgMiAzIDE0IDEyIDE0IDExIDIyIDIxIDEwIDEyIDEwIDEzIDIiLz48L3N2Zz4="
}
def get_icon(name):
    return f"data:image/svg+xml;base64,{ICONS[name]}"

# --- DAFTAR HALAMAN ---
PAGES = {
    "Home": { "module": "Home", "icon": "fish_hero" },
    "CPUE & MSY": { "module": "analysis", "icon": "cpue" },
    "Prediksi Stok": { "module": "model", "icon": "stock" },
    "Proyeksi Harga": { "module": "market_foresight", "icon": "price" },
    "Visualisasi": { "module": "dashboard", "icon": "viz" },
    "Klasifikasi": { "module": "marine_classification_simple", "icon": "classify" },
    "Simulasi": { "module": "game", "icon": "sim" }
}

# Inisialisasi session state
if 'page' not in st.session_state:
    st.session_state['page'] = "Home"

def set_page(page_name):
    st.session_state['page'] = page_name

# --- RENDER NAVBAR ATAS (PENGGANTI SIDEBAR) ---
st.markdown(
    f"""
    <header class="custom-header">
        <div class="custom-header-container">
            <div class="custom-header-logo">
                <div class="icon-bg">
                    <img src="{get_icon('fish_header')}" width="24" height="24">
                </div>
                <h1>FishStatik</h1>
            </div>
            <nav class="custom-nav" id="custom-nav-buttons">
                {
                    " ".join([
                        f'<button id="btn-nav-{page_info["module"]}" class="custom-nav-button {"active" if st.session_state["page"] == name else ""}">{name}</button>'
                        for name, page_info in PAGES.items()
                    ])
                }
            </nav>
        </div>
    </header>
    """,
    unsafe_allow_html=True
)

# --- Logika Navigasi (Trik Streamlit) ---
cols = st.columns(len(PAGES))
for i, (page_name, page_info) in enumerate(PAGES.items()):
    with cols[i]:
        st.button(
            label=f"hidden_{page_name}",
            key=f"btn_key_{page_info['module']}",
            on_click=set_page,
            args=(page_name,)
        )

st.markdown("<style>div[data-testid='stHorizontalBlock'] { display: none; }</style>", unsafe_allow_html=True)

# --- PERBAIKAN: Hapus unsafe_allow_html=True dari html() ---
html(f"""
    <script>
    // Pastikan skrip ini berjalan setelah elemen dimuat
    (function() {{
        // Fungsi untuk dijalankan
        const setupNavListeners = () => {{
            const buttons = document.querySelectorAll('.custom-nav-button');
            if (buttons.length === 0) {{
                // Jika tombol belum ada, coba lagi
                setTimeout(setupNavListeners, 100);
                return;
            }}
            
            buttons.forEach(btn => {{
                const moduleName = btn.id.replace('btn-nav-', '');
                
                // Selector yang lebih kuat untuk menemukan tombol Streamlit
                const stButton = window.parent.document.querySelector('button[data-testid^="stButton-"][key="btn_key_{moduleName}"]');
                
                if (stButton) {{
                    btn.addEventListener('click', () => {{
                        stButton.click();
                    }});
                }} else {{
                    console.warn(`Tombol Streamlit untuk modul ${moduleName} tidak ditemukan.`);
                }}
            }});
        }};

        // Coba jalankan saat DOM siap
        if (document.readyState === 'complete' || document.readyState === 'interactive') {{
            setupNavListeners();
        }} else {{
            document.addEventListener('DOMContentLoaded', setupNavListeners);
        }}
    }})();
    </script>
""")


# --- KONTEN HALAMAN ---
current_page_name = st.session_state['page']
current_page_info = PAGES[current_page_name]

# Wrap semua pemanggilan halaman dalam container
st.markdown('<div class="main-content-wrapper">', unsafe_allow_html=True)

if current_page_name == "Home":
    try:
        from Home import main as home_main
        home_main(ICONS) # Kirim ikon ke halaman utama
    except ImportError:
        st.error("File 'Home.py' tidak ditemukan. Buat file tersebut untuk halaman utama.")
    except Exception as e:
        st.error(f"Error saat memuat Home.py: {e}")

elif current_page_name == "Simulasi":
    try:
        with open("game.html", "r", encoding="utf-8") as f:
            game_html = f.read()
        html(game_html, height=1000, scrolling=True)
    except FileNotFoundError:
        st.error("File 'game.html' tidak ditemukan.")

else:
    # Muat modul halaman lain seperti biasa
    try:
        module = __import__(current_page_info["module"])
        module.main()
    except ImportError as e:
        st.error(f"Gagal memuat modul: {current_page_info['module']}.py\nError: {e}")
    except Exception as e:
        st.error(f"Error saat menjalankan {current_page_name}: {e}")

st.markdown('</div>', unsafe_allow_html=True)