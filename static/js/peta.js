// PENTING: Salin konfigurasi Firebase Anda (dari konsol Firebase) di sini
// Ini adalah konfigurasi sisi KLIEN/WEB, berbeda dari file .json
// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";

// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyBywdiIgUrxtxdxpKLO-0Jd0ioETQFa_z0",
  authDomain: "perikanan-73237.firebaseapp.com",
  databaseURL: "https://perikanan-73237-default-rtdb.firebaseio.com",
  projectId: "perikanan-73237",
  storageBucket: "perikanan-73237.firebasestorage.app",
  messagingSenderId: "128236265158",
  appId: "1:128236265158:web:8f21839850d3cb9abfc1c7",
  measurementId: "G-7PF627P3ZT"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);

// Inisialisasi Firebase
firebase.initializeApp(firebaseConfig);
const db = firebase.database();

// --- LOGIKA UNTUK HALAMAN PETA VIEWER ---
if (window.location.pathname.includes("/peta")) {
    // Pastikan DOM dimuat sebelum menjalankan kode Leaflet
    document.addEventListener('DOMContentLoaded', () => {
        const mapElement = document.getElementById('peta-map');
        if (mapElement) {
            console.log("Memuat Peta Viewer...");
            const map = L.map('peta-map').setView([-6.1215, 106.7741], 12);

            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                maxZoom: 19,
                attribution: '© OpenStreetMap'
            }).addTo(map);

            const markers = {};
            let vesselCount = 0;
            const vesselCountEl = document.getElementById('vessel-count');
            let hasFitted = false;

            // Ikon kapal kustom
            const shipIcon = L.icon({
                iconUrl: 'https://cdn-icons-png.flaticon.com/512/3135/3135687.png', // Ganti dengan URL ikon kapal jika Anda mau
                iconSize: [32, 32],
                iconAnchor: [16, 16]
            });

            db.ref("locations").on("value", snapshot => {
                const data = snapshot.val();
                if (!data) return;

                console.log("Menerima data lokasi:", data);
                vesselCount = 0;
                const activeIds = Object.keys(data);
                
                // Hapus marker yang sudah tidak ada di database
                Object.keys(markers).forEach(id => {
                    if (!activeIds.includes(id)) {
                        map.removeLayer(markers[id]);
                        delete markers[id];
                    }
                });

                activeIds.forEach(id => {
                    const vessel = data[id];
                    // Hanya tampilkan jika ada lat/lng
                    if (vessel.lat && vessel.lng) {
                        vesselCount++;
                        const popupContent = `
                            <div class="font-sans">
                                <strong class="text-base text-primary">${vessel.nama_kapal || 'Tanpa Nama'}</strong><br>
                                <strong>Jenis:</strong> ${vessel.jenis_kapal || 'N/A'}<br>
                                <strong>Muatan:</strong> ${vessel.muatan || 'N/A'}<br>
                                <hr class="my-1">
                                <span class="text-xs text-muted-foreground">ID: ${id}</span><br>
                                <span class="text-xs text-muted-foreground">Lat: ${vessel.lat.toFixed(4)}, Lng: ${vessel.lng.toFixed(4)}</span>
                            </div>
                        `;

                        if (!markers[id]) {
                            markers[id] = L.marker([vessel.lat, vessel.lng], { icon: shipIcon }).addTo(map)
                                .bindPopup(popupContent);
                        } else {
                            markers[id].setLatLng([vessel.lat, vessel.lng])
                                .getPopup().setContent(popupContent);
                        }
                    }
                });
                
                if (vesselCountEl) {
                    vesselCountEl.textContent = vesselCount;
                }

                // Auto-fit bounds saat pertama kali memuat
                if (!hasFitted && vesselCount > 0) {
                    const group = new L.featureGroup(Object.values(markers));
                    map.fitBounds(group.getBounds().pad(0.1));
                    hasFitted = true;
                }
            });
        }
    });
}

// --- LOGIKA UNTUK HALAMAN SHARE LOCATION ---
if (window.location.pathname.includes("/peta/share")) {
    document.addEventListener('DOMContentLoaded', () => {
        console.log("Memuat Halaman Share Lokasi...");
        
        const form = document.getElementById('share-form');
        const shareBtn = document.getElementById('share-btn');
        const statusEl = document.getElementById('status-message');
        const coordinatesEl = document.getElementById('coordinates');

        let isSharing = false;
        let watchId = null;
        let userKey = "vessel_" + Math.random().toString(36).substr(2, 9); // Buat ID kapal acak

        const showStatus = (message, type = 'info') => {
            if (statusEl) {
                statusEl.textContent = message;
                statusEl.classList.remove('text-green-600', 'text-red-600', 'text-gray-600');
                if (type === 'success') {
                    statusEl.classList.add('text-green-600');
                } else if (type === 'error') {
                    statusEl.classList.add('text-red-600');
                } else {
                    statusEl.classList.add('text-gray-600');
                }
            }
        };

        const updateLocationToFirebase = (position) => {
            const data = {
                lat: position.coords.latitude,
                lng: position.coords.longitude,
                nama_kapal: document.getElementById('namaKapal').value.trim(),
                jenis_kapal: document.getElementById('jenisKapal').value.trim(),
                muatan: document.getElementById('muatan').value.trim(),
                timestamp: firebase.database.ServerValue.TIMESTAMP
            };

            db.ref(`locations/${userKey}`).set(data)
                .then(() => {
                    showStatus('Lokasi berhasil dikirim!', 'success');
                    if (coordinatesEl) {
                        coordinatesEl.textContent = `Lat: ${data.lat.toFixed(5)}, Lng: ${data.lng.toFixed(5)}`;
                    }
                })
                .catch(err => {
                    showStatus(`Gagal mengirim lokasi: ${err.message}`, 'error');
                });
        };

        const startLocationSharing = () => {
            if (!("geolocation" in navigator)) {
                showStatus("Geolocation tidak didukung di browser ini.", "error");
                return;
            }

            showStatus("Memulai pelacakan...", "info");
            watchId = navigator.geolocation.watchPosition(
                updateLocationToFirebase,
                (err) => {
                    showStatus(`Error GPS: ${err.message}`, 'error');
                },
                {
                    enableHighAccuracy: true,
                    timeout: 10000,
                    maximumAge: 0
                }
            );

            isSharing = true;
            shareBtn.textContent = 'Hentikan Berbagi Lokasi';
            shareBtn.classList.remove('bg-primary', 'hover:bg-primary/90');
            shareBtn.classList.add('bg-destructive', 'hover:bg-destructive/90');
            ['namaKapal', 'jenisKapal', 'muatan'].forEach(id => document.getElementById(id).disabled = true);
        };

        const stopLocationSharing = () => {
            if (watchId) {
                navigator.geolocation.clearWatch(watchId);
                watchId = null;
            }
            
            // Hapus data dari Firebase saat berhenti
            db.ref(`locations/${userKey}`).remove()
                .then(() => showStatus('Berbagi lokasi dihentikan. Data dihapus.', 'info'))
                .catch(err => showStatus(`Error menghapus data: ${err.message}`, 'error'));

            isSharing = false;
            shareBtn.textContent = 'Mulai Berbagi Lokasi';
            shareBtn.classList.add('bg-primary', 'hover:bg-primary/90');
            shareBtn.classList.remove('bg-destructive', 'hover:bg-destructive/90');
            ['namaKapal', 'jenisKapal', 'muatan'].forEach(id => document.getElementById(id).disabled = false);
            if (coordinatesEl) coordinatesEl.textContent = '';
        };

        if (form) {
            form.addEventListener('submit', (e) => {
                e.preventDefault();

                if (isSharing) {
                    stopLocationSharing();
                } else {
                    const namaKapal = document.getElementById('namaKapal').value.trim();
                    const jenisKapal = document.getElementById('jenisKapal').value.trim();
                    const muatan = document.getElementById('muatan').value.trim();

                    if (!namaKapal || !jenisKapal || !muatan) {
                        showStatus('Semua data kapal harus diisi!', 'error');
                        return;
                    }
                    startLocationSharing();
                }
            });
        }
        
        // Hentikan sharing jika halaman ditutup
        window.addEventListener('beforeunload', () => {
            if (isSharing) {
                db.ref(`locations/${userKey}`).remove();
            }
        });
    });
}