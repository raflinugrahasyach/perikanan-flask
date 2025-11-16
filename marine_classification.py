import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
from io import BytesIO
from classification_config import (
    MODEL_PATHS, IMAGE_SETTINGS, CONFIDENCE_THRESHOLDS, 
    SPECIES_INFO, CONSERVATION_STATUS, UI_CONFIG
)

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

class MarineClassifier:
    def __init__(self):
        self.model1 = None
        self.model2 = None
        self.labels1 = []
        self.labels2 = []
        self.load_models()
        self.load_labels()
    
    def load_models(self):
        """Load kedua model TensorFlow"""
        try:
            model1_path = MODEL_PATHS['species_model']
            model2_path = MODEL_PATHS['conservation_model']
            
            if not os.path.exists(model1_path):
                raise FileNotFoundError(f"Model 1 not found: {model1_path}")
            
            if not os.path.exists(model2_path):
                raise FileNotFoundError(f"Model 2 not found: {model2_path}")
            
            # Load models with custom_objects to handle Teachable Machine models
            custom_objects = {
                'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D
            }
            
            # Load models dengan kompilasi ulang untuk menghindari warning
            self.model1 = tf.keras.models.load_model(
                model1_path, 
                custom_objects=custom_objects,
                compile=False
            )
            self.model2 = tf.keras.models.load_model(
                model2_path, 
                custom_objects=custom_objects,
                compile=False
            )
            
            # Compile ulang model untuk inference
            self.model1.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model2.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")  # Print untuk debugging
            self.model1 = None
            self.model2 = None
    
    def load_labels(self):
        """Load labels untuk kedua model"""
        try:
            # Load labels model 1 (jenis biota laut)
            with open(MODEL_PATHS['species_labels'], "r") as f:
                self.labels1 = [line.strip() for line in f.readlines()]
            
            # Load labels model 2 (status konservasi)
            with open(MODEL_PATHS['conservation_labels'], "r") as f:
                self.labels2 = [line.strip() for line in f.readlines()]
                
        except Exception as e:
            st.error(f"❌ Error loading labels: {str(e)}")
    
    def preprocess_image(self, image):
        """Preprocess gambar untuk input ke model"""
        try:
            # Resize image menggunakan konfigurasi
            target_size = IMAGE_SETTINGS['target_size']
            image = image.resize(target_size)
            
            # Convert to RGB if needed
            if image.mode != IMAGE_SETTINGS['color_mode']:
                image = image.convert(IMAGE_SETTINGS['color_mode'])
            
            # Convert to numpy array
            image_array = np.array(image)
            
            # Normalize jika diaktifkan
            if IMAGE_SETTINGS['normalize']:
                image_array = image_array.astype(np.float32) / 255.0
            
            # Convert to specified data type
            image_array = image_array.astype(IMAGE_SETTINGS['data_type'])
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            return image_array
        except Exception as e:
            st.error(f"❌ Error preprocessing image: {str(e)}")
            return None
    
    def predict(self, image):
        """Prediksi menggunakan kedua model"""
        if self.model1 is None or self.model2 is None:
            return None, None
        
        processed_image = self.preprocess_image(image)
        if processed_image is None:
            return None, None
        
        try:
            # Prediksi dengan model 1 (jenis biota laut)
            prediction1 = self.model1.predict(processed_image, verbose=0)
            class1_idx = np.argmax(prediction1[0])
            confidence1 = float(prediction1[0][class1_idx])
            
            # Prediksi dengan model 2 (status konservasi)
            prediction2 = self.model2.predict(processed_image, verbose=0)
            class2_idx = np.argmax(prediction2[0])
            confidence2 = float(prediction2[0][class2_idx])
            
            result1 = {
                'class': self.labels1[class1_idx] if class1_idx < len(self.labels1) else "Unknown",
                'confidence': confidence1,
                'all_predictions': [(self.labels1[i], float(prediction1[0][i])) 
                                    for i in range(len(prediction1[0]))]
            }
            
            result2 = {
                'class': self.labels2[class2_idx] if class2_idx < len(self.labels2) else "Unknown",
                'confidence': confidence2,
                'all_predictions': [(self.labels2[i], float(prediction2[0][i])) 
                                    for i in range(len(prediction2[0]))]
            }
            
            return result1, result2
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            return None, None

def get_conservation_info(species_class, conservation_class):
    """Memberikan informasi konservasi berdasarkan hasil klasifikasi"""
    info = {}
    
    # Get species info from config
    species_name = species_class.split()[-1] if species_class else ""
    info['species'] = SPECIES_INFO.get(species_name, {
        'description': "Informasi spesies tidak tersedia",
        'habitat': "Tidak diketahui",
        'importance': "Perlu penelitian lebih lanjut"
    })
    
    # Get conservation info from config
    info['conservation'] = CONSERVATION_STATUS.get(conservation_class, {
        "color": "#95a5a6",
        "icon": "❓",
        "urgency": "UNKNOWN",
        "description": "Status konservasi tidak dapat ditentukan",
        "actions": ["Konsultasikan dengan ahli biologi laut"],
        "legal_basis": "Tidak tersedia"
    })
    
    return info

def get_confidence_color(confidence):
    """Get color based on confidence level"""
    if confidence > CONFIDENCE_THRESHOLDS['high']:
        return "🟢", "confidence-high"
    elif confidence > CONFIDENCE_THRESHOLDS['medium']:
        return "🟡", "confidence-medium"
    else:
        return "🔴", "confidence-low"

def main():
    st.title("🐠 Klasifikasi Biota Laut")
    st.markdown("### Sistem Identifikasi Jenis dan Status Konservasi Biota Laut")
    
    # Initialize classifier
    if 'classifier' not in st.session_state:
        with st.spinner("Memuat model klasifikasi..."):
            try:
                st.session_state.classifier = MarineClassifier()
                st.success("✅ Sistem klasifikasi siap digunakan!")
            except Exception as e:
                st.error(f"❌ Error initializing classifier: {str(e)}")
                st.stop()
    
    classifier = st.session_state.classifier
    
    # Debug information
    with st.expander("🔧 Debug Information", expanded=False):
        st.write("Model 1 loaded:", classifier.model1 is not None)
        st.write("Model 2 loaded:", classifier.model2 is not None)
        st.write("Labels 1 count:", len(classifier.labels1))
        st.write("Labels 2 count:", len(classifier.labels2))
        
        # Test with dummy data
        if st.button("🧪 Test dengan Data Dummy"):
            import numpy as np
            # Create dummy image data
            dummy_image = np.random.rand(224, 224, 3)
            dummy_image = (dummy_image * 255).astype(np.uint8)
            dummy_pil = Image.fromarray(dummy_image)
            
            st.write("Testing with dummy image...")
            test_result1, test_result2 = classifier.predict(dummy_pil)
            
            if test_result1 and test_result2:
                st.success("✅ Test berhasil!")
                st.write("Test Result 1:", test_result1['class'])
                st.write("Test Result 2:", test_result2['class'])
            else:
                st.error("❌ Test gagal")
    
    st.markdown("---")
    input_method = st.radio(
        "Pilih metode input:",
        ["📁 Upload Gambar", "📷 Ambil Foto dari Kamera"],
        horizontal=True
    )
    
    image = None
    
    if input_method == "📁 Upload Gambar":
        uploaded_file = st.file_uploader(
            "Upload gambar biota laut",
            type=UI_CONFIG['supported_formats'],
            help=f"Format yang didukung: {', '.join(UI_CONFIG['supported_formats']).upper()}"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    
    elif input_method == "📷 Ambil Foto dari Kamera":
        st.markdown("**Menggunakan Kamera:**")
        camera_input = st.camera_input("📷 Ambil foto biota laut")
        if camera_input is not None:
            image = Image.open(camera_input)
            st.success("✅ Foto berhasil diambil!")
    
    # Process image if available
    if image is not None:
        st.markdown("---")
        st.markdown("### 📸 Hasil Analisis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Gambar Input")
            st.image(image, caption="Gambar yang akan dianalisis", use_container_width=True)
            
            # Show image info
            st.markdown("**Info Gambar:**")
            st.write(f"- Format: {image.format}")
            st.write(f"- Mode: {image.mode}")
            st.write(f"- Ukuran: {image.size}")
        
        with col2:
            st.markdown("#### Hasil Klasifikasi")
            
            # Check if models are loaded
            if classifier.model1 is None or classifier.model2 is None:
                st.error("❌ Model tidak dapat dimuat. Periksa file model di folder Model_Klasifikasi/")
                st.stop()
            
            with st.spinner("🔄 Menganalisis gambar..."):
                try:
                    result1, result2 = classifier.predict(image)
                    st.write("DEBUG: Prediction completed")  # Debug info
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
                    st.write("Full error:", e)  # Debug info
                    result1, result2 = None, None
            
            if result1 is not None and result2 is not None:
                # Hasil klasifikasi jenis
                st.markdown("**🔍 Jenis Biota Laut:**")
                species_class = result1['class']
                species_confidence = result1['confidence']
                
                # Format confidence dengan warna
                confidence_icon, confidence_class = get_confidence_color(species_confidence)
                
                st.markdown(f"{confidence_icon} **{species_class}**")
                st.markdown(f"Confidence: {species_confidence:.1%}")
                
                # Hasil klasifikasi status konservasi
                st.markdown("**🛡️ Status Konservasi:**")
                conservation_class = result2['class']
                conservation_confidence = result2['confidence']
                
                conf_icon, conf_class = get_confidence_color(conservation_confidence)
                
                st.markdown(f"{conf_icon} **{conservation_class}**")
                st.markdown(f"Confidence: {conservation_confidence:.1%}")
                
                # Success indicator
                st.success("✅ Analisis selesai!")
                
            else:
                st.warning("⚠️ Tidak dapat menganalisis gambar. Periksa debug information di atas.")
                st.write("DEBUG: result1 =", result1)
                st.write("DEBUG: result2 =", result2)
    
        # Detailed information
        if result1 is not None and result2 is not None:
            st.markdown("---")
            st.markdown("### 📊 Informasi Detail")
            
            # Conservation information
            info = get_conservation_info(species_class, conservation_class)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 🐟 Informasi Spesies")
                species_info = info['species']
                
                st.info(f"**Deskripsi:** {species_info['description']}")
                st.markdown(f"**Habitat:** {species_info['habitat']}")
                st.markdown(f"**Peran Ekologis:** {species_info['importance']}")
                
                # Top predictions for species
                st.markdown("**Top 5 Prediksi Jenis:**")
                top5_species = sorted(result1['all_predictions'], key=lambda x: x[1], reverse=True)[:UI_CONFIG['display_top_predictions']]
                for i, (label, conf) in enumerate(top5_species, 1):
                    icon, _ = get_confidence_color(conf)
                    st.markdown(f"{i}. {icon} {label}: {conf:.1%}")
            
            with col2:
                st.markdown("#### 🛡️ Status Konservasi")
                status_info = info['conservation']
                
                # Status dengan warna sesuai konfigurasi
                status_color = status_info['color']
                status_icon = status_info['icon']
                urgency = status_info['urgency']
                
                if urgency == "CRITICAL":
                    st.error(f"**{status_icon} {urgency}**")
                elif urgency == "SUSTAINABLE":
                    st.success(f"**{status_icon} {urgency}**")
                else:
                    st.warning(f"**{status_icon} {urgency}**")
                
                st.markdown(f"**Deskripsi:** {status_info['description']}")
                
                # Actions
                st.markdown("**Tindakan yang Dianjurkan:**")
                for action in status_info['actions']:
                    st.markdown(f"• {action}")
                
                st.markdown(f"**Dasar Hukum:** {status_info['legal_basis']}")
                
                # Predictions for conservation status
                st.markdown("**Prediksi Status:**")
                for i, (label, conf) in enumerate(result2['all_predictions'], 1):
                    icon, _ = get_confidence_color(conf)
                    st.markdown(f"{i}. {icon} {label}: {conf:.1%}")
    
    # Usage instructions
    st.markdown("---")
    st.markdown("### 📋 Petunjuk Penggunaan")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **📁 Upload Gambar:**
        - Pilih gambar biota laut dari perangkat Anda
        - Format yang didukung: PNG, JPG, JPEG
        - Pastikan gambar jelas dan fokus pada biota laut
        """)
    
    with col2:
        st.markdown("""
        **📷 Ambil Foto:**
        - Gunakan kamera perangkat untuk mengambil foto langsung
        - Pastikan pencahayaan cukup
        - Arahkan kamera ke biota laut yang ingin diidentifikasi
        """)
    
    # Model information
    st.markdown("---")
    st.markdown("### 🤖 Informasi Model")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **Model 1 - Klasifikasi Jenis Biota Laut:**
        - 21 kelas jenis biota laut
        - Termasuk berbagai jenis ikan, mamalia laut, moluska, dll.
        - Dilatih menggunakan Teachable Machine
        """)
    
    with col2:
        st.markdown("""
        **Model 2 - Status Konservasi:**
        - 3 kategori status: Dilindungi, Masih Lestari, Bukan Biota Laut
        - Membantu dalam konservasi dan pengelolaan sumber daya laut
        - Mengacu pada status konservasi internasional
        """)

if __name__ == "__main__":
    main()