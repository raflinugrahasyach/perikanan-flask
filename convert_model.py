import tensorflow as tf
import os

print(f"TensorFlow versi: {tf.__version__}")

# --- PERBAIKAN WAJIB ---
# Kita harus mendefinisikan ulang kelas kustom ini
# agar 'load_model' bisa memuat arsitektur Keras 2.x
# yang memiliki argumen 'groups'.
# Ini disalin dari app.py Anda.
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(*args, **kwargs)
# --- AKHIR PERBAIKAN ---

# Definisikan custom objects dictionary
CUSTOM_OBJECTS = {'CustomDepthwiseConv2D': CustomDepthwiseConv2D}

# Tentukan path model Anda
MODEL_DIR = 'Model_Klasifikasi'
MODEL_H5_1 = os.path.join(MODEL_DIR, 'keras_model1.h5')
MODEL_H5_2 = os.path.join(MODEL_DIR, 'keras_model2.h5')

# Tentukan path output (tempat model baru akan disimpan)
MODEL_TFLITE_1 = os.path.join(MODEL_DIR, 'model1.tflite')
MODEL_TFLITE_2 = os.path.join(MODEL_DIR, 'model2.tflite')

def convert_model(h5_path, tflite_path, custom_objects):
    """
    Memuat model Keras (.h5) dan mengonversinya ke TensorFlow Lite (.tflite).
    """
    if not os.path.exists(h5_path):
        print(f"ERROR: File model tidak ditemukan di: {h5_path}")
        return

    print(f"\nMemuat model dari: {h5_path}...")
    try:
        # Muat model dengan custom objects
        model = tf.keras.models.load_model(
            h5_path, 
            custom_objects=custom_objects, 
            compile=False
        )
        print("Model H5 berhasil dimuat.")
        
        # Inisialisasi TFLiteConverter
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Tambahkan optimasi (opsional tapi disarankan)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Lakukan konversi
        tflite_model = converter.convert()
        print("Model berhasil dikonversi ke TFLite.")
        
        # Simpan model .tflite baru
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"BERHASIL! Model baru disimpan di: {tflite_path}")
        print(f"Ukuran file: {os.path.getsize(tflite_path) / 1024:.2f} KB")
        
    except Exception as e:
        print(f"GAGAL mengonversi {h5_path}: {e}")
        # Cetak error 'groups=1' jika terjadi lagi
        if 'groups' in str(e):
            print("ERROR: Pastikan kelas CustomDepthwiseConv2D sudah benar.")

# --- Jalankan Konversi ---
if __name__ == "__main__":
    # Konversi model pertama
    convert_model(MODEL_H5_1, MODEL_TFLITE_1, CUSTOM_OBJECTS)
    
    # Konversi model kedua
    convert_model(MODEL_H5_2, MODEL_TFLITE_2, CUSTOM_OBJECTS)
    
    print("\n--- Konversi Selesai ---")