#!/usr/bin/env python3
"""
Simple model loading test without Streamlit dependencies
"""

import os
import tensorflow as tf
import numpy as np
from PIL import Image

print("TensorFlow version:", tf.__version__)
print("Current directory:", os.getcwd())

# Test file existence
model1_path = 'Model_Klasifikasi/keras_model1.h5'
model2_path = 'Model_Klasifikasi/keras_model2.h5'

print(f"Model 1 exists: {os.path.exists(model1_path)}")
print(f"Model 2 exists: {os.path.exists(model2_path)}")

try:
    print("Loading Model 1...")
    model1 = tf.keras.models.load_model(model1_path, compile=False)
    print("✅ Model 1 loaded successfully")
    print(f"Input shape: {model1.input_shape}")
    
    print("Loading Model 2...")
    model2 = tf.keras.models.load_model(model2_path, compile=False)
    print("✅ Model 2 loaded successfully")
    print(f"Input shape: {model2.input_shape}")
    
    # Test prediction with dummy data
    print("Testing prediction with dummy data...")
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    
    pred1 = model1.predict(dummy_input, verbose=0)
    pred2 = model2.predict(dummy_input, verbose=0)
    
    print(f"Model 1 prediction shape: {pred1.shape}")
    print(f"Model 2 prediction shape: {pred2.shape}")
    print(f"Model 1 max class: {np.argmax(pred1)}")
    print(f"Model 2 max class: {np.argmax(pred2)}")
    
    print("✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()