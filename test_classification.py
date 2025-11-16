import streamlit as st
import os
from marine_classification import MarineClassifier
from PIL import Image
import numpy as np

def test_marine_classification():
    """
    Script testing untuk sistem klasifikasi biota laut
    """
    st.title("🧪 Test Marine Classification System")
    
    # Test model loading
    st.header("1. Model Loading Test")
    try:
        classifier = MarineClassifier()
        if classifier.model1 is not None and classifier.model2 is not None:
            st.success("✅ Both models loaded successfully!")
            st.write(f"Model 1 input shape: {classifier.model1.input_shape}")
            st.write(f"Model 2 input shape: {classifier.model2.input_shape}")
        else:
            st.error("❌ Failed to load models")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
    
    # Test labels loading
    st.header("2. Labels Loading Test")
    try:
        st.write(f"Species labels count: {len(classifier.labels1)}")
        st.write(f"Conservation labels count: {len(classifier.labels2)}")
        
        st.subheader("Species Labels:")
        for i, label in enumerate(classifier.labels1[:5]):  # Show first 5
            st.write(f"{i}: {label}")
        st.write("...")
        
        st.subheader("Conservation Labels:")
        for i, label in enumerate(classifier.labels2):
            st.write(f"{i}: {label}")
            
    except Exception as e:
        st.error(f"❌ Error loading labels: {str(e)}")
    
    # Test image preprocessing
    st.header("3. Image Preprocessing Test")
    try:
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        pil_image = Image.fromarray(dummy_image)
        
        processed = classifier.preprocess_image(pil_image)
        
        if processed is not None:
            st.success("✅ Image preprocessing successful!")
            st.write(f"Original shape: {dummy_image.shape}")
            st.write(f"Processed shape: {processed.shape}")
            st.write(f"Data type: {processed.dtype}")
            st.write(f"Value range: [{processed.min():.3f}, {processed.max():.3f}]")
        else:
            st.error("❌ Image preprocessing failed")
            
    except Exception as e:
        st.error(f"❌ Error in preprocessing: {str(e)}")
    
    # Test prediction (with dummy image)
    st.header("4. Prediction Test")
    try:
        if st.button("Run Prediction Test"):
            # Create a random test image
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            pil_test_image = Image.fromarray(test_image)
            
            with st.spinner("Running prediction..."):
                result1, result2 = classifier.predict(pil_test_image)
            
            if result1 and result2:
                st.success("✅ Prediction successful!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Species Classification")
                    st.write(f"Predicted: {result1['class']}")
                    st.write(f"Confidence: {result1['confidence']:.3f}")
                
                with col2:
                    st.subheader("Conservation Status")
                    st.write(f"Predicted: {result2['class']}")
                    st.write(f"Confidence: {result2['confidence']:.3f}")
            else:
                st.error("❌ Prediction failed")
                
    except Exception as e:
        st.error(f"❌ Error in prediction: {str(e)}")
    
    # System information
    st.header("5. System Information")
    try:
        import tensorflow as tf
        import PIL
        import cv2
        
        st.write(f"TensorFlow version: {tf.__version__}")
        st.write(f"Pillow version: {PIL.__version__}")
        st.write(f"OpenCV version: {cv2.__version__}")
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            st.success(f"✅ GPU available: {len(gpus)} device(s)")
        else:
            st.info("ℹ️ Running on CPU")
            
    except Exception as e:
        st.error(f"❌ Error getting system info: {str(e)}")

if __name__ == "__main__":
    test_marine_classification()