import streamlit as st
import os
import sys

# Add current directory to path
sys.path.append('.')

st.title("🧪 Debug Klasifikasi Biota Laut")

# Test configuration import
try:
    from classification_config import MODEL_PATHS
    st.success("✅ Configuration imported successfully")
    st.write("Model paths:", MODEL_PATHS)
except Exception as e:
    st.error(f"❌ Configuration import error: {e}")

# Test file existence
st.markdown("### File Existence Check")
for key, path in MODEL_PATHS.items():
    exists = os.path.exists(path)
    if exists:
        size = os.path.getsize(path)
        st.success(f"✅ {key}: {path} ({size} bytes)")
    else:
        st.error(f"❌ {key}: {path} not found")

# Test TensorFlow import
st.markdown("### TensorFlow Test")
try:
    import tensorflow as tf
    st.success(f"✅ TensorFlow {tf.__version__} imported successfully")
except Exception as e:
    st.error(f"❌ TensorFlow import error: {e}")

# Test marine classification import
st.markdown("### Marine Classification Module Test")
try:
    from marine_classification import MarineClassifier, get_confidence_color
    st.success("✅ Marine classification module imported successfully")
    
    # Test confidence color function
    test_conf = 0.9
    icon, css_class = get_confidence_color(test_conf)
    st.write(f"Test confidence {test_conf}: {icon} ({css_class})")
    
except Exception as e:
    st.error(f"❌ Marine classification import error: {e}")
    st.write("Error details:", str(e))

# Test model loading
if st.button("🧪 Test Model Loading"):
    try:
        with st.spinner("Loading models..."):
            classifier = MarineClassifier()
        
        if classifier.model1 is not None and classifier.model2 is not None:
            st.success("✅ Models loaded successfully!")
            st.write("Model 1 input shape:", classifier.model1.input_shape)
            st.write("Model 2 input shape:", classifier.model2.input_shape)
            st.write("Labels 1 count:", len(classifier.labels1))
            st.write("Labels 2 count:", len(classifier.labels2))
        else:
            st.error("❌ Models failed to load")
            
    except Exception as e:
        st.error(f"❌ Model loading error: {e}")
        st.write("Error details:", str(e))