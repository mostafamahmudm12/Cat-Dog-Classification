import streamlit as st
import requests
import json
from typing import List, Optional
import base64
from PIL import Image
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuration
API_BASE_URL = "http://localhost:8000"  # Update this to your FastAPI server URL
API_KEY = ""  # Set your API key here

# Page configuration
st.set_page_config(
    page_title="Image Classification App",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
    }
    .result-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .error-message {
        color: #dc3545;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 4px;
        padding: 0.75rem;
        margin: 1rem 0;
    }
    .success-message {
        color: #155724;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 4px;
        padding: 0.75rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def check_api_connection():
    """Check if the API is accessible"""
    try:
        headers = {"X-API-Key": API_KEY} if API_KEY else {}
        response = requests.get(f"{API_BASE_URL}/", headers=headers, timeout=5)
        return response.status_code == 200
    except:
        return False

def classify_images_memory(files: List[bytes], filenames: List[str]) -> Optional[dict]:
    """Send images to the API for classification (in-memory)"""
    try:
        headers = {"X-API-Key": API_KEY} if API_KEY else {}
        
        # Prepare files for the request
        files_data = []
        for i, (file_bytes, filename) in enumerate(zip(files, filenames)):
            files_data.append(("files", (filename, file_bytes, "image/jpeg")))
        
        response = requests.post(
            f"{API_BASE_URL}/classify-batch-memory",
            files=files_data,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return None

def classify_images_paths(files: List[bytes], filenames: List[str]) -> Optional[dict]:
    """Send images to the API for classification (file-based)"""
    try:
        headers = {"X-API-Key": API_KEY} if API_KEY else {}
        
        # Prepare files for the request
        files_data = []
        for i, (file_bytes, filename) in enumerate(zip(files, filenames)):
            files_data.append(("files", (filename, file_bytes, "image/jpeg")))
        
        response = requests.post(
            f"{API_BASE_URL}/classify-batch-paths",
            files=files_data,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return None

def display_prediction_results(results: dict, uploaded_files):
    """Display prediction results with images and charts"""
    if not results or "predictions" not in results:
        st.error("No predictions found in the response")
        return
    
    predictions = results["predictions"]
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Classification Results")
        
        # Display each prediction with its image
        for i, (pred, uploaded_file) in enumerate(zip(predictions, uploaded_files)):
            with st.expander(f"Image {i+1}: {uploaded_file.name}", expanded=True):
                img_col, result_col = st.columns([1, 2])
                
                with img_col:
                    # Display the image
                    image = Image.open(uploaded_file)
                    st.image(image, caption=uploaded_file.name, use_column_width=True)
                
                with result_col:
                    st.write("**Prediction:**", pred["predicted_class"])
                    st.write("**Confidence:**", f"{pred['confidence']:.2%}")
                    
                    # Progress bar for confidence
                    st.progress(pred["confidence"])
                    
                    # Display top probabilities if available
                    if "probabilities" in pred:
                        st.write("**Top Probabilities:**")
                        probs_df = pd.DataFrame(
                            list(pred["probabilities"].items()),
                            columns=["Class", "Probability"]
                        )
                        probs_df = probs_df.sort_values("Probability", ascending=False).head(5)
                        probs_df["Probability"] = probs_df["Probability"].apply(lambda x: f"{x:.2%}")
                        st.dataframe(probs_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("📈 Summary Statistics")
        
        # Overall statistics
        confidences = [pred["confidence"] for pred in predictions]
        avg_confidence = sum(confidences) / len(confidences)
        
        st.metric("Average Confidence", f"{avg_confidence:.2%}")
        st.metric("Total Images", len(predictions))
        st.metric("Processing Time", f"{results.get('processing_time', 0):.2f}s")
        
        # Confidence distribution
        fig_conf = px.histogram(
            x=confidences,
            nbins=10,
            title="Confidence Distribution",
            labels={"x": "Confidence", "y": "Count"}
        )
        fig_conf.update_layout(height=300)
        st.plotly_chart(fig_conf, use_container_width=True)
        
        # Class distribution
        classes = [pred["predicted_class"] for pred in predictions]
        class_counts = pd.Series(classes).value_counts()
        
        fig_classes = px.pie(
            values=class_counts.values,
            names=class_counts.index,
            title="Predicted Classes Distribution"
        )
        fig_classes.update_layout(height=300)
        st.plotly_chart(fig_classes, use_container_width=True)

def main():
    st.markdown('<h1 class="main-header">🖼️ Image Classification App</h1>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API configuration
        global API_BASE_URL, API_KEY
        API_BASE_URL = st.text_input("API Base URL", value=API_BASE_URL)
        API_KEY = st.text_input("API Key", value=API_KEY, type="password")
        
        # Classification method
        classification_method = st.selectbox(
            "Classification Method",
            ["Memory-based (Small batches)", "File-based (Large batches)"],
            help="Memory-based is faster for small batches, file-based is better for large batches"
        )
        
        # Check API connection
        if st.button("🔗 Test API Connection"):
            if check_api_connection():
                st.success("✅ API connection successful!")
            else:
                st.error("❌ Cannot connect to API. Please check the URL and API key.")
    
    # Main content area
    st.header("📤 Upload Images for Classification")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose image files",
        type=['png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff'],
        accept_multiple_files=True,
        help="Upload one or more images to classify"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
        
        # Show preview of uploaded images
        if len(uploaded_files) <= 5:  # Show preview for small batches
            st.subheader("📋 Preview")
            cols = st.columns(min(len(uploaded_files), 5))
            for i, uploaded_file in enumerate(uploaded_files[:5]):
                with cols[i]:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=uploaded_file.name, use_column_width=True)
        
        # Classification button
        if st.button("🚀 Classify Images", type="primary"):
            if not API_KEY:
                st.warning("⚠️ Please set your API key in the sidebar")
                return
            
            # Prepare files for API call
            files_data = []
            filenames = []
            
            with st.spinner("Processing images..."):
                for uploaded_file in uploaded_files:
                    # Reset file pointer
                    uploaded_file.seek(0)
                    files_data.append(uploaded_file.read())
                    filenames.append(uploaded_file.name)
                
                # Call the appropriate API endpoint
                if "Memory-based" in classification_method:
                    results = classify_images_memory(files_data, filenames)
                else:
                    results = classify_images_paths(files_data, filenames)
                
                if results:
                    # Reset file pointers for display
                    for uploaded_file in uploaded_files:
                        uploaded_file.seek(0)
                    
                    st.success("🎉 Classification completed!")
                    display_prediction_results(results, uploaded_files)
                    
                    # Download results as JSON
                    json_str = json.dumps(results, indent=2)
                    st.download_button(
                        label="📥 Download Results (JSON)",
                        data=json_str,
                        file_name="classification_results.json",
                        mime="application/json"
                    )
                else:
                    st.error("❌ Classification failed. Please check your API connection and try again.")
    
    else:
        st.info("👆 Please upload one or more images to get started")
        
        # Show example section
        st.header("💡 How to Use")
        st.markdown("""
        1. **Configure API**: Set your API base URL and key in the sidebar
        2. **Upload Images**: Click the upload button and select your image files
        3. **Choose Method**: Select memory-based for small batches or file-based for large batches
        4. **Classify**: Click the "Classify Images" button to process your images
        5. **View Results**: See predictions with confidence scores and visualizations
        6. **Download**: Save your results as a JSON file
        """)

if __name__ == "__main__":
    main()