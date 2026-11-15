import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import os

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Pneumonia Detection from Chest X-rays",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# EXACT preprocessing from your training pipeline
# ------------------------------------------------------------------
def preprocess_image(image):
    """
    Preprocess a single image exactly as your ImageDataGenerator did:
      1. Resize to (224,224)
      2. Convert to float32
      3. Rescale by 1/255.
      4. Samplewise center (subtract mean of this image)
      5. Samplewise std normalization (divide by std of this image)
    """
    try:
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image

        # Ensure RGB (if grayscale, stack)
        if len(img.shape) == 2:
            img = np.stack((img,) * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        # Resize to (224,224) – same as training
        img = cv2.resize(img, (224, 224))

        # Convert to float32
        img = img.astype(np.float32)

        # 1. Rescale by 1/255.
        img = img / 255.0

        # 2. Samplewise center: subtract mean
        mean = np.mean(img)
        img = img - mean

        # 3. Samplewise std normalization: divide by std (with safety)
        std = np.std(img)
        if std == 0:
            std = 1e-7
        img = img / std

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        return img
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        # Update this path to where your model is stored
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'pneumonia_detection_model.keras')
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            return model
        else:
            st.error(f"Model file not found at: {os.path.abspath(model_path)}. See models/README.md for the download link.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = 0
if 'normal_count' not in st.session_state:
    st.session_state.normal_count = 0
if 'pneumonia_count' not in st.session_state:
    st.session_state.pneumonia_count = 0

# Fixed threshold (as per your request)
THRESHOLD = 0.5

# ------------------------------------------------------------------
# Main app
# ------------------------------------------------------------------
def main():
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/4228/4228684.png", width=100)
        st.title("Navigation")
        menu_options = ["🏠 Home", "📋 About Pneumonia", "📊 Dataset & Results", "®️ Model Info", "ℹ️ How It Works"]
        choice = st.radio("Go to", menu_options)

        st.markdown("---")
        st.markdown("### Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Predictions", st.session_state.predictions_made)
        with col2:
            if st.session_state.predictions_made > 0:
                pneumonia_percent = (st.session_state.pneumonia_count / st.session_state.predictions_made) * 100
                st.metric("Pneumonia Cases", f"{pneumonia_percent:.1f}%")

        st.markdown("---")
        st.markdown("### About")
        st.info("AI-powered pneumonia detection from chest X-rays. Always consult a doctor.")

        # Thank you note
        st.markdown("---")
        st.markdown("### 👨‍🏫 Special Thanks")
        st.success("**Dr. Mohamed Zorkany**\n\nFor his invaluable guidance and support throughout this project.")

    if choice == "🏠 Home":
        home_page()
    elif choice == "📋 About Pneumonia":
        about_pneumonia_page()
    elif choice == "📊 Dataset & Results":
        statistics_page()
    elif choice == "®️ Model Info":
        model_info_page()
    elif choice == "ℹ️ How It Works":
        how_it_works_page()

# ------------------------------------------------------------------
# Home page with prediction
# ------------------------------------------------------------------
def home_page():
    st.title("🫁 Pneumonia Detection from Chest X-rays")
    st.markdown("---")

    # Welcome message
    st.markdown("""
    <div style='background-color: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h3 style='color: white;'>Welcome to the Pneumonia Detection System</h3>
        <p style='color: rgba(255,255,255,0.9);'>This tool uses a deep learning model (ResNet50) to analyze chest X-rays and detect signs of pneumonia. 
        Upload a chest X-ray image below to get started.</p>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    model = load_model()
    if model is None:
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📤 Upload Chest X-ray Image")
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a frontal chest X-ray"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray", use_container_width=True)

            if st.button("🔍 Analyze X-ray", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    # Preprocess exactly as in training
                    processed = preprocess_image(image)
                    if processed is None:
                        return

                    # Predict
                    pred = model.predict(processed, verbose=0)
                    probability = float(pred[0][0])   # raw probability (sigmoid output)

                    # Determine class using fixed threshold
                    if probability > THRESHOLD:
                        predicted_class = "PNEUMONIA"
                        confidence = probability
                    else:
                        predicted_class = "NORMAL"
                        confidence = 1 - probability

                    # Update stats
                    st.session_state.predictions_made += 1
                    if predicted_class == "PNEUMONIA":
                        st.session_state.pneumonia_count += 1
                    else:
                        st.session_state.normal_count += 1

                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Analysis Results")

                    res_col1, res_col2, res_col3 = st.columns(3)
                    with res_col1:
                        if predicted_class == "PNEUMONIA":
                            st.error(f"### 🚨 {predicted_class}")
                        else:
                            st.success(f"### ✅ {predicted_class}")
                    with res_col2:
                        st.metric("Confidence", f"{confidence*100:.2f}%")
                    with res_col3:
                        st.metric("Raw Prob", f"{probability:.3f}")

                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Confidence Score"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#ff4b4b" if predicted_class == "PNEUMONIA" else "#00cc96"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 100], 'color': "gray"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)

                    # Additional information
                    with st.expander("📋 Detailed Analysis"):
                        st.markdown(f"""
                        - **Prediction:** {predicted_class}
                        - **Raw Model Output:** {probability:.4f}
                        - **Confidence:** {confidence*100:.2f}%
                        - **Model Architecture:** ResNet50
                        - **Input Size:** 224x224 pixels
                        - **Threshold:** {THRESHOLD}
                        """)

                    # Disclaimer
                    st.warning("⚠️ **Disclaimer**: This is an AI-assisted tool and should not replace professional medical diagnosis. Please consult with a healthcare provider.")

    with col2:
        st.subheader("ℹ️ Quick Guide")

        # Tabs
        tab1, tab2 = st.tabs(["📝 Instructions", "⚡ Tips"])
        with tab1:
            st.markdown("""
            **How to use:**
            1. Upload a frontal chest X‑ray (JPG/PNG).
            2. Click **Analyze**.
            3. View the result and confidence.
            4. Always consult a doctor.
            """)
        with tab2:
            st.markdown("""
            - Use high‑quality, well‑centered images.
            - Avoid blurred or low‑contrast images.
            - Frontal view works best.
            """)

        st.markdown("---")
        st.subheader("📈 Session Stats")
        st.metric("Images Analyzed", st.session_state.predictions_made)
        if st.session_state.predictions_made > 0:
            pct = (st.session_state.pneumonia_count / st.session_state.predictions_made) * 100
            st.metric("Pneumonia Rate", f"{pct:.1f}%")

# ------------------------------------------------------------------
# About Pneumonia page
# ------------------------------------------------------------------
def about_pneumonia_page():
    st.title("📋 About Pneumonia")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("What is Pneumonia?")
        st.markdown("""
        Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, 
        causing cough with phlegm or pus, fever, chills, and difficulty breathing.

        **Key Facts:**
        - Pneumonia is the single largest infectious cause of death in children worldwide
        - It can be caused by bacteria, viruses, or fungi
        - Vaccination can prevent some types of pneumonia
        - Early detection significantly improves treatment outcomes
        """)
    with col2:
        st.image("https://img.icons8.com/color/240/000000/lungs.png", use_container_width=True)

    st.header("🚨 Common Symptoms")
    sym_col1, sym_col2, sym_col3 = st.columns(3)
    with sym_col1:
        st.markdown("""
        - 🤒 **Chest pain** when breathing or coughing
        - 🌡️ **Fever** and chills
        - 😮‍💨 **Shortness of breath**
        """)
    with sym_col2:
        st.markdown("""
        - 🫁 **Cough** with phlegm
        - 😴 **Fatigue** and tiredness
        - 🤢 **Nausea** or vomiting
        """)
    with sym_col3:
        st.markdown("""
        - 😓 **Sweating** and shaking
        - 🫤 **Loss of appetite**
        - 🥶 **Confusion** (in elderly)
        """)

    st.header("⚠️ Risk Factors")
    risk_data = {
        "Age": ["Children under 5", "Adults over 65"],
        "Lifestyle": ["Smoking", "Excessive alcohol use", "Malnutrition"],
        "Medical Conditions": ["Chronic lung disease", "Diabetes", "Heart disease", "Weakened immune system"],
        "Environment": ["Hospital stay", "Nursing home residence", "Exposure to pollutants"]
    }
    for category, factors in risk_data.items():
        st.subheader(f"**{category}**")
        st.markdown(" • " + " • ".join(factors))

    st.header("🛡️ Prevention")
    prev_col1, prev_col2 = st.columns(2)
    with prev_col1:
        st.markdown("""
        **Vaccination:**
        - 💉 Pneumococcal vaccine
        - 💉 Flu vaccine
        - 💉 COVID-19 vaccine
        """)
    with prev_col2:
        st.markdown("""
        **Healthy Habits:**
        - 🧼 Wash hands regularly
        - 🚭 Don't smoke
        - 💪 Maintain strong immune system
        - 🏃 Regular exercise
        """)

# ------------------------------------------------------------------
# Dataset & Results page (real numbers from the training notebook)
# ------------------------------------------------------------------
def statistics_page():
    st.title("📊 Dataset & Results")
    st.markdown("---")

    st.header("🗂️ Dataset: Kaggle Chest X-Ray Images (Pneumonia)")
    split_df = pd.DataFrame({
        "Split": ["Train", "Train", "Test", "Test", "Validation", "Validation"],
        "Class": ["Normal", "Pneumonia"] * 3,
        "Images": [1341, 3875, 234, 390, 8, 8],
    })
    fig = px.bar(split_df, x="Split", y="Images", color="Class", barmode="group",
                 text="Images", title="Images per split and class",
                 color_discrete_map={"Normal": "#00cc96", "Pneumonia": "#ff4b4b"})
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    st.info("The training set is imbalanced (about 2.9x more pneumonia than normal), "
            "so class weights were used during training. The official validation split has only 16 images.")

    st.header("📈 Test-set results (624 images)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", "85.1%")
    c2.metric("Precision", "87.2%")
    c3.metric("Recall", "89.2%")
    c4.metric("AUC", "0.908")
    st.caption("Measured with model.evaluate on the official test split, threshold 0.5. "
               "Test images come from a different distribution than training, so expect lower numbers than on train data.")

# ------------------------------------------------------------------
# Model Info page
# ------------------------------------------------------------------
def model_info_page():
    st.title("®️ ResNet50 Model Information")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **ResNet50 (Residual Networks)** is a deep convolutional neural network that is 50 layers deep.

        ### Key Features:
        - **Depth:** 50 layers
        - **Input Size:** 224x224 pixels
        - **Parameters:** about 23.6 million (backbone, frozen in this project)
        - **Pretrained on:** ImageNet

        ### Why ResNet50 for Pneumonia Detection?
        - ✅ Excellent feature extraction
        - ✅ Proven on medical imaging
        - ✅ Transfer learning capability
        """)
    with col2:
        st.image("https://miro.medium.com/max/1400/1*ZCNQJvN9R3PvFjJXpE1MqA.png", 
                caption="ResNet Architecture", use_container_width=True)

    st.header("📊 Model Performance Metrics (test set)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", "85.1%")
    with col2:
        st.metric("Precision", "87.2%")
    with col3:
        st.metric("Recall", "89.2%")


    st.header("🎓 Training Details")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Dataset Information:**
        - **Source:** Kaggle - Chest X-Ray Images (Pneumonia)
        - **Train:** 5,216 images (1,341 normal / 3,875 pneumonia)
        - **Validation:** 16 images
        - **Test:** 624 images (234 normal / 390 pneumonia)
        """)
    with col2:
        st.markdown("""
        **Training Parameters:**
        - **Base model:** ResNet50 (ImageNet weights, frozen)
        - **Head:** BatchNorm + Dense 512/256/128 + Dropout 0.5 + sigmoid
        - **Optimizer:** Adam (default learning rate)
        - **Batch Size:** 32, up to 50 epochs with early stopping
        - **Loss Function:** Binary Crossentropy (class-weighted)
        """)

# ------------------------------------------------------------------
# How It Works page
# ------------------------------------------------------------------
def how_it_works_page():
    st.title("ℹ️ How It Works")
    st.markdown("---")

    st.header("🔬 Detection Workflow")
    cols = st.columns(5)
    steps = ["📤 Upload", "🖼️ Preprocess", "🤖 ResNet50", "📊 Analyze", "📋 Result"]
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            st.markdown(f"""
            <div style='text-align: center; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 10px;'>
                <h3>Step {i+1}</h3>
                <p>{step}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.header("1. Image Upload & Preprocessing")
        st.markdown("""
        **What happens:**
        - User uploads a chest X-ray image
        - Image is converted to RGB format
        - Resized to 224x224 pixels
        - Pixel values normalized (rescale, mean subtraction, std division) exactly as during training
        """)

        st.header("2. ResNet50 Analysis")
        st.markdown("""
        **How it works:**
        - Image passes through 50 layers
        - Each layer extracts features
        - Final layers identify pneumonia patterns
        """)

    with col2:
        st.header("3. Results Interpretation")
        st.markdown("""
        **Understanding Results:**
        - **Score > 0.5:** Pneumonia detected
        - **Score < 0.5:** Normal findings
        - **Confidence:** Indicates certainty level

        **Result Categories:**
        - **90-100%:** High confidence
        - **70-89%:** Moderate confidence
        - **50-69%:** Low confidence
        - **<50%:** Very low confidence
        """)

# ------------------------------------------------------------------
# Run the app
# ------------------------------------------------------------------
if __name__ == "__main__":
    main()