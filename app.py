import os
from pathlib import Path
from PIL import Image
import streamlit as st
from model import load_trained_model, predict_piece

# Absolute path resolution matching model.py
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "chess_piece_model.h5"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

st.set_page_config(
    page_title="Chess Piece Detector", page_icon="♟", layout="centered"
)

st.title("Chess Piece Detector ♟")
st.write("Upload an image or capture a chess piece using your camera.")


@st.cache_resource(show_spinner="Loading TensorFlow model into memory...")
def get_model():
    """Caches the model so it loads only once per deployment container session."""
    try:
        model, class_names = load_trained_model()
        return model, class_names
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        st.info(
            "Ensure 'chess_piece_model.h5' and 'class_names.json' are present in your repo."
        )
        st.stop()


# Load model safely using resource caching
model, class_names = get_model()

option = st.radio(
    "Select Input Method", ["Upload Image", "Camera"], horizontal=True
)

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload Chess Piece Image", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

else:
    camera_file = st.camera_input("Take a photo of the chess piece")
    if camera_file is not None:
        image = Image.open(camera_file).convert("RGB")


# Display & Inference section (wrapped safely inside conditional block)
if image is not None:
    st.image(image, caption="Input Image", use_container_width=True)

    with st.spinner("Detecting chess piece..."):
        result = predict_piece(image, model, class_names)

    st.divider()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader(f"{result['emoji']} {result['display_name']}")
        st.success(f"Class: **{result['class_name']}**")
    with col2:
        st.metric(
            label="Confidence Score", value=f"{result['acc_percentage']:.1f}%"
        )

    st.write("---")
    st.subheader("Prediction Probabilities")

    # Sort probabilities descending for clean UI representation
    sorted_probs = sorted(
        result["probabilities"].items(), key=lambda x: x[1], reverse=True
    )

    for name, probability in sorted_probs:
        st.write(f"**{name}**: {probability:.1f}%")
        st.progress(min(max(int(probability), 0), 100))