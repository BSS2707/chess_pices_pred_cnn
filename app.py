
import os
import streamlit as st
from PIL import Image

from model import load_trained_model, predict_piece, train_model


MODEL_PATH = "chess_piece_model.keras"
CLASS_NAMES_PATH = "class_names.json"


st.set_page_config(
    page_title="Chess Piece Detector",
    page_icon="♟",
    layout="centered"
)


st.title("Chess Piece Detector")
st.write("Upload an image or capture a chess piece using your camera.")


@st.cache_resource
def get_model():

    if not os.path.exists(MODEL_PATH):
        with st.spinner("Model not found. Training model..."):
            model, class_names = train_model()

        return model, class_names

    try:
        return load_trained_model()

    except Exception as e:
        st.error("The existing model could not be loaded.")
        st.warning(
            "Delete chess_piece_model.keras and run the app again "
            "to train a new model."
        )
        st.stop()


model, class_names = get_model()


option = st.radio(
    "Select Input",
    [
        "Upload Image",
        "Camera"
    ],
    horizontal=True
)


image = None


if option == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload Chess Piece Image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


else:

    camera_file = st.camera_input(
        "Take a photo of the chess piece"
    )

    if camera_file is not None:

        image = Image.open(
            camera_file
        ).convert("RGB")


if image is not None:

    st.image(
        image,
        caption="Input Image",
        width=350
    )

    with st.spinner("Detecting chess piece..."):

        result = predict_piece(
            image,
            model,
            class_names
        )


    st.divider()

    st.subheader(
        result["display_name"]
    )

    st.success(
        f"Chess Piece: {result['class_name']}"
    )

    st.metric(
        "Confidence",
        f"{result['acc_percentage']:.2f}%"
    )


    st.subheader("Prediction Probabilities")

    for name, probability in result[
        "probabilities"
    ].items():

        st.write(
            f"{name}: {probability:.2f}%"
        )

        st.progress(
            min(
                int(probability),
                100
            )
        )

