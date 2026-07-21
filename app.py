"""
app.py
Streamlit app: upload a video (or use your webcam) and see the predicted
activity + confidence score.

Run with:
    streamlit run app.py
"""

import tempfile

import numpy as np
import onnxruntime as ort
import streamlit as st

from utils import extract_frames

st.set_page_config(page_title="Human Activity Recognition", layout="centered")

# ---- Load model + class names once and cache them across reruns ----
@st.cache_resource
def load_har_model():
    session = ort.InferenceSession("har_model.onnx", providers=["CPUExecutionProvider"])
    with open("class_names.txt") as f:
        class_names = [line.strip() for line in f.readlines()]
    return session, class_names

session, class_names = load_har_model()


def predict_activity(frames):
    input_name = session.get_inputs()[0].name
    inputs = np.expand_dims(frames, axis=0).astype(np.float32)
    return session.run(None, {input_name: inputs})[0][0]

# If your dataset's "fall" class is named something else, change this to match.
FALL_CLASS_NAME = "Falling"
FALL_CONFIDENCE_THRESHOLD = 0.6

st.title("🏃 Human Activity Recognition")
st.write("Upload a video and the model will predict the activity being performed.")

tab_upload, tab_webcam = st.tabs(["📁 Upload Video", "📷 Webcam"])

# ============================================================
# TAB 1: Upload a video file
# ============================================================
with tab_upload:
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Save to a temp file so OpenCV (which reads from disk paths) can open it
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())

        st.video(uploaded_file)

        with st.spinner("Analyzing video..."):
            frames = extract_frames(tfile.name)

        if frames is None:
            st.error("Could not read this video. Try a different file.")
        else:
            predictions = predict_activity(frames)

            predicted_idx = np.argmax(predictions)
            predicted_class = class_names[predicted_idx]
            confidence = predictions[predicted_idx]

            st.subheader(f"Predicted Activity: **{predicted_class}**")
            st.metric("Confidence Score", f"{confidence * 100:.1f}%")

            if predicted_class == FALL_CLASS_NAME and confidence >= FALL_CONFIDENCE_THRESHOLD:
                st.error("⚠️ Fall detected!")

            st.write("All class probabilities:")
            st.bar_chart({name: float(prob) for name, prob in zip(class_names, predictions)})

# ============================================================
# TAB 2: Live webcam prediction (bonus feature)
# ============================================================
with tab_webcam:
    st.write(
        "This deployed app supports public video upload prediction. "
        "Server-side webcam capture is disabled because Streamlit Community Cloud "
        "cannot access a visitor's browser camera through OpenCV."
    )
    st.info("Use the Upload Video tab to test the deployed app from any device.")
