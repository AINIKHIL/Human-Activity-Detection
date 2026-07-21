"""
app.py
Streamlit app: upload a video (or use your webcam) and see the predicted
activity + confidence score.

Run with:
    streamlit run app.py
"""

import time
import tempfile
from collections import deque

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

from utils import extract_frames, SEQUENCE_LENGTH, IMG_SIZE

st.set_page_config(page_title="Human Activity Recognition", layout="centered")

# ---- Load model + class names once and cache them across reruns ----
@st.cache_resource
def load_har_model():
    model = load_model("har_model.h5")
    with open("class_names.txt") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

model, class_names = load_har_model()

# If your dataset's "fall" class is named something else, change this to match.
FALL_CLASS_NAME = "Falling"
FALL_CONFIDENCE_THRESHOLD = 0.6

st.title("🏃 Human Activity Recognition")
st.write("Upload a video and the model will predict the activity being performed.")

tab_upload, tab_webcam = st.tabs(["📁 Upload Video", "📷 Live Webcam"])

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
            X = np.expand_dims(frames, axis=0)  # add the batch dimension the model expects
            predictions = model.predict(X, verbose=0)[0]

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
        "Grabs frames from your webcam into a rolling window and "
        "re-predicts the activity roughly every couple of seconds. "
        "Note: this reads the webcam attached to the machine running "
        "`streamlit run app.py` -- see README.md for the note on browser-based "
        "webcam access (needed if you deploy this to Streamlit Cloud)."
    )
    run = st.checkbox("Start webcam")
    frame_placeholder = st.empty()
    result_placeholder = st.empty()
    timeline_placeholder = st.empty()

    if run:
        import cv2
        cap = cv2.VideoCapture(0)
        frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        timeline = []  # (time, activity) pairs -- the "activity timeline" bonus feature

        while run:
            success, frame = cap.read()
            if not success:
                st.warning("Could not access webcam.")
                break

            frame_placeholder.image(frame, channels="BGR")

            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))
            frame_buffer.append(small.astype(np.float32) / 255.0)

            if len(frame_buffer) == SEQUENCE_LENGTH:
                X = np.expand_dims(np.array(frame_buffer), axis=0)
                predictions = model.predict(X, verbose=0)[0]
                predicted_idx = np.argmax(predictions)
                predicted_class = class_names[predicted_idx]
                confidence = predictions[predicted_idx]

                result_placeholder.subheader(
                    f"Predicted Activity: {predicted_class} | Confidence Score: {confidence * 100:.1f}%"
                )

                if predicted_class == FALL_CLASS_NAME and confidence >= FALL_CONFIDENCE_THRESHOLD:
                    result_placeholder.error("⚠️ Fall detected!")

                timeline.append((time.strftime("%H:%M:%S"), predicted_class))
                timeline_placeholder.table(timeline[-10:])  # show the last 10 predictions

            time.sleep(0.1)

        cap.release()
