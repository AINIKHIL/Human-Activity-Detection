"""
utils.py
Shared helper function for turning a video file into a fixed-size sequence
of frames. Used by BOTH the training pipeline (data_prep.py) and the
Streamlit app (app.py) so that training and inference always preprocess
video in exactly the same way.
"""

import cv2
import numpy as np
import os

# These two numbers define the "shape" of one example the model sees.
# Every video, no matter how long, gets turned into exactly SEQUENCE_LENGTH
# frames, and every frame is resized to IMG_SIZE x IMG_SIZE pixels.
SEQUENCE_LENGTH = 20
IMG_SIZE = 64

# Canonical activities required by the project specification.
TARGET_ACTIVITIES = [
    "Walking",
    "Running",
    "Sitting",
    "Standing",
    "Jumping",
    "Waving",
    "Clapping",
    "Falling",
]

# Optional aliases when dataset folder names differ from canonical names.
# Example: if your dataset has "WalkingWithDog", it can still map to "Walking".
CLASS_ALIASES = {
    "Walking": ["Walking", "walking", "WalkingWithDog", "walk"],
    "Running": ["Running", "running", "Run", "run", "Jogging", "jogging"],
    "Sitting": ["Sitting", "sitting", "Sit", "sit"],
    "Standing": ["Standing", "standing", "Stand", "stand"],
    "Jumping": ["Jumping", "jumping", "Jump", "jump", "JumpingJack", "TrampolineJumping", "HighJump", "PoleVault", "JumpRope"],
    "Waving": ["Waving", "waving", "Wave", "wave"],
    "Clapping": ["Clapping", "clapping", "Clap", "clap"],
    "Falling": ["Falling", "falling", "Fall", "fall", "fall_floor"],
}


def extract_frames(video_path, sequence_length=SEQUENCE_LENGTH, img_size=IMG_SIZE):
    """
    Reads a video file and returns `sequence_length` evenly-spaced frames.

    Why evenly spaced? Videos have different lengths, but our CNN+LSTM model
    needs a fixed-size input every time. So instead of using every single
    frame, we pick a fixed number of frames spread across the whole video.

    Returns:
        np.array of shape (sequence_length, img_size, img_size, 3), scaled to [0, 1]
        or None if the video could not be opened / read at all.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return None

    # Choose `sequence_length` frame positions, evenly spread from start to end
    frame_indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = cap.read()
        if not success:
            # Corrupt/short video: pad with a black frame rather than crashing
            frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame)

    cap.release()

    frames = np.array(frames, dtype=np.float32) / 255.0  # normalize pixel values
    return frames


def extract_frames_from_image_sequence(sequence_dir, sequence_length=SEQUENCE_LENGTH, img_size=IMG_SIZE):
    """Reads a folder of image frames and returns `sequence_length` evenly-spaced frames."""
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_files = sorted(
        f for f in os.listdir(sequence_dir)
        if f.lower().endswith(image_exts)
    )

    if not image_files:
        return None

    frame_indices = np.linspace(0, len(image_files) - 1, sequence_length, dtype=int)

    frames = []
    for idx in frame_indices:
        image_path = os.path.join(sequence_dir, image_files[int(idx)])
        frame = cv2.imread(image_path)
        if frame is None:
            frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame)

    frames = np.array(frames, dtype=np.float32) / 255.0
    return frames
