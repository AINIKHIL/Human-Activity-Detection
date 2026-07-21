"""
data_prep.py
Loads an entire video dataset into memory as NumPy arrays -- the simplest
way to get from "folder of videos" to arrays Keras can train on directly.

Expected folder layout (one sub-folder per activity class):

    dataset/
        Walking/
            video1.avi
            video2.avi
        Running/
            ...

This works well for a modest number of classes/videos (tens to a few
hundred videos per class), since everything is held in RAM at once. If you
later outgrow available memory, you'd want to switch to loading videos in
batches instead of all at once.
"""

import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from utils import (
    extract_frames,
    extract_frames_from_image_sequence,
    SEQUENCE_LENGTH,
    IMG_SIZE,
    TARGET_ACTIVITIES,
    CLASS_ALIASES,
)


def _resolve_class_folder(dataset_dir, activity_name, aliases):
    """Returns the first matching folder path for a canonical activity name."""
    candidate_names = aliases.get(activity_name, [activity_name])
    existing_candidates = []
    for candidate in candidate_names:
        candidate_path = os.path.join(dataset_dir, candidate)
        if os.path.isdir(candidate_path):
            existing_candidates.append((candidate_path, candidate))

    for candidate_path, candidate in existing_candidates:
        if _collect_class_samples(candidate_path):
            return candidate_path, candidate

    if existing_candidates:
        return existing_candidates[0]
    return None, None


def _collect_class_samples(class_dir):
    """Collects sample entries from a class folder.

    Returns a list of tuples:
        ("video", file_path) for video files
        ("sequence", dir_path) for frame-sequence directories
    """
    video_exts = (".avi", ".mp4", ".mov", ".mkv", ".webm", ".mpg", ".mpeg")
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    samples = []

    for root, _, files in os.walk(class_dir):
        has_images = any(file_name.lower().endswith(image_exts) for file_name in files)
        if has_images:
            samples.append(("sequence", root))

        for file_name in files:
            if file_name.lower().endswith(video_exts):
                samples.append(("video", os.path.join(root, file_name)))

    return sorted(set(samples), key=lambda x: x[1])


def load_dataset(
    dataset_dir,
    sequence_length=SEQUENCE_LENGTH,
    img_size=IMG_SIZE,
    target_activities=TARGET_ACTIVITIES,
    class_aliases=CLASS_ALIASES,
    max_samples_per_class=None,
):
    """
    Walks dataset_dir, extracts frames from every video, and returns:
        X: np.array of shape (num_videos, sequence_length, img_size, img_size, 3)
        y: one-hot encoded labels, shape (num_videos, num_classes)
        class_names: list of class names, where class_names[i] is the name for label i
    """
    requested_class_names = list(target_activities)
    class_names = []

    X, y = [], []
    for class_name in requested_class_names:
        class_dir, matched_folder_name = _resolve_class_folder(dataset_dir, class_name, class_aliases)
        if class_dir is None:
            print(f"{class_name} -> folder not found (checked aliases: {class_aliases.get(class_name, [class_name])})")
            continue

        samples = _collect_class_samples(class_dir)
        if max_samples_per_class is not None:
            samples = samples[:max_samples_per_class]

        loaded = 0
        for sample_type, sample_path in samples:
            if sample_type == "video":
                frames = extract_frames(sample_path, sequence_length, img_size)
            else:
                frames = extract_frames_from_image_sequence(sample_path, sequence_length, img_size)

            if frames is not None and frames.shape == (sequence_length, img_size, img_size, 3):
                X.append(frames)
                loaded += 1

        if loaded > 0:
            label = len(class_names)
            class_names.append(class_name)
            y.extend([label] * loaded)

        print(f"{class_name} (from '{matched_folder_name}') -> loaded {loaded}/{len(samples)} samples")

    if not X:
        raise ValueError(
            "No videos were loaded for the configured activities. "
            "Check dataset folder names or update CLASS_ALIASES in utils.py."
        )

    missing_classes = sorted(set(requested_class_names) - set(class_names))
    if missing_classes:
        print(f"Skipped activities with no loaded samples: {missing_classes}")

    X = np.array(X, dtype=np.float32)
    y = to_categorical(y, num_classes=len(class_names))

    print("\nData loaded:")
    print("X shape:", X.shape)  # (samples, time_steps, height, width, channels)
    print("y shape:", y.shape)  # (samples, num_classes)

    return X, y, class_names
