"""
train.py
Trains the CNN+LSTM model on your dataset folder.

Loads the WHOLE dataset into memory before training (simple, easy-to-follow
style), then reports the metrics this project needs: Accuracy, Precision,
Recall, F1 Score -- plus a few plots so you can sanity-check the model.

Usage:
    python train.py --data_dir dataset --epochs 10
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from data_prep import load_dataset
from model import build_model
from utils import TARGET_ACTIVITIES


def plot_sample_frames(X, y, class_names):
    """One middle frame per class -- a quick visual check that the data
    loaded correctly before you spend time training on it."""
    num_classes = len(class_names)
    cols = min(6, num_classes)
    rows = int(np.ceil(num_classes / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    labels = np.argmax(y, axis=1)
    for i, class_name in enumerate(class_names):
        idx = np.where(labels == i)[0][0]
        mid_frame = (X[idx][len(X[idx]) // 2] * 255).astype(np.uint8)
        axes[i].imshow(mid_frame)
        axes[i].set_title(class_name, fontsize=10)
        axes[i].axis("off")
    for j in range(len(class_names), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Sample Frame from Each Activity Class")
    plt.tight_layout()
    plt.savefig("sample_frames.png")
    print("Saved sample_frames.png")


def plot_history(history):
    """Training vs validation accuracy and loss, side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("training_history.png")
    print("Saved training_history.png")


def show_sample_predictions(model, X_test, y_test, class_names, num_samples=5):
    """Prints predictions on a few random test videos next to the ground
    truth -- an easy way to eyeball whether the model learned anything
    sensible."""
    n = min(num_samples, len(X_test))
    indices = np.random.choice(len(X_test), n, replace=False)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        video = X_test[idx]
        true_label = class_names[np.argmax(y_test[idx])]

        prediction = model.predict(np.expand_dims(video, axis=0), verbose=0)[0]
        predicted_label = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        mid_frame = (video[len(video) // 2] * 255).astype(np.uint8)
        ax.imshow(mid_frame)
        ax.set_title(f"True: {true_label}\nPred: {predicted_label} ({confidence*100:.0f}%)", fontsize=10)
        ax.axis("off")

    plt.suptitle("Sample Predictions on Test Videos")
    plt.tight_layout()
    plt.savefig("sample_predictions.png")
    print("Saved sample_predictions.png")


def main(data_dir, epochs, batch_size, max_per_class):
    # 1. Load every video into memory as NumPy arrays
    X, y, class_names = load_dataset(data_dir, max_samples_per_class=max_per_class)
    num_classes = len(class_names)
    print("Requested activities:", TARGET_ACTIVITIES)
    print("Training on activities:", class_names)
    print("Max samples per class:", max_per_class)

    # 2. Quick visual sanity check before training
    plot_sample_frames(X, y, class_names)

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
    )
    print("Training videos:", X_train.shape[0])
    print("Testing videos:", X_test.shape[0])

    # 4. Build and train the model
    model = build_model(num_classes)
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size
    )

    plot_history(history)
    show_sample_predictions(model, X_test, y_test, class_names)

    # 5. Evaluate with the metrics this project asks for
    y_pred_probs = model.predict(X_test, verbose=0)
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n=== Overall Metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\n=== Classification Report (Accuracy / Precision / Recall / F1) ===")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Saved confusion_matrix.png")

    # 6. Save the model + class names -- app.py needs both of these
    model.save("har_model.h5")
    with open("class_names.txt", "w") as f:
        f.write("\n".join(class_names))
    print("Saved har_model.h5 and class_names.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="UCF50")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_per_class", type=int, default=50)
    args = parser.parse_args()
    main(args.data_dir, args.epochs, args.batch_size, args.max_per_class)
