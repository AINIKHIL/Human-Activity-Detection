"""
model.py
Defines the CNN + LSTM architecture for activity recognition.

How it works, step by step:
1. Each of the SEQUENCE_LENGTH frames is passed through the SAME small CNN.
   `TimeDistributed` applies one CNN to every frame in the sequence, sharing
   weights across time -- this is the CNN's job: "what's in this frame".
2. The sequence of per-frame CNN outputs is flattened and fed into an LSTM,
   which learns how those features change from one frame to the next --
   this is the LSTM's job: "how is it changing over time". This is what
   lets the model tell "sitting down" apart from "standing up", which a
   single frame can't do on its own.
3. A final Dense (softmax) layer turns the LSTM's output into a probability
   for each activity class.
"""

from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from utils import SEQUENCE_LENGTH, IMG_SIZE


def build_model(num_classes, sequence_length=SEQUENCE_LENGTH, img_size=IMG_SIZE):
    model = models.Sequential(name="CNN_LSTM_HAR")

    # ---- CNN feature extractor, applied to every frame ----
    model.add(layers.TimeDistributed(
        layers.Conv2D(32, (3, 3), activation="relu"),
        input_shape=(sequence_length, img_size, img_size, 3)
    ))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))

    model.add(layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation="relu")))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))

    model.add(layers.TimeDistributed(layers.Flatten()))
    # Shape at this point: (batch, sequence_length, flattened_features)
    # -> one feature vector per frame. This is what the LSTM reads.

    # ---- LSTM: learns how those per-frame features evolve over time ----
    model.add(layers.LSTM(64))
    model.add(layers.Dropout(0.5))

    # ---- Classification head ----
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.0005),
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    # Quick sanity check when running this file directly
    m = build_model(num_classes=8)
    m.summary()
