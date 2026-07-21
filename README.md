# Human Activity Recognition (CNN + LSTM)

A video-based activity recognition system: upload a video (or use a webcam),
and a CNN+LSTM model predicts the activity being performed along with a
confidence score. Built with TensorFlow/Keras and deployed as a Streamlit app.

## Target activities

The training pipeline is configured to train only for these activities:

- Walking
- Running
- Sitting
- Standing
- Jumping
- Waving
- Clapping
- Falling

If your dataset uses different folder names, update aliases in `utils.py`
(`CLASS_ALIASES`) so those folders map to the canonical names above.

## Functional Requirements

The application supports:

- Upload Video
- Predict Activity
- Display Activity Name
- Display Confidence

## Inputs and Outputs

- Input: Video, Webcam
- Output: Predicted Activity, Confidence Score

## Suggested AI Model

- CNN + LSTM

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score

## Read this first: UCF101 and your activity list

The problem statement asks for **Walking, Running, Sitting, Standing, Jumping,
Waving, Clapping, Falling**, and suggests **UCF101** as the dataset.

Worth knowing before you download anything: UCF101's 101 classes are mostly
*specific* actions like `PlayingGuitar`, `HorseRiding`, `JumpRope`,
`HandstandPushups`, etc. It does **not** contain plain `Sitting`,
`Standing`, or `Falling` classes at all, and its closest matches for the
others are approximate (e.g. `WalkingWithDog` for Walking, `JumpingJack`
for Jumping).

Two reasonable options -- the code works with either, since all it needs is
"one folder per class, videos inside":

1. **Stick with UCF101**, picking a subset of classes that reasonably stand
   in for what you need (e.g. `WalkingWithDog`, `JumpingJack`, `JumpRope`,
   `BoxingPunchingBag`), and say so explicitly in your report/viva.
2. **Use HMDB51 instead** for the exact classes you need -- it has `clap`,
   `fall_floor`, `jump`, `run`, `sit`, `stand`, `walk`, `wave`, which line up
   almost exactly with the problem statement. No code changes needed, just
   point `--data_dir` at a folder with that layout.

Put whichever videos you use into this folder structure before training:

```text
dataset/
    Walking/
        video1.avi
        video2.avi
    Running/
        ...
    Falling/
        ...
```

## Project structure

```text
har_project/
├── utils.py        # extract_frames(): video -> fixed-size frame sequence (shared by training + app)
├── data_prep.py     # load_dataset(): loads every video into memory as NumPy arrays
├── model.py         # the CNN+LSTM architecture
├── train.py         # trains the model, plots EDA/history/predictions, prints Accuracy/Precision/Recall/F1
├── app.py           # Streamlit app: upload video or webcam -> predicted activity + confidence
├── requirements.txt
└── README.md
```

## How the model works (plain-English version)

1. A video is turned into 20 evenly-spaced frames, each resized to 64x64.
2. Every frame is passed through the **same small CNN** (`TimeDistributed`
   in `model.py`) to turn each frame into a compact feature vector -- the
   CNN's job: "what's in this frame".
3. The sequence of 20 feature vectors goes into an **LSTM**, which learns
   how those features change frame-to-frame -- the LSTM's job: "how is it
   changing over time". This is what tells apart, say, sitting-down from
   standing-up, which a single frame can't do reliably.
4. A final Dense+softmax layer converts that into a probability for each
   activity, and the highest one is your prediction + confidence score.

## Setup

For Streamlit deployment / inference only:

```bash
pip install -r requirements.txt
```

For local training or retraining:

```bash
pip install -r requirements-train.txt
```

Training a CNN+LSTM video model needs a GPU to be practical. If you don't
have a local GPU, **Google Colab** or **Kaggle** (both offer a free GPU) are
the easiest options: upload these files plus your `dataset/` folder and run
`train.py` there.

Note: `train.py` loads the **entire** dataset into memory before training
(simple and easy to follow). This is fine for a modest number of
classes/videos (tens to a few hundred videos per class). If you use many
more classes or videos and run out of RAM, say so and I can switch this to
a generator that loads videos in batches instead.

## Step 1: Train the model

```bash
python train.py --data_dir dataset --epochs 10 --batch_size 4
```

This will:

- Load every video, print how many loaded per class
- Save `sample_frames.png` -- one sample frame per class, so you can eyeball
  that the data loaded correctly
- Split 80% train / 20% test, train the CNN+LSTM model
- Save `training_history.png` (accuracy & loss curves) and
  `sample_predictions.png` (a few test videos with true vs. predicted labels)
- Print a classification report with **Accuracy, Precision, Recall, F1
  Score** per class, and save `confusion_matrix.png`
- Save `har_model.h5` and `class_names.txt` -- then convert/export to `har_model.onnx`
  for TensorFlow-free deployment in the Streamlit app

## Step 2: Run the app

```bash
streamlit run app.py
```

This gives you:

- **Upload Video** tab -> predicts the activity + shows confidence + a bar
  chart of all class probabilities (all required functional requirements)
- **Live Webcam** tab (bonus) -> rolling-window live prediction, an activity
  timeline table of recent predictions (bonus: Activity Timeline), and a
  red alert banner if the "Falling" class fires with high confidence
  (bonus: Fall Detection)

**Note on webcam + deployment**: the webcam tab uses OpenCV
(`cv2.VideoCapture(0)`), which reads the webcam of the machine *running*
the Streamlit server -- fine for local testing, but it won't reach a
*viewer's* browser webcam if you deploy to Streamlit Community Cloud. For
true browser-based webcam access in a deployed app, swap that section for
the `streamlit-webrtc` package instead -- the frame-buffering and
prediction logic stays the same, only how frames arrive changes.

## Multiple Person Detection (optional requirement)

Not implemented, since it changes the problem meaningfully (you'd need a
person detector like YOLO to first crop each person, then run the
CNN+LSTM per person). If you want it: run a pretrained person detector on
each frame, crop each detected person's bounding box, and feed each
person's cropped frame sequence through the existing model separately.
