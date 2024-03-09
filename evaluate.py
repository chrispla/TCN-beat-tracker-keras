import argparse
import os
from pathlib import Path

import mir_eval
import numpy as np
from tensorflow.keras.models import load_model

from dataset import Ballroom
from utils import process_comparands

os.environ["CUDA_VISIBLE_DEVICES"] = ""

parser = argparse.ArgumentParser(description="Evaluate model")
parser.add_argument("--name", "-n", type=str, required=False, help="model name")
parser.add_argument(
    "--type", "-t", type=str, default="beats", help="model type (beats or downbeats)"
)
args = parser.parse_args()
label_type = args.type


if not args.name:
    # get the model with the highest validation accuracy
    if label_type == "beats":
        models = [
            x
            for x in Path("models").iterdir()
            if x.is_dir() and "beat_tracker_" in x.name
        ]
    elif label_type == "downbeats":
        models = [
            x
            for x in Path("models").iterdir()
            if x.is_dir() and "downbeat_tracker_" in x.name
        ]
    else:
        raise ValueError("Invalid model type, needs to be 'beats' or 'downbeats'.")
    models.sort(key=lambda x: float(x.stem.split("_")[-1]))
    try:
        model_dir = models[-1]
    except IndexError:
        print("No models found in ./models/")
        exit()
else:
    model_dir = Path("models", args.name)
print("Evaluating model:", model_dir)

# Load the model from a checkpoint
model = load_model(model_dir)  # replace with your model path

# Load the datasets
ballroom = Ballroom(
    audio_dir="data/ballroom/audio",
    annotation_dir="data/ballroom/annotations",
    spectrogram_dir="data/ballroom/spectrograms",
    label_type=label_type,
)
train_dataset, valid_dataset, test_dataset = ballroom.get_datasets()

# Use the model for prediction
predictions = model.predict(test_dataset)

metrics = {"f1": [], "CMLc": [], "CMLt": [], "AMLc": [], "AMLt": [], "D": []}

for y_truth, y_pred in zip(test_dataset, predictions):

    # clean up
    truth = y_truth[1][0]
    pred = np.squeeze(y_pred)

    # process
    true_times, pred_times = process_comparands(
        truth, pred, sr=44100, hop=441, label_type=label_type
    )

    # metrics
    metrics["f1"].append(mir_eval.beat.f_measure(true_times, pred_times))
    CMLc, CMLt, AMLc, AMLt = mir_eval.beat.continuity(true_times, pred_times)
    metrics["CMLc"].append(CMLc)
    metrics["CMLt"].append(CMLt)
    metrics["AMLc"].append(AMLc)
    metrics["AMLt"].append(AMLt)
    metrics["D"].append(mir_eval.beat.information_gain(true_times, pred_times))

# average metrics
for key in metrics:
    metrics[key] = np.mean(metrics[key])
    print(f"{key: <4}: {metrics[key]:.3f}")
