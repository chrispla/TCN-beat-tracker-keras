import argparse
import os

import numpy as np
from tensorflow.keras.models import load_model

from dataset import Ballroom

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import matplotlib.pyplot as plt

from metrics import f1_score

parser = argparse.ArgumentParser(description="Evaluate model")
parser.add_argument("--name", "-n", type=str, required=False, help="model name")
parser.add_argument(
    "--type", "-t", type=str, default="beats", help="model type (beats or downbeats)"
)
args = parser.parse_args()
label_type = args.type

from pathlib import Path

if not args.name:
    # get the model with the highest validation accuracy (split("_")[-1][-3])
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
    models.sort(key=lambda x: float(x.stem.split("_")[-1][:-3]))
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
    label_type=label_type,
)
train_dataset, valid_dataset, test_dataset = ballroom.get_datasets()

# Use the model for prediction
predictions = model.predict(test_dataset)

f1_scores = []
for truth, pred in zip(test_dataset, predictions):
    f1_scores.append(f1_score(truth[1][0], np.squeeze(pred)))

print("Mean F1 score:", sum(f1_scores) / len(f1_scores))
for truth, pred in zip(test_dataset, predictions):
    f1_scores.append(f1_score(np.squeeze(truth[1]), np.squeeze(pred)))

print("Mean F1 score:", sum(f1_scores) / len(f1_scores))
