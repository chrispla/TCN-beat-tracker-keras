"""Training script for the beat/downbeat tracking model, using Keras.

example:
`python train.py -t beats -l`
where `-t` specifies the type of model to train (either "beats" or "downbeats")
and `-l` logs the training process to Weights & Biases.
"""

import argparse

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

from dataset import Ballroom
from model import BeatTrackingTCN

parser = argparse.ArgumentParser(description="Train a beat/downbeat tracking model.")
parser.add_argument(
    "--log",
    "-l",
    "--log",
    action="store_true",
    help="Log the training process to Weights & Biases.",
)
parser.add_argument(
    "--type",
    "-t",
    default="beats",
    choices=["beats", "downbeats"],
)
parser.add_argument(
    "--eager",
    "-e",
    action="store_true",
    help="Run the model eagerly.",
)
args = parser.parse_args()
logging = args.log
label_type = args.type
eager = True if args.eager else False

# Define the model
model = BeatTrackingTCN()

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=BinaryCrossentropy(),
    metrics=["accuracy"],
    run_eagerly=eager,
)

# Define callbacks
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=10, min_lr=1e-6)
early_stopping = EarlyStopping(monitor="val_loss", patience=50)
model_checkpoint = ModelCheckpoint(
    "models/beat_tracker_{val_accuracy:.4f}",
    monitor="val_accuracy",
    mode="max",
    save_best_only=False,
    save_format="tf",
)
callbacks = [reduce_lr, early_stopping, model_checkpoint]

if logging:
    from wandb.keras import WandbCallback

    import wandb

    run = wandb.init(project="beat-tracking")
    wandb_callback = WandbCallback()
    callbacks.append(wandb_callback)

# Load the datasets
ballroom = Ballroom(
    audio_dir="data/ballroom/audio",
    annotation_dir="data/ballroom/annotations",
    spectrogram_dir="data/ballroom/spectrograms",
    label_type=label_type,
)
train_dataset, valid_dataset, test_dataset = ballroom.get_datasets()

# Train the model
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=100,
    callbacks=callbacks,
    batch_size=32,
    shuffle=True,
)
