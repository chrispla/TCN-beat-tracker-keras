from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

from dataset import Ballroom
from model import BeatTrackingTCN

# Define the model
model = BeatTrackingTCN()

# Compile the model
model.compile(optimizer=Adam(lr=0.01), loss=BinaryCrossentropy(), metrics=["accuracy"])

# Define callbacks
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=20, min_lr=1e-6)
early_stopping = EarlyStopping(monitor="val_loss", patience=500)

# Load the datasets
ballroom = Ballroom(
    audio_dir="data/ballroom/audio",
    annotation_dir="data/ballroom/annotations",
    label_type="beats",
)
train_dataset, valid_dataset, test_dataset = ballroom.get_datasets()

# Train the model
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=500,
    callbacks=[reduce_lr, early_stopping],
    batch_size=1,
)
