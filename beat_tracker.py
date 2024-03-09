"""Compute beats and downbeats from an audio file."""

import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model

from utils import output_to_beat_times


def beatTracker(inputFile):

    # load audio
    y, sr = librosa.load(inputFile, sr=44100, mono=False)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=81, n_fft=2048, hop_length=441
    )
    print(mel.shape)
    # trim to 2935
    mel = mel[:, :2935].T
    mel = tf.expand_dims(tf.convert_to_tensor(mel), axis=0)
    mel = tf.expand_dims(mel, axis=-1)

    # compute beats
    beat_model = load_model("models/beat_tracker_0.9029")
    downbeat_model = load_model("models/downbeat_tracker_0.9709")

    beat_activations = tf.squeeze(beat_model.predict(mel))
    print(">>>>>>>>>>>>>>>", beat_activations.shape)
    beat_times = output_to_beat_times(beat_activations, 44100, 441, "beats")
    downbeat_activations = tf.squeeze(downbeat_model.predict(mel))
    downbeat_times = output_to_beat_times(downbeat_activations, 44100, 441, "downbeats")

    return beat_times, downbeat_times


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute beats and downbeats")
    parser.add_argument("-i", type=str, help="path to the input audio file")
    args = parser.parse_args()

    beat_times, downbeat_times = beatTracker(args.i)
    print("Beat times:", beat_times)
    print("Downbeat times:", downbeat_times)
    print("Downbeat times:", downbeat_times)
    print("Downbeat times:", downbeat_times)
