"""Utilities for beat processing."""

import librosa
import numpy as np
from madmom.features import DBNBeatTrackingProcessor


def vector_to_times(beat_vector, sr, hop):
    """Convert beat vector to beat times."""
    frames = np.where(beat_vector == 1.0)[0]
    return librosa.frames_to_time(frames, sr=sr, hop_length=hop, n_fft=2048)


def output_to_beat_times(output, sr, hop, label_type):
    """Convert model output to beat times using DBN."""
    if label_type == "beats":
        min_bpm, max_bpm = 55, 215
    elif label_type == "downbeats":
        min_bpm, max_bpm = 10, 75

    postprocessor = DBNBeatTrackingProcessor(
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        fps=sr / hop,
        online=False,
        transition_lambda=100,
    )
    return postprocessor.process_offline(output)


def process_comparands(y_true, y_pred, sr, hop, label_type):
    true_times = vector_to_times(y_true, sr=44100, hop=441)
    pred_times = output_to_beat_times(y_pred, sr=44100, hop=441, label_type="beats")
    return true_times, pred_times
