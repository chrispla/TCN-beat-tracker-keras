from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf


class Ballroom:
    def __init__(
        self, audio_dir, annotation_dir, sr=44100, hop=441, label_type="beats"
    ):
        self.audio_dir = audio_dir
        self.annotation_dir = annotation_dir
        self.sr = sr
        self.hop = hop
        self.min_duration = 29.35
        self.min_duration_samples = int(self.min_duration * self.sr)
        self.label_type = label_type

        with open(Path(self.audio_dir) / "allBallroomFiles", "r") as f:
            self.audio_files = f.readlines()

        self.basenames = [Path(f.strip()).stem for f in self.audio_files]
        self.audio_files = {
            Path(f.strip()).stem: str(Path(self.audio_dir) / f.strip())
            for f in self.audio_files
        }

        self.beat_times = {}
        self.downbeat_times = {}
        for basename in self.basenames:
            annotation_path = Path(annotation_dir) / (basename + ".beats")
            beats = []
            downbeats = []
            with open(annotation_path, "r") as f:
                for line in f:
                    beat_time, beat_index = line.strip().split()
                    beats.append(float(beat_time))
                    if beat_index == "1":
                        downbeats.append(float(beat_time))
            self.beat_times[basename] = np.array(beats)
            self.downbeat_times[basename] = np.array(downbeats)

        # get beat and downbeat vectors
        self.beat_vectors = {}
        self.downbeat_vectors = {}
        for basename in self.basenames:
            beat_times = self.beat_times[basename]
            downbeat_times = self.downbeat_times[basename]

            beat_vector = np.zeros(int(self.min_duration_samples // self.hop))
            downbeat_vector = np.zeros(int(self.min_duration_samples // self.hop))

            beat_indices = (beat_times * self.sr // self.hop).astype(int)
            downbeat_indices = (downbeat_times * self.sr // self.hop).astype(int)

            for i in beat_indices:
                try:
                    beat_vector[i] = 1
                    # surround with 0.5s
                    if i < len(beat_vector) - 1:
                        beat_vector[i + 1] = 0.5
                    if i < len(beat_vector) - 2:
                        beat_vector[i + 2] = 0.5
                    if i > 1:
                        beat_vector[i - 1] = 0.5
                    if i > 2:
                        beat_vector[i - 2] = 0.5
                except IndexError:
                    pass
            for i in downbeat_indices:
                try:
                    downbeat_vector[i] = 1
                    if i < len(downbeat_vector) - 1:
                        downbeat_vector[i + 1] = 0.5
                    if i < len(downbeat_vector) - 2:
                        downbeat_vector[i + 2] = 0.5
                    if i > 1:
                        downbeat_vector[i - 1] = 0.5
                    if i > 2:
                        downbeat_vector[i - 2] = 0.5
                except IndexError:
                    pass

            self.beat_vectors[basename] = beat_vector
            self.downbeat_vectors[basename] = downbeat_vector

    def load_mel(self, basename):
        audio_path = self.audio_files[basename]
        audio = librosa.load(audio_path, sr=self.sr)[0]
        audio = audio[: self.min_duration_samples]
        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, hop_length=self.hop, n_mels=81, n_fft=2048
        )
        mel = mel[:, : int(self.min_duration_samples / self.hop)].T

        mel = tf.convert_to_tensor(mel)
        mel = tf.expand_dims(mel, axis=-1)  # Add channel dimension
        # add None dim for batch
        mel = tf.expand_dims(mel, axis=0)

        return mel

    def get_splits(self, seed=42, split=[0.8, 0.1, 0.1]):
        if seed is not None:
            np.random.seed(seed)

        basenames = np.array(self.basenames)
        np.random.shuffle(basenames)
        train = basenames[: int(split[0] * len(basenames))]
        valid = basenames[
            int(split[0] * len(basenames)) : int((split[0] + split[1]) * len(basenames))
        ]
        test = basenames[int((split[0] + split[1]) * len(basenames)) :]
        test = basenames[int((split[0] + split[1]) * len(basenames)) :]

        return train, valid, test

    def to_dataset(self, basenames):
        def gen():
            for basename in basenames:
                if self.label_type == "beats":
                    yield self.load_mel(basename), tf.expand_dims(
                        tf.convert_to_tensor(self.beat_vectors[basename]), axis=0
                    )
                elif self.label_type == "downbeats":
                    yield self.load_mel(basename), tf.expand_dims(
                        tf.convert_to_tensor(self.downbeat_vectors[basename]), axis=0
                    )
                else:
                    raise ValueError("Invalid label type")

        return tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(
                    shape=(None, int(self.min_duration_samples / self.hop), 81, 1),
                    dtype=tf.float32,
                ),
                tf.TensorSpec(
                    shape=(None, int(self.min_duration_samples / self.hop)),
                    dtype=tf.float32,
                ),
            ),
        )

    def get_datasets(self, seed=42, split=[0.8, 0.1, 0.1]):
        train_basenames, valid_basenames, test_basenames = self.get_splits(seed, split)
        train_dataset = self.to_dataset(train_basenames)
        valid_dataset = self.to_dataset(valid_basenames)
        test_dataset = self.to_dataset(test_basenames)
        return train_dataset, valid_dataset, test_dataset
