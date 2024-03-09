"""Compute mel-spectrograms from audio files for faster training."""

from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

with open("data/ballroom/audio/allBallroomFiles", "r") as f:
    audio_files = f.readlines()
    audio_files = [x.strip() for x in audio_files]

spectrogram_dir = Path("data/ballroom/spectrograms")
spectrogram_dir.mkdir(exist_ok=True)

for audio_file in tqdm(audio_files, total=len(audio_files)):
    y, sr = librosa.load(f"data/ballroom/audio/{audio_file}", sr=44100)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=81, hop_length=441, n_fft=2048
    )
    np.save(spectrogram_dir / f"{Path(audio_file).stem}.npy", mel_spectrogram)
