import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import (
    ELU,
    Conv1D,
    Conv2D,
    Dropout,
    MaxPooling2D,
    ZeroPadding2D,
)


def compute_mel_spectrogram(waveform, sample_rate, n_fft, hop, mel):
    # Compute STFT
    stft = tf.signal.stft(waveform, frame_length=n_fft, frame_step=hop)

    # Compute power spectrogram
    spectrogram = tf.abs(stft) ** 2

    # Build Mel filterbank
    mel_weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=mel,
        num_spectrogram_bins=spectrogram.shape[-1],
        sample_rate=sample_rate,
        lower_edge_hertz=0.0,
        upper_edge_hertz=sample_rate / 2.0,
    )

    # Apply Mel filterbank
    mel_spectrogram = tf.tensordot(spectrogram, mel_weights, 1)

    return mel_spectrogram


class TCNLayer(Model):
    def __init__(
        self, inputs, outputs, dilation, kernel_size=5, padding=4, stride=1, dropout=0.1
    ):
        super(TCNLayer, self).__init__()

        self.conv1 = Conv1D(
            outputs,
            kernel_size,
            strides=stride,
            padding="same",
            dilation_rate=dilation,
            kernel_initializer=HeNormal(),
        )
        self.elu1 = ELU()
        self.dropout1 = Dropout(dropout)

        self.conv2 = Conv1D(
            outputs,
            kernel_size,
            strides=stride,
            padding="same",
            dilation_rate=dilation,
            kernel_initializer=HeNormal(),
        )
        self.elu2 = ELU()
        self.dropout2 = Dropout(dropout)

    def call(self, x):
        y = self.conv1(x)
        y = self.elu1(y)
        y = self.dropout1(y)
        y = self.conv2(y)
        y = self.elu2(y)
        y = self.dropout2(y)

        return y


class fullTCN(Model):
    def __init__(self, inputs, channels, kernel_size=5, dropout=0.1):
        super(fullTCN, self).__init__()

        self.model_layers = []
        n_levels = len(channels)

        for i in range(n_levels):
            dilation = 2**i

            n_channels_in = channels[i - 1] if i > 0 else inputs
            n_channels_out = channels[i]

            self.model_layers.append(
                TCNLayer(
                    n_channels_in,
                    n_channels_out,
                    dilation,
                    kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout,
                )
            )

    def call(self, x):
        for layer in self.model_layers:
            x = layer(x)
        return x


class BeatTrackingTCN(Model):
    def __init__(self, channels=16, kernel_size=5, dropout=0.1):
        super(BeatTrackingTCN, self).__init__()

        self.convblock1 = [
            ZeroPadding2D(padding=((1, 1), (0, 0))),
            Conv2D(
                channels,
                (3, 3),
                padding="valid",
                kernel_initializer=HeNormal(),
            ),
            ELU(),
            Dropout(dropout),
            MaxPooling2D((1, 3)),
        ]

        self.convblock2 = [
            ZeroPadding2D(padding=((1, 1), (0, 0))),
            Conv2D(
                channels,
                (3, 3),
                padding="valid",
                kernel_initializer=HeNormal(),
            ),
            ELU(),
            Dropout(dropout),
            MaxPooling2D((1, 3)),
        ]

        self.convblock3 = [
            Conv2D(channels, (1, 8), kernel_initializer=HeNormal()),
            ELU(),
            Dropout(dropout),
        ]

        self.tcn = fullTCN(channels, [channels] * 11, kernel_size, dropout)

        self.out = Conv1D(1, 1)

    def call(self, spec):

        print("Mel shape:", spec.shape)
        for layer in self.convblock1:
            spec = layer(spec)
        print("Conv1:", spec.shape)
        for layer in self.convblock2:
            spec = layer(spec)
        print("Conv2:", spec.shape)
        for layer in self.convblock3:
            spec = layer(spec)
        print("Conv3:", spec.shape)

        pre_tcn = tf.squeeze(spec, axis=-2)
        tcn_out = self.tcn(pre_tcn)

        logits = self.out(tcn_out)

        return tf.sigmoid(logits)