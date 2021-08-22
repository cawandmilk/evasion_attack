import tensorflow as tf
import tensorflow_addons as tfa


class Preprocessing(tf.keras.Model):

    def __init__(
        self,
        frame_length: int,
        frame_step: int, 
        fft_length: int, 
        pad_end: bool,
        num_mel_bins: int,
        # num_spectrogram_bins: int,
        sample_rate: int,
        lower_edge_hertz: float,
        upper_edge_hertz: float,
        **kwargs,
    ):
        """ Initialize.
        """
        super(Preprocessing, self).__init__(self, **kwargs)
        ## STFT parameters.
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.window_fn = tf.signal.hamming_window
        self.pad_end = pad_end

        ## Mel-spectrogram parameters.
        self.num_mel_bins = num_mel_bins
        self.num_spectrogram_bins = self.fft_length // 2 + 1
        self.sample_rate = sample_rate
        self.lower_edge_hertz = lower_edge_hertz
        self.upper_edge_hertz = upper_edge_hertz

        ## Instance normalization layer.
        self.norm_layer = tfa.layers.InstanceNormalization()


    def _mel_spectrogram(self, x: tf.Tensor):
        """ Mel-Spectrogarm implementation.

            It was referenced from:
             - https://www.tensorflow.org/api_docs/python/tf/signal/mfccs_from_log_mel_spectrograms
        """
        stfts = tf.signal.stft(
            x,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.fft_length,
            window_fn=self.window_fn,
            pad_end=self.pad_end,
        )

        spectrograms = tf.abs(stfts)

        mel_spectrograms = tf.tensordot(
            spectrograms,
            tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins=self.num_mel_bins,
                num_spectrogram_bins=self.num_spectrogram_bins,
                sample_rate=self.sample_rate,
                lower_edge_hertz=self.lower_edge_hertz,
                upper_edge_hertz=self.upper_edge_hertz,
            ),
            1,
        )

        return mel_spectrograms


    def _instance_normalization(self, x: tf.Tensor):
        """ Call a instance normalization layer.
        """
        return self.norm_layer(x)


    def call(self, x: tf.Tensor):
        """ Callable body.
        """
        x = self._mel_spectrogram(x)
        x = self._instance_normalization(x)
        return x
