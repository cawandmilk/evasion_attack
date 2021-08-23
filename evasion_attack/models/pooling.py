import tensorflow as tf


class AttentiveStatisticsPooling(tf.keras.layers.Layer):

    def __init__(
        self,
        reduction_dim: int = 128,
        epsilon: float = 1e-5,
        activation_fn: callable = tf.nn.relu6,
        **kwargs,
    ):
        """ Initial variables of the class.

            It was referenced from:
            - https://github.com/clovaai/voxceleb_trainer/blob/master/models/ResNetSE34V2.py
        """
        super(AttentiveStatisticsPooling, self).__init__(**kwargs)
        self.reduction_dim = reduction_dim
        self.epsilon = epsilon
        self.activation_fn = activation_fn


    def build(self, input_shape: tuple):
        """ Build the flexible layers.

            Before creating a layer, the output demension of the model
            must be determined, so overwrite the 'build' method to create
            it flexibly.
        """
        self.attention = tf.keras.Sequential([
            tf.keras.layers.Conv1D(self.reduction_dim, 1),
            tf.keras.layers.Activation(self.activation_fn),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(input_shape[-1], 1),
            tf.keras.layers.Activation(tf.nn.softmax),
        ])


    def call(self, x: tf.Tensor, training: bool = None):
        """ Callable body.
             - input.shape:  (batch_size, time_step, frames)
             - output.shape: (batch_size, 2 * frames)
        """
        w = self.attention(x, training=training)

        mu = tf.math.reduce_sum(x * w, axis=-1)
        va = tf.math.reduce_sum(w * (x ** 2), axis=-1) - (mu ** 2)
        sg = tf.math.sqrt(tf.where(va < self.epsilon, self.epsilon, va))

        return tf.concat([mu, sg], axis=-1)
