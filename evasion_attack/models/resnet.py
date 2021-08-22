import tensorflow as tf

from evasion_attack.models.pooling import AttentiveStatisticsPooling


def conv_bn_relu(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    strides: int = 1,
    padding: str = "same",
    use_activation: bool = True,
    activation_fn: callable = tf.nn.relu6,
):
    """ Conv-BN-ReLU6.
    """
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if use_activation:
        x = tf.keras.layers.Activation(activation_fn)(x)

    return x


def residual_block(
    x: tf.Tensor,
    filters: int, ## output_filters
    kernel_size: int = 3,
    strides: int = 1,
    activation_fn: callable = tf.nn.relu6,
):
    """ Residual block using conv-bn-relu function.
    """
    residual = x

    x = conv_bn_relu(x, filters, kernel_size, strides=strides)
    x = conv_bn_relu(x, filters, kernel_size, use_activation=False)

    if strides == 2:
        residual = conv_bn_relu(residual, filters, 1, strides=strides, use_activation=False)

    x = tf.keras.layers.Add()([x, residual])
    x = tf.keras.layers.Activation(activation_fn)(x)

    return x


def embedding_model(
    input_shape: tuple,
    num_classes: int,
    embedding_dim: int,
    preprocessing_fn: callable,
    model_name: str = None,
):
    """ Embedding model.
    """
    x = model_input = tf.keras.layers.Input(shape=input_shape)  # (slice_length,)

    ## Preprocessing.
    x = preprocessing_fn(x)

    ## Entry flow (stem).
    x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(x)
    x = conv_bn_relu(x, 32, 3, strides=1)

    ## Middle flow.
    args = [
        ## (output_filters, strides, repeats)
        (32,  1, 3),
        (64,  2, 4),
        (128, 2, 6),
        (256, 2, 3),
    ]

    for (output_filters, strides, repeats) in args:
        x = residual_block(x, output_filters, strides=strides)
        for _ in range(1, repeats):
            ## Only the first repeat may has strides == 2.
            x = residual_block(x, output_filters)

    ## Exit flow.
    x = tf.keras.layers.Reshape((x.shape[1], -1))(x)
    x = tf.keras.layers.Permute((2, 1))(x)
    x = AttentiveStatisticsPooling()(x)

    x = tf.keras.layers.Dense(embedding_dim)(x)

    ## Throw l2 normalized vectors.
    model_output = x = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(x)

    return tf.keras.Model(model_input, model_output, name=model_name)
