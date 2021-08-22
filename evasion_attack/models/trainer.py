import tensorflow as tf

from tqdm import tqdm


class AngularPrototypicalModel(tf.keras.Model):

    def __init__(
        self,
        embedding_model: tf.keras.Model,
        centroids: tf.keras.metrics.Metric,
        name: str = None,
        **kwargs,
    ):
        super(AngularPrototypicalModel, self).__init__(name=name, **kwargs)
        self.embedding_model = embedding_model
        self.centroids = centroids
        ## Weights for scaling cosine similarity.
        ## Initial weights are referenced from the below:
        ##  - https://github.com/clovaai/voxceleb_trainer/blob/master/loss/angleproto.py
        self.w = self.add_weight(
            name="w",
            shape=(),
            initializer=tf.keras.initializers.Constant(10.),
        )
        self.b = self.add_weight(
            name="b",
            shape=(),
            initializer=tf.keras.initializers.Constant(-5.),
        )


    def adapt(self, ds: tf.data.Dataset):
        """ Determine the initial states of the weights in 'Centroids'.
        """
        desc = "Adapting"
        for element in tqdm(ds, desc=desc):
            inp, tar = element
            self.centroids.update_state(
                y_true=tar,
                y_pred=self.embedding_model(inp, training=False),
                adapting=True,
            )
        ## Make normzliaed centroids from embedding sum.
        self.centroids.divide_and_normalize()


    def compile(self, ds: tf.data.Dataset, metric_fn: tf.keras.metrics.Metric, **kwargs):
        """ Overwrite 'compile' API.
        """
        super(AngularPrototypicalModel, self).compile(**kwargs)
        self.metric_fn = metric_fn
        ## 'ds is not None' means, 'if it is not inference mode'.
        if ds is not None:
            self.adapt(ds)


    @tf.function
    def call(self, inp: tf.Tensor, training: bool = None):
        """ Callable body.
             - inp.shape:               (batch_size, slice_len_sec * sample_rate) = (batch_size, 32_000)
             - y_pred.shape:            (batch_size, embedding_dim) = (batch_size, 512)
             - scaled_similarity.shape: (batch_size, num_classes) = (batch_size, 1_251)
        """
        ## First, get embedded features.
        ##  - y_pred.shape: (batch_size, embedding_dim) = (batch_size, 512)
        y_pred = self.embedding_model(inp, training=training)

        ## And calculate cosine similarity.
        ##  - similarity.shape: (batch_size, num_classes) = (batch_size, 1_251)
        similarity = tf.linalg.matmul(y_pred, self.centroids.result())  # (batch, 1_251)

        ## Similar to temperature scaling, affin transform is performed.
        ## Noted that it is not an element-wise operation.
        scaled_similarity = self.w * similarity + self.b
        
        ## Throw scaled similarity and y_true also for update centroids.
        return y_pred, scaled_similarity


    @tf.function
    def train_step(self, x: tf.Tensor):
        """ Overwritten tf.keras.Model.train_step() function.

            Mixed precision policy ('AMP' in pytorch) was not applied.
        """
        ## Unpack.
        inp, tar = x

        with tf.GradientTape() as tape:
            y_pred, scaled_similarity = self(inp, training=True)
            loss_value = self.compiled_loss(tar, scaled_similarity)

        ## "self.trainable_variables" contains "self.embedding_model.trainable_variables"
        grads = tape.gradient(loss_value, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        ## Update metrics include 'centroids'.
        self.centroids.update_state(tar, y_pred)
        self.metric_fn.update_state(tar, scaled_similarity)

        results = {"loss": loss_value}
        results.update({self.metric_fn.name: self.metric_fn.result()})

        return results


    @tf.function
    def test_step(self, x: tf.Tensor):
        """ Overwritten tf.keras.Model.test_step() function.
        """
        ## Unpack.
        inp, tar = x

        y_pred, scaled_similarity = self(inp, training=False)
        loss_value = self.compiled_loss(tar, scaled_similarity)

        ## Update metrics exclude 'centroids'.
        self.metric_fn.update_state(tar, scaled_similarity)

        results = {"loss": loss_value}
        results.update({self.metric_fn.name: self.metric_fn.result()})

        return results
