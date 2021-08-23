import tensorflow as tf


class Centroids(tf.keras.metrics.Metric):

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        **kwargs,
    ):
        """ Initialize.
        """
        super(Centroids, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        ## Variables.
        ##  - centroids.shape: (embedding_dim, num_classes) = (512, 1_251)
        ##  - counter.shape:   (num_classes,) = (1_251,)
        self.centroids = self.add_weight(
            name="centroids",
            shape=(self.embedding_dim, self.num_classes),
            initializer="zeros",
        )
        self.counter = self.add_weight(
            name="counter",
            shape=(self.num_classes,),
            initializer="zeros",
        )


    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor, adapting: bool = False):
        """ Update untrainable weights.
             - y_true.shape: (batch_size,)
             - y_pred.shape: (batch_size, embedding_dim) = (batch_size, 512)
        """
        y_true = tf.cast(y_true, dtype=tf.int64)

        ## Get one-hot variables.
        ##  - one_hot_y_true.shape: (batch_size, num_classes) ~= (batch_size, 1_251)
        ##  - one_hot_y_pred.shape: (batch_size, embedding_dim, num_classes) ~= (batch_size, 512, 1_251)
        one_hot_y_true = tf.one_hot(y_true, depth=self.num_classes, dtype=y_pred.dtype)
        one_hot_y_pred = tf.einsum("ij,ik->ijk", y_pred, one_hot_y_true)

        ## Squeeze batch dimension.
        ##  - y_true_sum.shape: (num_classes,) ~= (1_251,)
        ##  - y_pred_sum.shape: (embedding_dim, num_classes) ~= (512, 1_251)
        y_true_sum = tf.math.reduce_sum(one_hot_y_true, axis=0)
        y_pred_sum = tf.math.reduce_sum(one_hot_y_pred, axis=0)

        ## 'Adapting" is the step of determining initial 'centroids' and 'counter'
        ## weight values from an untrained, random initialized model. It is called
        ## from 'model.compile()', and the 'self.divide_and_mormalize()' function 
        ## is called explicitly in the 'trainer' after all adapting is finished.
        if adapting:
            self.centroids.assign_add(y_pred_sum)
            self.counter.assign_add(y_true_sum)
        else:
            self.centroids.assign(self.centroids.value() * (self.counter.value() - y_true_sum) + y_pred_sum)
            self.divide_and_normalize()


    def divide_and_normalize(self):
        """ Custom function s.t. do normalize the centroids of each classes.
        """
        divided = tf.math.divide_no_nan(self.centroids.value(), self.counter.value())
        self.centroids.assign(tf.math.l2_normalize(divided, axis=0)) ## not axis = -1


    def result(self):
        """ Overwritten function s.t. get(=copy as tensor) weights of 'self.centroids'.
        """
        return self.centroids.value()


    def reset_state(self):
        """ Overwritten function.
        """
        pass
