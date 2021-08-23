import tensorflow as tf

import numpy as np

from tqdm import tqdm


class InferenceIdentificationModel():
    
    @staticmethod
    def inference_random_sliced_dataset(latest_model: tf.keras.Model, ts_ds: tf.data.Dataset, total: int = None):
        """ Inference random sliced dataset in identification task.

            Either you can just use 'latest_model.predict(ts_ds, verbose=1)'.
        """
        y_true = []
        y_pred = []

        desc = f"Inference {latest_model.name}"
        for element in tqdm(ts_ds, total=total, desc=desc):
            ## Unpack.
            inp, tar = element

            ## Inference.
            _, scaled_similarity = latest_model(inp, training=False)

            y_true.append(tar.numpy())
            y_pred.append(tf.nn.softmax(scaled_similarity).numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        return y_true, y_pred


    @staticmethod
    def inference_fixed_sliced_dataset(latest_model: tf.keras.Model, ts_ds: tf.data.Dataset, total: int = None):
        """ Inference fixed sliced dataset in identification task.
        """
        y_true = []
        y_pred = []

        desc = f"Inference {latest_model.name}"
        for element in tqdm(ts_ds, total=total, desc=desc):
            ## Unpack.
            inp, tar = element

            ## Inference.
            _, scaled_similarity = latest_model(inp, training=False)

            y_true.append(tar[0].numpy())
            ## Apply mean first, softmax last.
            ## If you apply softmax first, then your score will be '0.8809'.
            y_pred.append(tf.nn.softmax(tf.math.reduce_mean(scaled_similarity, axis = 0)).numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        return y_true, y_pred


class InferenceVerificationModel():
    
    @staticmethod
    def inference_random_sliced_dataset(latest_model: tf.keras.Model, ts_ds: tf.data.Dataset, total: int = None):
        """ Inference random sliced dataset in verification task.
        """
        y_true = []
        y_pred = []

        desc = f"Inference {latest_model.name}"
        for element in tqdm(ts_ds, total=total, desc=desc):
            ## Unpack.
            inp_1, inp_2, tar = element

            ## Inference.
            ## Combine 'inp_1', 'inp_2' together to maker predictions
            ## at once. Inference uses less GPU memroy than training.
            inp = tf.concat([inp_1, inp_2], axis=0)
            embeddings = latest_model.embedding_model.predict(inp)

            ## Calculate cosine similarity.
            similarity = tf.einsum("ik,ik->i", *tf.split(embeddings, num_or_size_splits=2, axis=0))

            y_true.append(tar)
            y_pred.append(similarity)

        y_true = tf.stack(y_true, axis=0).numpy()
        y_pred = tf.stack(y_pred, axis=0).numpy()

        ## [0, 1] -> [-1, 1]
        y_true = np.where(y_true == 0, -1, y_true)

        return y_true, y_pred


    @staticmethod
    def inference_fixed_sliced_dataset(latest_model: tf.keras.Model, ts_ds: tf.data.Dataset, total: int = None):
        """ Inference fixed sliced dataset in verification task.
        """
        y_true = []
        y_pred = []

        desc = f"Inference {latest_model.name}"
        for element in tqdm(ts_ds, total=total, desc=desc):
            ## Unpack.
            inp_1, inp_2, tar = element

            ## Inference.
            ## Combine 'inp_1', 'inp_2' together to maker predictions
            ## at once. Inference uses less GPU memroy than training.
            inp = tf.concat([inp_1, inp_2], axis=0)
            embeddings = latest_model.embedding_model.predict(inp)

            ## Calculate cosine similarity.
            cross_similarity = tf.einsum("ik,jk->ij", *tf.split(embeddings, num_or_size_splits=2, axis=0))
            similarity = tf.math.reduce_mean(cross_similarity)

            y_true.append(tar)
            y_pred.append(similarity)

        y_true = tf.stack(y_true, axis=0).numpy()
        y_pred = tf.stack(y_pred, axis=0).numpy()

        ## [0, 1] -> [-1, 1]
        y_true = np.where(y_true == 0, -1, y_true)

        return y_true, y_pred
