import tensorflow as tf


class IdentificationDataLoader():

    @staticmethod
    def get_parse_audio_function():
        """ Get function s.t. parse audio examples.
        """
        @tf.function
        def _parse_audio_function(example_proto):
            ## Parse the input tf.train.Example proto using description.
            audio_feature_description = {
                "length": tf.io.FixedLenFeature([], tf.int64),
                "label": tf.io.FixedLenFeature([], tf.int64),
                "audio_raw": tf.io.FixedLenFeature([], tf.string),
            }
            audio_features = tf.io.parse_single_example(example_proto, audio_feature_description)

            audio = tf.audio.decode_wav(audio_features["audio_raw"]).audio[:, 0]
            length = audio_features["length"]
            label = audio_features["label"]

            return audio, length, label

        return _parse_audio_function


    @staticmethod
    def get_slice_function(random_slice: bool, slice_len: int, num_slice: int):
        """ Get function considering iden/veri and train/test phase.
        """
        ## Define pseudo-randon number generator for random slicing.
        prng = tf.random.Generator.from_non_deterministic_state()

        @tf.function
        def _random_slice_function(audio, length, label):
            sp = prng.uniform(
                shape=[],
                minval=0,
                maxval=length - slice_len,
                dtype=tf.int64
            )

            return audio[sp:sp + slice_len], label

        @tf.function
        def _fixed_slice_function(audio, length, label):
            sp = tf.range(
                start=0,
                limit=length - slice_len + 1,
                delta=tf.math.floor((length - slice_len) / (num_slice - 1)),
                dtype=tf.int64,
            )

            audio_list = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
            for _sp in sp:
                audio_list = audio_list.write(audio_list.size(), audio[_sp:_sp + slice_len])

            return audio_list.stack(), tf.tile(tf.expand_dims(label, axis=0), [num_slice])

        return _random_slice_function if random_slice else _fixed_slice_function


    @staticmethod
    def get_check_numerics_function():
        """ Check numeric errors if in dataset, such as nan, inf, -inf.
        """
        @tf.function
        def _check_numerics_function(audio, label):
            tf.debugging.check_numerics(audio, message="")
            return audio, label

        return _check_numerics_function


    @staticmethod
    def get_dataset(
        tfrecord_filenames: list,
        cache: bool,
        repeats: bool,
        random_slice: bool,
        slice_len: int,
        num_slice: int, ## only required when using fixed_dataset
        shuffle: bool,
        buffer_size: int,
        global_batch_size: int,
        auto=tf.data.AUTOTUNE,
    ):
        """ Make dataset for identification using tf.data.Dataset() API.
        """
        ds = tf.data.TFRecordDataset(tfrecord_filenames, num_parallel_reads=auto)
        ds = ds.map(IdentificationDataLoader.get_parse_audio_function(), num_parallel_calls=auto)

        if cache:
            ds = ds.cache()

        if repeats:
            ds = ds.repeat()

        ds = ds.map(IdentificationDataLoader.get_slice_function(
            random_slice=random_slice,
            slice_len=slice_len,
            num_slice=num_slice
        ), num_parallel_calls=auto)

        if shuffle:
            ds = ds.shuffle(buffer_size, reshuffle_each_iteration=False)

        if global_batch_size:  # if not None
            ds = ds.batch(global_batch_size, num_parallel_calls=auto)

        ds = ds.map(IdentificationDataLoader.get_check_numerics_function(), num_parallel_calls=auto)
        ds = ds.prefetch(auto)

        return ds


class VerificationDataLoader():

    @staticmethod
    def get_parse_audio_function():
        """ Get function s.t. parse audio examples.
        """
        @tf.function
        def _parse_audio_function(example_proto):
            ## Parse the input tf.train.Example proto using description.
            audio_feature_description = {
                "audio_raw_1": tf.io.FixedLenFeature([], tf.string),
                "audio_raw_2": tf.io.FixedLenFeature([], tf.string),
                "length_1": tf.io.FixedLenFeature([], tf.int64),
                "length_2": tf.io.FixedLenFeature([], tf.int64),
                "y_true": tf.io.FixedLenFeature([], tf.int64),
            }
            audio_features = tf.io.parse_single_example(example_proto, audio_feature_description)

            audio_1 = tf.audio.decode_wav(audio_features["audio_raw_1"]).audio[:, 0]
            audio_2 = tf.audio.decode_wav(audio_features["audio_raw_2"]).audio[:, 0]
            length_1 = audio_features["length_1"]
            length_2 = audio_features["length_2"]
            y_true = audio_features["y_true"]

            return audio_1, audio_2, length_1, length_2, y_true

        return _parse_audio_function


    @staticmethod
    def get_slice_function(random_slice: bool, slice_len: int, num_slice: int):
        """ Get function considering iden/veri and train/test phase.
        """
        ## Define pseudo-randon number generator for random slicing.
        prng = tf.random.Generator.from_non_deterministic_state()

        @tf.function
        def _random_slice_function(audio_1, audio_2, length_1, length_2, y_true):
            sp_1 = prng.uniform(
                shape=[],
                minval=0,
                maxval=length_1 - slice_len,
                dtype=tf.int64,
            )
            sp_2 = prng.uniform(
                shape=[],
                minval=0,
                maxval=length_2 - slice_len,
                dtype=tf.int64,
            )

            audio_1 = audio_1[sp_1:sp_1 + slice_len]
            audio_2 = audio_2[sp_2:sp_2 + slice_len]

            return audio_1, audio_2, y_true

        @tf.function
        def _fixed_slice_function(audio_1, audio_2, length_1, length_2, y_true):
            sp_1 = tf.range(
                start=0,
                limit=length_1 - slice_len + 1,
                delta=tf.math.floor((length_1 - slice_len) / (num_slice - 1)),
                dtype=tf.int64,
            )
            sp_2 = tf.range(
                start=0,
                limit=length_2 - slice_len + 1,
                delta=tf.math.floor((length_2 - slice_len) / (num_slice - 1)),
                dtype=tf.int64,
            )

            audio_list_1 = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
            audio_list_2 = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

            for _sp in sp_1:
                audio_list_1 = audio_list_1.write(audio_list_1.size(), audio_1[_sp:_sp + slice_len])
            for _sp in sp_2:
                audio_list_2 = audio_list_2.write(audio_list_2.size(), audio_2[_sp:_sp + slice_len])

            return audio_list_1.stack(), audio_list_2.stack(), y_true

        return _random_slice_function if random_slice else _fixed_slice_function


    @staticmethod
    def get_check_numerics_function():
        """ Check numeric errors if in dataset, such as nan, inf, -inf.
        """
        @tf.function
        def _check_numerics_function(audio_1, audio_2, label):
            tf.debugging.check_numerics(audio_1, message="")
            tf.debugging.check_numerics(audio_2, message="")
            return audio_1, audio_2, label

        return _check_numerics_function


    @staticmethod
    def get_dataset(
        tfrecord_filenames: list,
        cache: bool,
        repeats: bool,
        random_slice: bool,
        slice_len: int,
        num_slice: int, ## only required when using fixed_dataset
        shuffle: bool,
        buffer_size: int,
        global_batch_size: int,
        auto=tf.data.AUTOTUNE,
    ):
        """ Make dataset for verification using tf.data.Dataset() API.
        """
        ds = tf.data.TFRecordDataset(tfrecord_filenames, num_parallel_reads=auto)
        ds = ds.map(VerificationDataLoader.get_parse_audio_function(), num_parallel_calls=auto)

        if cache:
            ds = ds.cache()

        if repeats:
            ds = ds.repeat()

        ds = ds.map(VerificationDataLoader.get_slice_function(
            random_slice=random_slice,
            slice_len=slice_len,
            num_slice=num_slice
        ), num_parallel_calls=auto)

        if shuffle:
            ds = ds.shuffle(buffer_size, reshuffle_each_iteration=False)

        if global_batch_size:  # if not None
            ds = ds.batch(global_batch_size, num_parallel_calls=auto)

        ds = ds.map(VerificationDataLoader.get_check_numerics_function(), num_parallel_calls=auto)
        ds = ds.prefetch(auto)

        return ds
