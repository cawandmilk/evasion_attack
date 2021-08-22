import tensorflow as tf

import numpy as np
import pandas as pd

from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm


class TFRecordGenerator():

    def __init__(
        self,
        seed: int,
        tr_folder: str,
        ts_folder: str,
        split_ids: str,
        veri_test: str,
        iden_tfrec_folder: str,
        veri_tfrec_folder: str,
        num_classes_for_iden: int,
        num_classes_for_veri: int,
        file_num_per_tfrecord: int = 5_000,
        test_size: float = 0.2,
    ):
        """Define common params.
        """
        self.seed = seed
        self.tr_folder = tr_folder
        self.ts_folder = ts_folder
        self.split_ids = split_ids
        self.veri_test = veri_test
        self.iden_tfrec_folder = iden_tfrec_folder
        self.veri_tfrec_folder = veri_tfrec_folder
        self.num_classes_for_iden = num_classes_for_iden
        self.num_classes_for_veri = num_classes_for_veri

        self.file_num_per_tfrecord = file_num_per_tfrecord
        self.test_size = test_size

        self.TR_IDS = 1
        self.VL_IDS = 2
        self.TS_IDS = 3

        ## All folder paths must be created in advance.
        Path(self.iden_tfrec_folder).mkdir(parents=True, exist_ok=True)
        Path(self.veri_tfrec_folder).mkdir(parents=True, exist_ok=True)


    def _make_full_path(self, path):
        """Make the path absolute, resolving any symlinks.
        """
        if Path(self.tr_folder, path).exists():
            return Path(self.tr_folder, path)
        elif Path(self.ts_folder, path).exists():
            return Path(self.ts_folder, path)
        else:
            raise AssertionError(f"{path} is not exists either in dev and test folders.")


    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte.
        """
        if isinstance(value, type(tf.constant(0))):
            # BytesList won't unpack a string from an EagerTensor.
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    def _float_feature(self, value):
        """Returns a float_list from a float / double.
        """
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint.
        """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    def _iden_audio_example(self, audio_path, label):
        """Generate audio examples for identification task. (classification)
        """
        if isinstance(audio_path, type(Path("."))):
            audio_path = str(audio_path)

        audio_string = tf.io.read_file(audio_path)
        audio_shape = tf.audio.decode_wav(audio_string).audio.shape[0]

        feature = {
            "audio_raw": self._bytes_feature(audio_string),
            "length": self._int64_feature(audio_shape),
            "label": self._int64_feature(label),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))


    def _veri_audio_example(self, audio_path_1, audio_path_2, y_true):
        """Generate audio examples for verification task. (similarity)
        """
        if isinstance(audio_path_1, type(Path("."))):
            audio_path_1 = str(audio_path_1)
        if isinstance(audio_path_2, type(Path("."))):
            audio_path_2 = str(audio_path_2)

        audio_string_1 = tf.io.read_file(audio_path_1)
        audio_shape_1 = tf.audio.decode_wav(audio_string_1).audio.shape[0]

        audio_string_2 = tf.io.read_file(audio_path_2)
        audio_shape_2 = tf.audio.decode_wav(audio_string_2).audio.shape[0]

        feature = {
            "audio_raw_1": self._bytes_feature(audio_string_1),
            "audio_raw_2": self._bytes_feature(audio_string_2),
            "length_1": self._int64_feature(audio_shape_1),
            "length_2": self._int64_feature(audio_shape_2),
            "y_true": self._int64_feature(y_true),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))


    def _make_iden_tfrecords(
        self,
        X_path: np.array,
        Y: np.array,
        training_type: int,
        dest: str,
    ):
        """Write tfrecords for identification task.
        """
        tfrecord_num = int(np.ceil(X_path.shape[0] / self.file_num_per_tfrecord))
        prefix = {self.TR_IDS: "tr", self.VL_IDS: "vl", self.TS_IDS: "ts"}[training_type]

        ## Genate tfrecords for each loops.
        desc = f"Generating {prefix} dataset in {dest}"
        for i in tqdm(range(tfrecord_num), total=tfrecord_num, desc=desc):
            sub_X_path = X_path[self.file_num_per_tfrecord * i: self.file_num_per_tfrecord * (i+1)]
            sub_Y = Y[self.file_num_per_tfrecord * i: self.file_num_per_tfrecord * (i+1)]

            ## Example: data/tfrecord/iden/tr_001_005000.tfrec
            record_name = Path(dest, f"{prefix}_{i:03d}_{sub_X_path.shape[0]:05d}.tfrec")
            with tf.io.TFRecordWriter(str(record_name)) as writer: ## only allow str-type-path, not Pathlib.
                for x_path, y in zip(sub_X_path, sub_Y):
                    tf_example = self._iden_audio_example(x_path, y)
                    writer.write(tf_example.SerializeToString())

            ## Delete temporary files to save memory.
            del sub_X_path, sub_Y


    def _make_veri_tfrecords(
        self,
        X_path_1: np.array,
        X_path_2: np.array,
        y_true: np.array,
        training_type: int,
        dest: str,
    ):
        """Write tfrecords for verification task.
        """
        tfrecord_num = int(np.ceil(X_path_1.shape[0] / self.file_num_per_tfrecord)) ## 2_500
        prefix = {self.TR_IDS: "tr", self.VL_IDS: "vl", self.TS_IDS: "ts"}[training_type]

        ## Genate tfrecords for each loops.
        desc = f"Generating {prefix} dataset in {dest}"
        for i in tqdm(range(tfrecord_num), total=tfrecord_num, desc=desc):
            sub_X_path_1 = X_path_1[self.file_num_per_tfrecord * i : self.file_num_per_tfrecord * (i+1)]
            sub_X_path_2 = X_path_2[self.file_num_per_tfrecord * i : self.file_num_per_tfrecord * (i+1)]
            sub_y_true = y_true[self.file_num_per_tfrecord * i : self.file_num_per_tfrecord * (i+1)]

            ## Example: data/tfrecord/veri/ts_001_005000.tfrec
            record_name = Path(dest, f"{prefix}_{i:03d}_{sub_X_path_1.shape[0]:05d}.tfrec")
            with tf.io.TFRecordWriter(str(record_name)) as writer: ## only allow str-type-path, not Pathlib.
                for x_path_1, x_path_2, y in zip(sub_X_path_1, sub_X_path_2, sub_y_true):
                    tf_example = self._veri_audio_example(x_path_1, x_path_2, y)
                    writer.write(tf_example.SerializeToString())
        
            ## Delete temporary files to save memory.
            del sub_X_path_1, sub_X_path_2, sub_y_true


    def generate_iden_tfrecords(
        self,
    ):
        """Generate tfrecords for identification task.
        """
        ## Load split ids.
        ids = pd.read_csv(self.split_ids, sep=" ", header=None, names=["training_type", "path"])
        ids = ids.sort_values(by="path")

        ## Insert id(=class) and modify filepath.
        ids["id"] = ids["path"].apply(lambda x: Path(x).parts[-3])
        ids["path"] = ids["path"].apply(self._make_full_path)

        ## Split to train & validate & test.
        tr_ids = ids[ids["training_type"] == self.TR_IDS]
        vl_ids = ids[ids["training_type"] == self.VL_IDS]
        ts_ids = ids[ids["training_type"] == self.TS_IDS]

        ## Shuffle only training ids.
        tr_ids = tr_ids.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        ## Extract values.
        tr_X_path, tr_Y = tr_ids["path"].values, tr_ids["id"].values
        vl_X_path, vl_Y = vl_ids["path"].values, vl_ids["id"].values
        ts_X_path, ts_Y = ts_ids["path"].values, ts_ids["id"].values

        print("Assets of identification files:")
        print(f"  # of tr_X_path: {tr_X_path.shape[0]:>6}, tr_Y: {tr_Y.shape[0]:>6d}")
        print(f"  # of vl_X_path: {vl_X_path.shape[0]:>6}, vl_Y: {vl_Y.shape[0]:>6d}")
        print(f"  # of ts_X_path: {ts_X_path.shape[0]:>6}, ts_Y: {ts_Y.shape[0]:>6d}")

        ## Every classes have exactly 1_251 unique classes.
        ##  (preliminary of closed set identification tast)
        assert np.all([len(np.unique(label)) == self.num_classes_for_iden for label in [tr_Y, vl_Y, ts_Y]])

        ## Map class ids to integer.
        ##  Example: id00001 -> 0, id00002 -> 1, ...
        mapping_table = OrderedDict({j:i for i, j in enumerate(sorted(np.unique(tr_Y)))})
        # self.iden_mapping_table = mapping_table

        tr_Y = [mapping_table[i] for i in tr_Y]
        vl_Y = [mapping_table[i] for i in vl_Y]
        ts_Y = [mapping_table[i] for i in ts_Y]

        ## Now, we can generate tfrecords.
        self._make_iden_tfrecords(tr_X_path, tr_Y, self.TR_IDS, dest=self.iden_tfrec_folder)
        self._make_iden_tfrecords(vl_X_path, vl_Y, self.VL_IDS, dest=self.iden_tfrec_folder)
        self._make_iden_tfrecords(ts_X_path, ts_Y, self.TS_IDS, dest=self.iden_tfrec_folder)


    def generate_veri_tfrecords(
        self,
    ):
        """Generate tfrecords for verification task.
        """
        ## First, we generate shuffled training and validation dataset.
        tot_X_path = np.array(sorted([i for i in Path(self.tr_folder).glob("*/*/*.wav")]))

        np.random.seed(self.seed)
        np.random.shuffle(tot_X_path)

        tot_Y = np.array([i.parts[-3] for i in tot_X_path])

        ## And we can extract the integer labels.
        mapping_table = OrderedDict({j:i for i, j in enumerate(sorted(np.unique(tot_Y)))})
        tot_Y = [mapping_table[i] for i in tot_Y]

        ## Next, split the total dataset into training and validation.
        num_tr = int(np.ceil(len(tot_X_path) * (1. - self.test_size)))

        tr_X_path, tr_Y = tot_X_path[:num_tr], tot_Y[:num_tr]
        vl_X_path, vl_Y = tot_X_path[num_tr:], tot_Y[num_tr:]

        assert np.all([len(np.unique(label)) == self.num_classes_for_veri for label in [tr_Y, vl_Y]])

        ## Generate tfrecords for identificaiton task, not verification.
        self._make_iden_tfrecords(tr_X_path, tr_Y, self.TR_IDS, dest=self.veri_tfrec_folder)
        self._make_iden_tfrecords(vl_X_path, vl_Y, self.VL_IDS, dest=self.veri_tfrec_folder)

        ## Now we can take verification pairs.
        ids = pd.read_csv(self.veri_test, sep=" ", header=None, names=["y_true", "path_1", "path_2"])

        ids["path_1"] = ids["path_1"].apply(self._make_full_path)
        ids["path_2"] = ids["path_2"].apply(self._make_full_path)

        y_true = ids["y_true"].values
        ts_X_path_1 = ids["path_1"].values
        ts_X_path_2 = ids["path_2"].values

        ## Generate tfrecords for verification task, not identificaiton.
        self._make_veri_tfrecords(ts_X_path_1, ts_X_path_2, y_true, self.TS_IDS, dest=self.veri_tfrec_folder)
