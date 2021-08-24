import tensorflow as tf

import argparse
import pprint

import numpy as np

from collections import OrderedDict
from pathlib import Path

from evasion_attack.attack import get_assets, save_npz
from evasion_attack.callbacks import Callbacks
from evasion_attack.centroids import Centroids
from evasion_attack.checkpoints import Checkpoints
from evasion_attack.dataset import IdentificationDataLoader, VerificationDataLoader
from evasion_attack.evaluate import EvaluateIdentificationModel, EvaluateVerificationModel
from evasion_attack.inference import InferenceIdentificationModel, InferenceVerificationModel
from evasion_attack.optimizer import Optimizers
from evasion_attack.utils import set_gpu_growthable, save_config

from evasion_attack.models.preprocess import Preprocessing
from evasion_attack.models.resnet import embedding_model
from evasion_attack.models.trainer import AngularPrototypicalModel


def define_argparser():
    """ Define argumemts.
    """
    p = argparse.ArgumentParser()

    ## Default.
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Arbitrary seed value for reproducibility. Default=%(default)s",
    )
    p.add_argument(
        "--model_type",
        type=str,
        default="iden", ## or "veri"
        choices=["iden", "veri"],
        help="Default=%(default)s",
    )
    p.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Default=%(default)s",
    )
    p.add_argument(
        "--train_model", 
        dest="train_model", 
        action="store_true",
    )
    p.add_argument(
        "--no-train_model", 
        dest="train_model", 
        action="store_false",
    )
    p.set_defaults(
        train_model=True
    )
    p.add_argument(
        "--clear_assets",
        type=bool,
        default=True,
        help="Default=%(default)s",
    )

    ## Path.
    p.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Default=%(default)s",
    )

    ## TFRecord Dataset.
    p.add_argument(
        "--global_batch_size",
        type=int,
        default=64,
        help="Default=%(default)s",
    )

    ## Modeling.
    p.add_argument(
        "--embedding_dim",
        type=int,
        default=512,
        help="Default=%(default)s",
    )

    ## Training hyper-parameters.
    p.add_argument(
        "--epochs",
        type=int,
        default=80, ## 80
        help="Default=%(default)s",
    )
    p.add_argument(
        "--init_lr",
        type=float,
        default=1e-3,
        help="Default=%(default)s",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=1./20, ## min_lr = lr * alpha = 5e-5
        help="Default=%(default)s",
    )
    p.add_argument(
        "--rectify",
        type=bool,
        default=True,
        help="Default=%(default)s",
    )
    p.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Default=%(default)s",
    )

    ## Callbacks.
    p.add_argument(
        "--checkpoint_callback",
        type=bool,
        default=True,
        help="Default=%(default)s",
    )
    p.add_argument(
        "--tensorboard_callback",
        type=bool,
        default=True,
        help="Default=%(default)s",
    )
    p.add_argument(
        "--learning_rate_schedular_callback",
        type=bool,
        default=True,
        help="Default=%(default)s",
    )
    p.add_argument(
        "--csv_logger_callback",
        type=bool,
        default=True,
        help="Default=%(default)s",
    )

    config = p.parse_args()

    ## Add additional arguments.
    config.__setattr__("sample_rate", 16_000)
    config.__setattr__("slice_len_sec", 2)
    config.__setattr__("slice_len", config.sample_rate * config.slice_len_sec)
    config.__setattr__("num_slice", 10)

    config.__setattr__("buffer_size", 150_000)

    config.__setattr__("tr_folder", str(Path(config.data_path, "vox1_dev_wav", "wav")))
    config.__setattr__("ts_folder", str(Path(config.data_path, "vox1_test_wav", "wav")))

    config.__setattr__("tfrec_folder", str(Path(config.data_path, "tfrecord")))
    config.__setattr__("iden_tfrec_folder", str(Path(config.tfrec_folder, "iden")))
    config.__setattr__("veri_tfrec_folder", str(Path(config.tfrec_folder, "veri")))

    config.__setattr__("ckpt_dir", "ckpt")
    config.__setattr__("log_dir", "logs")
    config.__setattr__("result_path", "result")

    config.__setattr__("num_classes_for_iden", 1_251)
    config.__setattr__("num_classes_for_veri", 1_211)

    config.__setattr__("iden_model_name", "AngularPrototypicalModel-Identification")
    config.__setattr__("veri_model_name", "AngularPrototypicalModel-Verification")

    config.__setattr__("num_iden_ts_ds", 8_251)
    config.__setattr__("num_veri_ts_ds", 37_720)

    config.__setattr__("attack_type", ["fgm", "pgd"])
    config.__setattr__("epsilon", [1e-3, 1e-2, 1e-1])

    return config


def build_dataset(config):
    """ Build dataset for identification or verification task tricky.
    """
    assert config.model_type.lower() in ["iden", "veri"]

    def _build_iden_dataset():
        """ Build dataset for identification task.
        """
        ## Load tfrecords and build dataset.
        tr_filenames = sorted(list(Path(config.iden_tfrec_folder).glob("tr_*.tfrec")))
        vl_filenames = sorted(list(Path(config.iden_tfrec_folder).glob("vl_*.tfrec")))
        ts_filenames = sorted(list(Path(config.iden_tfrec_folder).glob("ts_*.tfrec")))

        tr_ds = IdentificationDataLoader.get_dataset(
            tfrecord_filenames=tr_filenames,
            cache=False,
            repeats=False,
            random_slice=True,
            slice_len=config.slice_len_sec * config.sample_rate,
            num_slice=config.num_slice,
            shuffle=False,
            buffer_size=config.buffer_size,
            global_batch_size=config.global_batch_size,
        )
        vl_ds = IdentificationDataLoader.get_dataset(
            tfrecord_filenames=vl_filenames,
            cache=False,
            repeats=False,
            random_slice=True,
            slice_len=config.slice_len_sec * config.sample_rate,
            num_slice=config.num_slice,
            shuffle=False,
            buffer_size=config.buffer_size,
            global_batch_size=config.global_batch_size,
        )
        ts_ds = IdentificationDataLoader.get_dataset(
            tfrecord_filenames=ts_filenames,
            cache=False,
            repeats=False,
            random_slice=False,
            slice_len=config.slice_len_sec * config.sample_rate,
            num_slice=config.num_slice,
            shuffle=False,
            buffer_size=config.buffer_size,
            global_batch_size=None,
        )

        return tr_ds, vl_ds, ts_ds
    
    def _build_veri_dataset():
        """ Build dataset for verification task.
        """
        ## Load tfrecords and build dataset.
        tr_filenames = sorted(list(Path(config.veri_tfrec_folder).glob("tr_*.tfrec")))
        vl_filenames = sorted(list(Path(config.veri_tfrec_folder).glob("vl_*.tfrec")))
        ts_filenames = sorted(list(Path(config.veri_tfrec_folder).glob("ts_*.tfrec")))

        tr_ds = IdentificationDataLoader.get_dataset( ## not VerificationDataLoader
            tfrecord_filenames=tr_filenames,
            cache=False,
            repeats=False,
            random_slice=True,
            slice_len=config.slice_len_sec * config.sample_rate,
            num_slice=config.num_slice,
            shuffle=False,
            buffer_size=config.buffer_size,
            global_batch_size=config.global_batch_size,
        )
        vl_ds = IdentificationDataLoader.get_dataset( ## not VerificationDataLoader
            tfrecord_filenames=vl_filenames,
            cache=False,
            repeats=False,
            random_slice=True,
            slice_len=config.slice_len_sec * config.sample_rate,
            num_slice=config.num_slice,
            shuffle=False,
            buffer_size=config.buffer_size,
            global_batch_size=config.global_batch_size,
        )
        ts_ds = VerificationDataLoader.get_dataset(
            tfrecord_filenames=ts_filenames,
            cache=False,
            repeats=False,
            random_slice=False,
            slice_len=config.slice_len_sec * config.sample_rate,
            num_slice=config.num_slice,
            shuffle=False,
            buffer_size=config.buffer_size,
            global_batch_size=None,
        )

        return tr_ds, vl_ds, ts_ds

    tr_ds, vl_ds, ts_ds = _build_iden_dataset() if config.model_type.lower() == "iden" else _build_veri_dataset()

    ## Priht the shapes.
    print(f"tr_ds: {tr_ds}")
    print(f"vl_ds: {vl_ds}")
    print(f"ts_ds: {ts_ds}")

    return tr_ds, vl_ds, ts_ds


def build_model(config):
    sample_rate_ms = int(config.sample_rate / 1_000)
    num_classes = config.num_classes_for_iden if config.model_type.lower() == "iden" else config.num_classes_for_veri

    ## Naming.
    if config.model_name == None:
        config.model_name = config.iden_model_name if config.model_type == "iden" else config.veri_model_name

    ## Define the parts.
    header = Preprocessing(
        frame_length=25 * sample_rate_ms,
        frame_step=10 * sample_rate_ms,
        fft_length=512,
        pad_end=True,
        num_mel_bins=64,
        sample_rate=config.sample_rate,
        lower_edge_hertz=0.,
        upper_edge_hertz=8_000.,
    )
    emb_model = embedding_model(
        input_shape=(config.slice_len_sec * config.sample_rate,),
        num_classes=num_classes,
        embedding_dim=config.embedding_dim,
        preprocessing_fn=header,
    )
    centroids = Centroids(
        num_classes=num_classes,
        embedding_dim=config.embedding_dim,
    )
    model = AngularPrototypicalModel(
        embedding_model=emb_model,
        centroids=centroids,
        name=config.model_name,
    )

    return model


def get_callbacks(config):
    """ Get all callbacks.
    """
    callbacks = list()

    if config.checkpoint_callback:
        callbacks.append(Callbacks.get_checkpoint_callback(
            ckpt_dir=str(Path(config.ckpt_dir, config.model_type)),
            clear_assets=config.clear_assets,
        ))

    if config.tensorboard_callback:
        callbacks.append(Callbacks.get_tensorboard_callback(
            log_dir=str(Path(config.log_dir, "fit")),
            model_name=config.model_name,
            clear_assets=config.clear_assets,
        ))

    if config.learning_rate_schedular_callback:
        callbacks.append(Callbacks.get_learning_rate_schedular_callback(
            init_lr=config.init_lr,
            epochs=config.epochs,
            alpha=config.epochs,
        ))

    if config.csv_logger_callback:
        callbacks.append(Callbacks.get_csv_logger_callback(
            log_dir=Path(config.log_dir, "csv"),
            model_name=config.model_name,
            clear_assets=config.clear_assets,
        ))

    return callbacks


def get_latest_model(config, model: tf.keras.Model):
    """ Make clean and get latest model.
    """
    Checkpoints.make_clean(
        ckpt_dir=str(Path(config.ckpt_dir, config.model_type)),
    )
    latest_model = Checkpoints.load_latest_checkpoint(
        ckpt_dir=str(Path(config.ckpt_dir, config.model_type)), 
        model=model,
    )

    return latest_model


def get_prediction(config, latest_model: tf.keras.Model, ts_ds: tf.data.Dataset, save: bool):
    """ Get 'y_true' and 'y_pred'.
    """
    if config.model_type.lower() == "iden":
        total = config.num_iden_ts_ds
        y_true, y_pred = InferenceIdentificationModel.inference_fixed_sliced_dataset(
            latest_model=latest_model,
            ts_ds=ts_ds,
            total=total,
        )

    else:
        total = config.num_veri_ts_ds
        y_true, y_pred = InferenceVerificationModel.inference_fixed_sliced_dataset(
            latest_model=latest_model,
            ts_ds=ts_ds,
            total=total,
        )
    
    ## Print the shapes.
    print(f"y_true.shape: {y_true.shape}, y_pred.shape: {y_pred.shape}")

    ## Save the results.
    if save:
        assets = get_assets(model_type=config.model_type, ds_type="fixed", attack_type=None, epsilon=None)
        assets.update({"y_true": y_true, "y_pred": y_pred, "dB_x_delta": None})
        save_npz(assets, save_dir=config.result_path)

    return y_true, y_pred


def get_evaluation(config, y_true: np.ndarray, y_pred: np.ndarray):
    """ Do evaluate performance.
    """
    if config.model_type.lower() == "iden":
        foo = EvaluateIdentificationModel.cmc(y_true, y_pred)
        print(f"Top-1 accuracy: {foo[0]:.4f}, Top-5 accuracy: {foo[4]:.4f}")

    else:
        foo = EvaluateVerificationModel.eer(y_true, y_pred)
        bar = EvaluateVerificationModel.auroc(y_true, y_pred)
        qux = EvaluateVerificationModel.min_dcf(y_true, y_pred)
        print(f"EER: {foo:.2f}, AUROC: {bar:.4f}, minDCF: {qux:.4f}")


def main(config):
    """ Main body.
    """
    def print_config(config):
        ## 'sort_dicts=False' params can only apply python>=3.8.
        pprint.PrettyPrinter(indent=4).pprint(OrderedDict(vars(config)))
    print_config(config)

    ## Set gpu memory growthable.
    set_gpu_growthable()

    ## Load tfrecords and build dataset.
    tr_ds, vl_ds, ts_ds = build_dataset(config)

    ## Modeling.
    ##  - total params:         8_082_750
    ##  - trainable params:     7_432_219
    ##  - non-trainable params:   650_531
    model = build_model(config)

    ## After forcibly building the model, print the number of parameters.
    model.build([config.global_batch_size, config.slice_len_sec * config.sample_rate])
    model.summary()

    if config.train_model:
        ## Compile.
        model.compile(
            ## Custom arguments.
            ds=tr_ds,
            loss_fn=tf.keras.metrics.Mean(name="loss"),
            metric_fn=tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
            ## Original arguments.
            optimizer=Optimizers.get_adabelief_optimizer(
                init_lr=config.init_lr,
                rectify=config.rectify,
                weight_decay=config.weight_decay,
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )

        ## Fit.
        _ = model.fit(
            tr_ds,
            validation_data=vl_ds,
            epochs=config.epochs,
            verbose=1,
            callbacks=get_callbacks(config),
        )

    ## Make clean checkpoints and load latest version.
    latest_model = get_latest_model(config, model)

    ## Inference with fixed-sliced dataset and save it.
    y_true, y_pred = get_prediction(config, latest_model, ts_ds, save=True)

    ## Evaluate with some performance.
    get_evaluation(config, y_true, y_pred)

    ## Save configuration.
    save_config(vars(config), file_path=Path("config", "train.json"))


if __name__ == "__main__":
    config = define_argparser()
    main(config)
