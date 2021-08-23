import tensorflow as tf

import json
import zipfile

from pathlib import Path


def check_file_exists(file_path: str):
    """ Check file exists.
    """
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"{file_path} is not a file.")
    else:
        print(f"Check {file_path} exists: Ok.")


def unzip(
    data_path: str,
    tr_fname: str = "vox1_dev_wav.zip",
    ts_fname: str = "vox1_test_wav.zip",
    num_tr_files: int = 148_642,
    num_ts_files: int = 4_874,
):
    """ Unzip data.
    """
    ## Check whether the file exists.
    if not Path(data_path, tr_fname).exists():
        raise FileNotFoundError
    if not Path(data_path, ts_fname).exists():
        raise FileNotFoundError

    ## Then unzip the files.
    def _extract(file_path, dest):
        print(f"Extracting file: {file_path}...", end=" ")
        with zipfile.ZipFile(file_path, 'r') as f:
            f.extractall(dest)
        print("done.")

    try:
        _extract(Path(data_path, tr_fname), Path(
            data_path, Path(tr_fname).stem))
        _extract(Path(data_path, ts_fname), Path(
            data_path, Path(ts_fname).stem))
    except Exception as e:
        print(e)

    ## And check the number of files.
    num = len(list(Path(data_path).glob("vox1_dev_wav/*/*/*/*.wav")))
    if num != num_tr_files:
        raise AssertionError(
            f"The number of tr_files does not match a predetermined value.\
                 {num} (current) != {num_tr_files} (original)")
    else:
        print(f"The number of tr_files: {num}... OK.")

    num = len(list(Path(data_path).glob("vox1_test_wav/*/*/*/*.wav")))
    if num != num_ts_files:
        raise AssertionError(
            f"The number of ts_files does not match a predetermined value.\
                 {num} (current) != {num_tr_files} (original)")
    else:
        print(f"The number of ts_files: {num}... OK.")


def set_gpu_growthable():
    """ Set gpu memory growthable.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            ## Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs.")
        except RuntimeError as e:
            ## Memory growth must be set before GPUs have been initialized
            print(e)


def save_config(config, file_path: Path):
    """ Save configuration.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(config, fp=f, indent=4)
    print(f"Configuration saved to {file_path}.")
