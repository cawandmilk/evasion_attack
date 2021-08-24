import argparse
import pprint

from collections import OrderedDict
from pathlib import Path

from evasion_attack.utils import check_file_exists, unzip, save_config
from evasion_attack.tfrecord import TFRecordGenerator


def define_argparser():
    """ Define arguments.
    """
    p = argparse.ArgumentParser()

    ## Seed.
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Arbitrary seed value for reproducibility. Default=%(default)s",
    )

    ## Path.
    p.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Default=%(default)s",
    )

    ## Params.
    p.add_argument(
        "--unzip", 
        dest="unzip", 
        action="store_true",
    )
    p.add_argument(
        "--no-unzip", 
        dest="unzip", 
        action="store_false",
    )
    p.set_defaults(
        unzip=True
    )
    p.add_argument(
        "--generate_tfrecords", 
        dest="generate_tfrecords", 
        action="store_true",
    )
    p.add_argument(
        "--no-generate_tfrecords", 
        dest="generate_tfrecords", 
        action="store_false",
    )
    p.set_defaults(
        generate_tfrecords=True
    )
    p.add_argument(
        "--file_num_per_tfrecord",
        type=int,
        default=5_000,
        help="Default=%(default)s",
    )

    config = p.parse_args()

    ## Add some arguments.
    config.__setattr__("tr_folder", str(Path(config.data_path, "vox1_dev_wav", "wav")))
    config.__setattr__("ts_folder", str(Path(config.data_path, "vox1_test_wav", "wav")))
    config.__setattr__("split_ids", str(Path(config.data_path, "iden_split.txt")))
    config.__setattr__("veri_test", str(Path(config.data_path, "veri_test.txt")))

    config.__setattr__("tfrec_folder", str(Path(config.data_path, "tfrecord")))
    config.__setattr__("iden_tfrec_folder", str(Path(config.tfrec_folder, "iden")))
    config.__setattr__("veri_tfrec_folder", str(Path(config.tfrec_folder, "veri")))

    config.__setattr__("num_classes_for_iden", 1_251)
    config.__setattr__("num_classes_for_veri", 1_211)

    return config


def main(config):
    """ Main body.
    """
    def print_config(config):
        pprint.PrettyPrinter(indent=4).pprint(OrderedDict(vars(config)))
    print_config(config)

    ## Unzip.
    if config.unzip:
        ## Assert the zip file exists.
        for fname in ["vox1_dev_wav.zip", "vox1_test_wav.zip", "iden_split.txt", "veri_test.txt"]:
            check_file_exists(Path(config.data_path, fname))
        ## Unzip.
        unzip(data_path=config.data_path)

    ## Generate tfrecord dataset.
    if config.generate_tfrecords:
        ## Whether iden or veri, datasets are, after all, one kind, 
        ## creating objects that specify a common path.
        gen = TFRecordGenerator(
            seed=config.seed,
            tr_folder=config.tr_folder,
            ts_folder=config.ts_folder,
            split_ids=config.split_ids,
            veri_test=config.veri_test,
            iden_tfrec_folder=config.iden_tfrec_folder,
            veri_tfrec_folder=config.veri_tfrec_folder,
            num_classes_for_iden=config.num_classes_for_iden,
            num_classes_for_veri=config.num_classes_for_veri,
        )

        gen.generate_iden_tfrecords()
        gen.generate_veri_tfrecords()

    ## Save configuation.
    save_config(vars(config), file_path=Path("config", "preliminary.json"))


if __name__ == "__main__":
    config = define_argparser()
    main(config)
