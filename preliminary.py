import argparse
import pprint
from pathlib import Path

from evasion_attack.utils import check_file_exists, unzip
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
        default=str(Path(".", "data")),
        help="Default=%(default)s",
    )
    p.add_argument(
        "--tr_folder",
        type=str,
        default=str(Path(".", "data", "vox1_dev_wav", "wav")),
        help="Default=%(default)s",
    )
    p.add_argument(
        "--ts_folder",
        type=str,
        default=str(Path(".", "data", "vox1_test_wav", "wav")),
        help="Default=%(default)s",
    )
    p.add_argument(
        "--split_ids",
        type=str,
        default=str(Path(".", "data", "iden_split.txt")),
        help="Default=%(default)s",
    )
    p.add_argument(
        "--veri_test",
        type=str,
        default=str(Path(".", "data", "veri_test.txt")),
        help="Default=%(default)s",
    )

    p.add_argument(
        "--iden_tfrec_folder",
        type=str,
        default=str(Path(".", "data", "tfrecord", "iden")),
        help="Default=%(default)s",
    )
    p.add_argument(
        "--veri_tfrec_folder",
        type=str,
        default=str(Path(".", "data", "tfrecord", "veri")),
        help="Default=%(default)s",
    )

    ## Params.
    p.add_argument(
        "--unzip",
        type=bool,
        default=False,  # True
        help="Default=%(default)s",
    )
    p.add_argument(
        "--generate_tfrecords",
        type=bool,
        default=True,  # True
        help="Default=%(default)s",
    )
    p.add_argument(
        "--file_num_per_tfrecord",
        type=int,
        default=5_000,
        help="Default=%(default)s",
    )
    p.add_argument(
        "--num_classes_for_iden",
        type=int,
        default=1_251,
        help="Default=%(default)s",
    )
    p.add_argument(
        "--num_classes_for_veri",
        type=int,
        default=1_211,
        help="Default=%(default)s",
    )

    config = p.parse_args()

    return config


def main(config):
    """ Main body.
    """
    def print_config(config):
        pprint.PrettyPrinter(indent=4).pprint(vars(config))
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


if __name__ == "__main__":
    config = define_argparser()
    main(config)
