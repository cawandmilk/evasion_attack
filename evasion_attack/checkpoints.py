import tensorflow as tf

from pathlib import Path


class Checkpoints():

    @staticmethod
    def make_clean(ckpt_dir: str, max_to_keep: int = 1):
        """ Unlink all checkpoints exclude the number of 'max_to_keep'.
        """
        ## Find the latest checkpoint.
        with open(Path(ckpt_dir, "checkpoint"), "r") as f:
            latest_checkpoint = f.readline().split()[-1].replace("\"", "")

        free_size = 0
        for f in Path(ckpt_dir).glob("*.ckpt.*"):
            if not (latest_checkpoint in str(f)):
                free_size += f.stat().st_size
                f.unlink()

        print(f"Checkpoint folder {ckpt_dir} is now clean, {free_size / (2 ** 20):.2f}MB free.")


    @staticmethod
    def load_latest_checkpoint(ckpt_dir: str, model: tf.keras.Model):
        """ Load latest checkpoints.
        """
        ## Load latest checkpoint.
        latest = tf.train.latest_checkpoint(ckpt_dir)
        if latest is None:
            raise FileNotFoundError(f"No checkpoints in {ckpt_dir}...")

        ## The model need to be compiled.
        ckpt = tf.train.Checkpoint(model)
        ckpt.restore(latest).expect_partial()

        print(f"Restored checkpoints: {latest}")

        return model
