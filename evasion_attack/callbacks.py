import tensorflow as tf

import shutil

from pathlib import Path


class Callbacks():

    @staticmethod
    def get_checkpoint_callback(ckpt_dir: str, clear_assets: bool):
        """ Get callbacks s.t. monitering val_loss.

            Unlike PyTorch, it only tracks the state values 
            for the model and does not participate in saving 
            the optimizer.
        """
        ## Clear folders.
        if clear_assets:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

        ckpt_path = Path(ckpt_dir, "cp-{epoch:03d}-{val_loss:.4f}.ckpt")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            verbose=0,
            monitor="val_loss",
            save_weights_only=True,
            save_best_only=True,
        )

        return cp_callback

    
    @staticmethod
    def get_tensorboard_callback(log_dir: str, model_name: str, clear_assets: bool):
        """ Get callbacks s.t. record logs for tensorboard visualize.
        """
        ## Clear folders.
        if clear_assets:
            shutil.rmtree(log_dir, ignore_errors=True)

        log_dir = Path(log_dir, model_name)
        # log_dir.mkdir(parents=True, exist_ok=True)

        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
        )

        return tb_callback


    @staticmethod
    def get_learning_rate_schedular_callback(init_lr: float, epochs: int, alpha: float):
        """ Get callbacks s.t. schedule learning rates.

            It performs a function similar to 'torch.optim.lr_schedular()' 
            of pytorch, except that lr is updated at the end of every 
            epoch rather than every mini-batch.
        """
        lr_callback = tf.keras.callbacks.LearningRateScheduler(
            schedule=tf.keras.experimental.CosineDecay(
                initial_learning_rate=init_lr,
                decay_steps=epochs,
                alpha=alpha,
            ),
            verbose=0,
        )

        return lr_callback

    
    @staticmethod
    def get_csv_logger_callback(log_dir: str, model_name: str, clear_assets: bool):
        """ Get callbacks s.t. record logs to csv file.

            It is similar to tb_callback, but it is simpler and 
            performs fewer functions.
        """
        ## Clear folders.
        if clear_assets:
            shutil.rmtree(log_dir, ignore_errors=True)
        
        file_path = Path(log_dir, f"{model_name}.csv")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        lg_callback = tf.keras.callbacks.CSVLogger(file_path)

        return lg_callback
