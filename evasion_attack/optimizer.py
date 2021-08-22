import adabelief_tf


class Optimizers():

    @staticmethod
    def get_adabelief_optimizer(init_lr: float, rectify: bool, weight_decay: float):
        """ Get AdaBelief Optimizer.

            You can find more information below:
             - https://github.com/juntang-zhuang/Adabelief-Optimizer
        """
        return adabelief_tf.AdaBeliefOptimizer(
            learning_rate=init_lr,
            rectify=rectify,
            weight_decay=weight_decay,
            print_change_log=False,
        )