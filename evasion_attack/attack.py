import tensorflow as tf

import numpy as np

from pathlib import Path
from tqdm import tqdm

from cleverhans.tf2.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.tf2.attacks.fast_gradient_method import fast_gradient_method


def get_assets(model_type: str, ds_type: str, attack_type: str, epsilon: float):
    """ Get asset dictionary.
    """
    ## Assertion phase.
    if not (model_type in ["iden", "veri"]):
        raise AssertionError(f"Argument 'model_type' must be 'iden' or 'veri': not {model_type}")

    if not (ds_type in ["random", "fixed"]):
        raise AssertionError(f"Argument 'ds_type' must be 'random' or 'fixed': not {ds_type}")

    if not (attack_type in [None, "fgm", "pgd"]):
        raise AssertionError(f"Argument 'attack_type' must be 'fgm' or 'pgd': not {attack_type}")

    if not (epsilon in [None, 1e-1, 1e-2, 1e-3]):
        raise AssertionError(f"Argument 'epsilon' must be 1e-1, 1e-2, or 1e-3: not {epsilon}")

    return {
        "y_true": list(),
        "y_pred": list(),
        "model_type": model_type,
        "ds_type": ds_type,
        "attack_type": attack_type,
        "epsilon": epsilon,
        "dB_x_delta": list(),
    }


def get_dB_x_delta(x: tf.Tensor, x_adv: tf.Tensor):
    """ Get dB_x_delta.
    """
    delta = x_adv - x
    dB_delta = 20 * np.log10(tf.math.reduce_max(delta, axis=-1).numpy())
    x_delta = 20 * np.log10(tf.math.reduce_max(x, axis=-1).numpy())

    dB_x_delta = dB_delta - x_delta

    return dB_x_delta


def save_npz(assets: dict, save_dir: str):
    """ Save result as npz format.
    """    
    ## Save as numpy zip format.
    ##  e.g. result/iden-random-fgm-0.1.npz
    file_name = Path(save_dir, f"{assets['model_type']}-{assets['ds_type']}-{assets['attack_type']}-{assets['epsilon']}")
    file_name.parent.mkdir(exist_ok=True, parents=True)

    np.savez(file_name, **assets)


def load_npz(save_dir: str, model_type: str, ds_type: str, attack_type: str, epsilon: float):
    """ Load result as npz format.
    """    
    ## Save as numpy zip format.
    file_name = Path(save_dir, f"{model_type}-{ds_type}-{attack_type}-{epsilon}.npz")
    assets = np.load(file_name, allow_pickle=True)

    ## Return as dictionary type.
    return {file_name: assets}


class AttackIdentificationModel():

    @staticmethod
    def attack_random_sliced_dataset(latest_model: tf.keras.Model, ds: tf.data.Dataset, attack_type: str, epsilon: float, total: int):
        """ Attack identification model using random sliced dataset.
        """
        assets = get_assets("iden", "random", attack_type, epsilon)

        desc = f"{'iden'}-{'random'}-{attack_type}-{epsilon}"
        for element in tqdm(ds, total=total, desc=desc):
            ## Unpack.
            inp, tar = element

            ## Generate adversarial examples.
            if attack_type == "fgm":
                x_adv = fast_gradient_method(
                    model_fn=latest_model.embedding_model,
                    x=inp,
                    eps=epsilon,
                    norm=np.inf,
                )
            elif attack_type == "pgd":
                x_adv = projected_gradient_descent(
                    model_fn=latest_model.embedding_model,
                    x=inp,
                    eps=epsilon,
                    eps_iter=epsilon * 0.1,
                    nb_iter=40,
                    norm=np.inf,
                )

            ## Predict.
            _, scaled_similarity = latest_model.predict(x_adv)

            ## Append to list.
            assets["y_true"].append(tar)
            assets["y_pred"].append(scaled_similarity)
            assets["dB_x_delta"].append(get_dB_x_delta(inp, x_adv))

        ## Postprocess.
        assets["y_true"] = np.concatenate(assets["y_true"], axis = 0)
        assets["y_pred"] = tf.nn.softmax(np.concatenate(assets["y_pred"], axis=0)).numpy()
        assets["dB_x_delta"] = np.concatenate(assets["dB_x_delta"], axis = 0)

        return assets


    @staticmethod
    def attack_fixed_sliced_dataset(latest_model: tf.keras.Model, ds: tf.data.Dataset, attack_type: str, epsilon: float, total: int):
        """ Attack identification model using fixed sliced dataset.
        """
        assets = get_assets("iden", "fixed", attack_type, epsilon)

        desc = f"{'iden'}-{'fixed'}-{attack_type}-{epsilon}"
        for element in tqdm(ds, total=total, desc=desc):
            ## Unpack.
            inp, tar = element

            ## Generate adversarial examples.
            if attack_type == "fgm":
                x_adv = fast_gradient_method(
                    model_fn=latest_model.embedding_model,
                    x=inp,
                    eps=epsilon,
                    norm=np.inf,
                )
            elif attack_type == "pgd":
                x_adv = projected_gradient_descent(
                    model_fn=latest_model.embedding_model,
                    x=inp,
                    eps=epsilon,
                    eps_iter=epsilon * 0.1,
                    nb_iter=40,
                    norm=np.inf,
                )

            ## Predict.
            _, scaled_similarity = latest_model.predict(x_adv)

            ## Append to list.
            assets["y_true"].append(tar[0])
            assets["y_pred"].append(scaled_similarity)
            assets["dB_x_delta"].append(np.mean(get_dB_x_delta(inp, x_adv)))

        ## Postprocess.
        assets["y_true"] = np.concatenate(assets["y_true"], axis=0)
        assets["y_pred"] = tf.nn.softmax(np.mean(np.stack(assets["y_pred"], axis=0), axis=1), axis=-1).numpy()
        assets["dB_x_delta"] = np.array(assets["dB_x_delta"])

        return assets


class AttackVerificationModel():

    @staticmethod
    def attack_random_sliced_dataset(latest_model: tf.keras.Model, ds: tf.data.Dataset, attack_type: str, epsilon: float, total: int):
        """ Attack verification model using random sliced dataset.
        """
        assets = get_assets("veri", "random", attack_type, epsilon)

        desc = f"{'veri'}-{'random'}-{attack_type}-{epsilon}"
        for element in tqdm(ds, total=total, desc=desc):
            ## Unpack.
            inp_1, inp_2, tar = element

            ## Generate adversarial examples with only the first audio (=inp_1).
            if attack_type == "fgm":
                x_adv = fast_gradient_method(
                    model_fn=latest_model.embedding_model,
                    x=inp_1,
                    eps=epsilon,
                    norm=np.inf,
                )
            elif attack_type == "pgd":
                x_adv = projected_gradient_descent(
                    model_fn=latest_model.embedding_model,
                    x=inp_1,
                    eps=epsilon,
                    eps_iter=epsilon * 0.1,
                    nb_iter=40,
                    norm=np.inf,
                )

            ## Predict.
            inp = tf.concat([x_adv, inp_2], axis=0)
            embeddings = latest_model.embedding_model.predict(inp)
            
            ## Calculate cosine similarity.
            similarity = tf.einsum("ik,ik->i", *tf.split(embeddings, num_or_size_splits=2, axis=0))

            ## Append to list.
            assets["y_true"].append(tar)
            assets["y_pred"].append(similarity)
            assets["dB_x_delta"].append(get_dB_x_delta(inp_1, x_adv))

        ## Postprocess.
        assets["y_true"] = np.concatenate(assets["y_true"], axis=0)
        assets["y_pred"] = np.concatenate(assets["y_pred"], axis=0)
        assets["dB_x_delta"] = np.concatenate(assets["dB_x_delta"], axis=0)

        ## [0, 1] -> [-1, 1]
        assets["y_true"] = np.where(assets["y_true"] == 0, -1, assets["y_true"])

        return assets


    @staticmethod
    def attack_fixed_sliced_dataset(latest_model: tf.keras.Model, ds: tf.data.Dataset, attack_type: str, epsilon: float, total: int):
        """ Attack verification model using fixed sliced dataset.
        """
        assets = get_assets("veri", "fixed", attack_type, epsilon)

        desc = f"{'veri'}-{'fixed'}-{attack_type}-{epsilon}"
        for element in tqdm(ds, total=total, desc=desc):
            ## Unpack.
            inp_1, inp_2, tar = element

            ## Generate adversarial examples with only the first audio (=inp_1).
            if attack_type == "fgm":
                x_adv = fast_gradient_method(
                    model_fn=latest_model.embedding_model,
                    x=inp_1,
                    eps=epsilon,
                    norm=np.inf,
                )
            elif attack_type == "pgd":
                x_adv = projected_gradient_descent(
                    model_fn=latest_model.embedding_model,
                    x=inp_1,
                    eps=epsilon,
                    eps_iter=epsilon * 0.1,
                    nb_iter=40,
                    norm=np.inf,
                )

            ## Predict.
            inp = tf.concat([x_adv, inp_2], axis=0)
            embeddings = latest_model.embedding_model.predict(inp)
            
            ## Calculate cosine similarity.
            cross_similarity = tf.einsum("ik,jk->ij", *tf.split(embeddings, num_or_size_splits=2, axis=0))
            similarity = tf.math.reduce_mean(cross_similarity)

            ## Append to list.
            assets["y_true"].append(tar)
            assets["y_pred"].append(similarity)
            assets["dB_x_delta"].append(np.mean(get_dB_x_delta(inp_1, x_adv)))

        ## Postprocess.
        assets["y_true"] = np.stack(assets["y_true"], axis=0)
        assets["y_pred"] = np.stack(assets["y_pred"], axis=0)
        assets["dB_x_delta"] = np.stack(assets["dB_x_delta"], axis=0)

        ## [0, 1] -> [-1, 1]
        assets["y_true"] = np.where(assets["y_true"] == 0, -1, assets["y_true"])

        return assets
