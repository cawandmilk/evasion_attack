import numpy as np

from sklearn.metrics import roc_curve, auc


class EvaluateIdentificationModel():

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray):
        """ Calculate (top-1) accuracy.
        """
        assert y_true.shape[0] == y_pred.shape[0]

        result = np.where(y_true == np.argmax(y_pred, axis=-1), 1, 0)

        return result


    @staticmethod
    def cmc(y_true: np.ndarray, y_pred: np.ndarray):
        """ Calculate Cumulative Match Characteristic (CMC) curve, 
            i.e. top-1 ~ top-{len(y_true)} accuracy.
        """
        assert y_true.shape[0] == y_pred.shape[0]

        argsort = np.argsort(y_pred)[:, ::-1]
        result = [np.cumsum(np.where(argsort[i] == y_true[i], 1, 0)) for i in range(y_true.shape[0])]
        result = np.mean(np.stack(result, axis=0), axis=0)

        return result


class EvaluateVerificationModel():

    @staticmethod
    def eer(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1):
        """ Calculate equal erro rate (EER).

            It was referenced from:
             - https://github.com/clovaai/voxceleb_trainer/blob/master/tuneThreshold.py
        """
        ## All fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array).
        fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label = positive_label)
        fnr = 1 - tpr

        ## Theoretically eer from fpr and eer from fnr should be 
        ## identical but they can be slightly differ in reality.
        eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

        ## Return the mean of eer from fpr and from fnr.
        result = (eer_1 + eer_2) / 2 * 100
        
        return result


    @staticmethod
    def auroc(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1):
        """ Compute Area Under the ROC Curve (AUROC).
        """
        ## All fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array).
        fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label = positive_label)
        result = auc(fpr, tpr)

        return result


    @staticmethod
    def min_dcf(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1, p_target: float = 0.05, c_miss: float = 1., c_fa: float = 1.):
        """ Minimum Detection Cost Function (minDCF) implementation.

            It was referenced from:
             - https://github.com/clovaai/voxceleb_trainer/blob/master/tuneThreshold.py
             - https://arxiv.org/pdf/2012.06867.pdf
             - https://www.nist.gov/system/files/documents/2018/08/17/sre18_eval_plan_2018-05-31_v6.pdf
        """
        ## All fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array).
        fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label = positive_label)
        fnr = 1 - tpr

        ## Numpy like.
        c_det = c_miss * fnr * p_target + c_fa * fpr * (1. - p_target)
        c_default = min(c_miss * p_target, c_fa * (1. - p_target))

        result = np.min(c_det) / c_default

        return result

