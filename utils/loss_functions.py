from utils.data_operation import sigmoid
import numpy as np


class Loss(object):
    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()

    def acc(self, y, y_pred):
        return 0


class SquareLoss(Loss):
    """sub class of loss object"""

    def __init__(self):
        pass

    def loss(self, y, y_pred):
        """take 1/2 in front  for easy gradient"""
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        """gradient for gradient boosting decision tree and XGBoost"""
        return -(y - y_pred)

    def hess(self, y, y_pred):
        """second order derivative of taylor expansion"""
        return np.ones_like(y)


class SotfMaxLoss(Loss):
    """sub class of loss object"""
    # TODO: need finish softmax loss and gradient`
    def __init__(self):
        pass

    def loss(self, y, y_pred):
        """take 1/2 in front  for easy gradient"""
        # return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        """negative gradient for gradient boosting decision tree"""
        # return -(y - y_pred)


class CrossEntropy(Loss):
    """sub class of loss object"""

    def __init__(self):
        pass

    def loss(self, y, y_pred):
        """according to sklearn, y_pred is raw prediction have to applied sigmoid  first to y_pred"""
        # Avoid division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        p = sigmoid(y_pred)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    # def acc(self, y, p):
    #     return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, y_pred):
        """
        this only for gradient boosting regression, since we do gradient of predict value not parameters
        according to sklearn
        https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/ensemble/_gb_losses.py#L596
        derive of raw prediction F(x) The raw predictions (i.e. values from the tree leaves) of the
        tree ensemble at iteration ``i - 1``
        should be ((y / p) - (1 - y) / (1 - p)) * (p*(1-p)), p is raw prediction applied sigmoid
        == y - p
        """
        # Avoid division by zero
        # p = np.clip(p, 1e-15, 1 - 1e-15)
        # return - (y / p) + (1 - y) / (1 - p)
        # this is negative gradient of F(x) --> y_pred
        return y - sigmoid(y_pred)

    def hess(self, y, y_pred):
        """second order derivative of taylor expansion"""
        p = sigmoid(y_pred)
        return p * (1 - p)
