from utils.data_operation import sigmoid
import numpy as np


class Loss:
    """
    Superclass of loss object
    """
    def loss(self, y_true, y_pred):

        return NotImplementedError()

    def gradient(self, y, y_pred):

        raise NotImplementedError()


class SquareLoss(Loss):
    """
    Subclass of loss object
    Loss for numerical value, usually for regression task.
    """

    def __init__(self):
        pass

    def loss(self, y, y_pred):
        """
        loss = 0.5 * (y - y_pred)^2
        Take 1/2 in front of easy gradient
        Args:
            y: array type dataset (n_samples)
            y_pred: array type dataset (n_samples)

        Returns:
            loss: float, square loss value
        """

        loss = 0.5 * np.power((y - y_pred), 2)

        return loss

    def gradient(self, y, y_pred):
        """
        This only for gradient boosting decision tree and XGBoost, it's negative gradient which same equal to residue
        Args:
            y: array type dataset (n_samples)
            y_pred: array type dataset (n_samples)

        Returns:
            negative_gradient: float
        """

        negative_gradient = -(y - y_pred)

        return negative_gradient

    def hess(self, y):
        """
        Second order derivative of taylor expansion
        Args:
            y: array type dataset (n_samples)

        Returns:
            hess: float, square loss second order negative gradient value
        """

        hess = np.ones_like(y)

        return hess


class CrossEntropy(Loss):
    """
    Subclass of loss object
    Loss for numerical value, usually for classification task.
    """

    def __init__(self):
        pass

    def loss(self, y, y_pred):
        """
        According to sklearn, y_pred is raw prediction which have to be applied sigmoid first
        Args:
            y: array type dataset (n_samples)
            y_pred: array type dataset (n_samples)

        Returns:
            loss: float, cross entropy loss value
        """

        # avoid division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        p = sigmoid(y_pred)
        loss = - y * np.log(p) - (1 - y) * np.log(1 - p)

        return loss

    def gradient(self, y, y_pred):
        """
        This only for gradient boosting regression, since we do gradient of predict value not parameters
        According to sklearn
        https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/ensemble/_gb_losses.py#L596
        derive of raw prediction F(x) The raw predictions (i.e. values from the tree leaves) of the
        tree ensemble at iteration ``i - 1``
        should be ((y / p) - (1 - y) / (1 - p)) * (p*(1-p)), p is raw prediction which need applied sigmoid
        then negative gradient is equal to y - p
        Args:
            y: array type dataset (n_samples)
            y_pred: array type dataset (n_samples)

        Returns:
            negative_gradient: float, cross entropy negative gradient value
        """

        # this is negative gradient of F(x) in GBDT --> y_pred
        negative_gradient = y - sigmoid(y_pred)

        return negative_gradient

    def hess(self, y_pred):
        """
        Second order derivative of taylor expansion, derive of gradient which is for y - p gradient.
        Args:
            y_pred: array type dataset (n_samples)

        Returns:
            hess: float, cross entropy second order negative gradient value
        """

        p = sigmoid(y_pred)
        hess = p * (1 - p)

        return hess

    # def acc(self, y, p):
    #     return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))


# class SoftMaxLoss(Loss):
#     """sub class of loss object"""
#     # TODO: need finish softmax loss and gradient`
#     def __init__(self):
#         pass
#
#     def loss(self, y, y_pred):
#         """take 1/2 in front  for easy gradient"""
#         # return 0.5 * np.power((y - y_pred), 2)
#
#     def gradient(self, y, y_pred):
#         """negative gradient for gradient boosting decision tree"""
#         # return -(y - y_pred)
