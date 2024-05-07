import numpy as np


class Loss:
    """
    Superclass of loss object
    """

    def loss(self, y_true, y_pred):
        return NotImplementedError()

    def gradient(self, y, y_pred):
        raise NotImplementedError()


class MeanSquareError(Loss):
    """
    Mean Square Error, loss for numerical value, usually for regression task.
    """

    def loss(self, y_true, y_pred):
        """
        This function calculates mse = mean((y_true - y_pred)^2)
        Args:
            y_true: array type dataset (n_samples), list of real numbers, true values
            y_pred: array type dataset (n_samples), list of real numbers, predicted values

        Returns:
            loss: float, mean square error value
        """
        # initialize error at 0
        error = 0
        # loop over all samples in the true and predicted list
        for yt, yp in zip(y_true, y_pred):
            # calculate squared error and add to error
            error += np.square(yt - yp)

        # return mean error
        return error / len(y_true)


class MeanAbsoluteError(Loss):
    """
    Mean Absolute Error, loss for numerical value, usually for regression task.
    """

    def loss(self, y_true, y_pred):
        """
        This function calculates mae = mean(abs(y_true - y_pred))
        Args:
            y_true: array type dataset (n_samples), list of real numbers, true values
            y_pred: array type dataset (n_samples), list of real numbers, predicted values

        Returns:
            loss: float, mean absolute error value
        """
        # initialize error at 0
        error = 0
        # loop over all samples in the true and predicted list
        for yt, yp in zip(y_true, y_pred):
            # calculate absolute error and add to error
            error += np.abs(yt - yp)

        # return mean error
        return error / len(y_true)


class SqrtMeanSquareError(MeanSquareError):
    """
    Sqrt Mean Square Error, loss for numerical value, usually for regression task.
    """

    def loss(self, y_true, y_pred):
        """
        This function calculates rmse = sqrt(mean((y_true - y_pred)^2))
        Args:
            y_true: array type dataset (n_samples), list of real numbers, true values
            y_pred: array type dataset (n_samples), list of real numbers, predicted values

        Returns:
            loss: float, sqrt mean square error value
        """
        # calculate mse use parent class method
        mse = super().loss(y_true, y_pred)

        # return sqrt error
        return np.sqrt(mse)


class MeanAbsolutePercentageError(Loss):
    """
    Mean Absolute Percentage Error, loss for numerical value, usually for regression task.
    """

    def loss(self, y_true, y_pred):
        """
        This function calculates mape = mean(abs(y_true - y_pred) / y_true)。
        Args:
            y_true: array type dataset (n_samples), list of real numbers, true values
            y_pred: array type dataset (n_samples), list of real numbers, predicted values

        Returns:
            loss: float, mean absolute percentage error value
        """
        # initialize error at 0
        error = 0
        # loop over all samples in true and predicted list
        for yt, yp in zip(y_true, y_pred):
            # calculate percentage error and add to error
            error += np.abs(yt - yp) / yt

        # return mean percentage error
        return error / len(y_true)


class HuberLoss(Loss):
    """
    Huber Los, loss for numerical value, usually for regression task.
    """

    def loss(self, y_true, y_pred, delta=1):
        """
        This function calculates
        huber loss:
            1/2 * (y_true - y_pred)^2  ---> abs(y_true - y_pred) < delta;
            delta * abs(y_true - y_pred) - 1/2 * delta^2   ---> otherwise;
        Args:
            delta: float, parameter to control which exactly loss function to use
            y_true: array type dataset (n_samples), list of real numbers, true values
            y_pred: array type dataset (n_samples), list of real numbers, predicted values

        Returns:
            loss: float, huber loss value
        """
        error = y_true - y_pred
        # calculate abs error
        abs_error = np.abs(error)
        # if smaller than delta this loss
        squared_loss = 0.5 * np.square(error)
        # otherwise this loss
        absolute_loss = delta * (abs_error - 0.5 * delta)
        # calculate which loss to use
        return np.where(abs_error <= delta, squared_loss, absolute_loss)


class R2(Loss):
    """
    Coefficient Of Determination, loss for numerical value, usually for linear regression task.
    """

    def loss(self, y_true, y_pred):
        """
        This function calculates r2 = 1 - (sum(y_true - y_pred)^2 / sum(y_true - y_true_mean)^2)。
        Args:
            y_true: array type dataset (n_samples), list of real numbers, true values
            y_pred: array type dataset (n_samples), list of real numbers, predicted values

        Returns:
            loss: float, r2 value
        """
        # calculate the mean value of true values
        mean_true_value = np.mean(y_true)

        # initialize numerator with 0
        numerator = 0
        # initialize denominator with 0
        denominator = 0

        # loop over all true and predicted values
        for yt, yp in zip(y_true, y_pred):
            # update numerator
            numerator += (yt - yp) ** 2
            # update denominator
            denominator += (yt - mean_true_value) ** 2

        # calculate the ratio
        ratio = numerator / denominator

        # return 1 - ratio
        return 1 - ratio


class LogLoss(Loss):
    """
    Logistics loss, loss for numerical value, usually for binary classification task.
    """

    def loss(self, y_true, y_pred):
        """
        This function calculates log loss = -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
        Args:
            y_true: array type dataset (n_samples), list of real numbers, true values
            y_pred: array type dataset (n_samples), list of real numbers, predicted values

        Returns:
            loss: float, log loss value
        """
        # 避免对数运算错误
        epsilon = 1e-15
        y_pred = np.maximum(np.minimum(y_pred, 1 - epsilon), epsilon)
        log_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        return log_loss


class CrossEntropyLoss(Loss):
    """
    Cross Entropy loss, loss for numerical value, usually for multi classification task.
    """

    def loss(self, y_true, y_pred):
        """
        This function calculates cross entropy loss = -mean(sum_sample(sum_class((y_true * log(y_pred)))))
        Args:
            y_true: array type dataset (n_samples), list of real numbers, true values
            y_pred: array type dataset (n_samples), list of real numbers, predicted values

        Returns:
            loss: float, cross entropy loss value
        """
        # 避免对数运算错误
        epsilon = 1e-15
        y_pred = np.maximum(np.minimum(y_pred, 1 - epsilon), epsilon)
        # 注意这里用 numpy 进行了矩阵运算，里面包含对每个样本的每个类别的乘积求和，再求平均数
        cross_entropy_loss = -np.mean(y_true * np.log(y_pred))

        return cross_entropy_loss


class HingeLoss(Loss):
    """
    Hinge loss, loss for numerical value, usually for binary classification task and svm model.
    """

    def loss(self, y_true, y_pred):
        """
        This function calculates hinge loss = mean(max(0, 1 - y_true * y_pred))
        Args:
            y_true: array type dataset (n_samples), list of real numbers, true values
            y_pred: array type dataset (n_samples), list of real numbers, predicted values

        Returns:
            loss: float, hinge loss value
        """

        hinge_loss = np.mean(np.maximum(0, 1 - y_true * y_pred))

        return hinge_loss


class KLDivergenceLoss(Loss):
    """
    KL divergence loss, probability diff between distributions, usually for predict probability distribution task.
    """
    def loss(self, y_true, y_pred):
        """
        This function calculates KL divergence loss = sum(y_true * log(y_true / y_pred))
        Args:
            y_true: array type dataset (n_samples), list of real numbers, true values
            y_pred: array type dataset (n_samples), list of real numbers, predicted values

        Returns:
            loss: float, KL divergence loss value
        """

        return np.sum(y_true * np.log(y_true / y_pred))
