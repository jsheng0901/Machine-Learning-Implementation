import math

import numpy as np


class LinearRegression:
    """
    Linear Regression.
    Parameters:
    -----------
    learning_rate: float
        The step length that will be taken when following the negative gradient during training.
    n_iterations: int
        Max iteration times for gradient descent.
    regularization: object or None
        L1 regularization or L2 regularization or None.
    gradient: bool
        Use gradient descent optimization method or least square method. Only support gradient descent when apply l1/l2.
    """

    def __init__(self, n_iterations=3000, learning_rate=0.00005, regularization=None, gradient=True):

        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.gradient = gradient
        self.param = None

        if regularization is None:
            self.regularization = lambda x: 0
            self.regularization.grad = lambda x: 0
        else:
            self.regularization = regularization

    def initialize_weights(self, x):
        """
        Initial parameters value
        Args:
            x: array type dataset (n_samples, n_features)

        Returns:
            None, initial self.params -> (w: n_features, b: 0)
        """
        n_features = np.shape(x)[1]
        # Initialize parameters between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / math.sqrt(n_features)
        # Initialize parameters with [n_features, ]
        self.param = np.random.uniform(-limit, limit, (n_features,))

    def fit(self, x, y):
        """
        Fit logistic regression with gradient descent
        Args:
            x: array type dataset (n_samples, n_features)
            y: array type dataset (n_samples)

        Returns:
            None, update self.param through gradient descant
        """

        # n_samples, n_features = x.shape
        self.initialize_weights(x)
        # X = np.insert(X, 0, 1, axis=1)
        # y = np.reshape(y, (m_samples, 1))
        # self.training_errors = []
        if self.gradient is True:
            # gradient descent
            for i in range(self.n_iterations):
                # Make a new prediction, x is [n, m], parameters is [m, ] --->  [n, ]
                y_pred = x.dot(self.param)
                # calculate the loss
                # loss = np.mean(0.5 * (y_pred - y) ** 2) + self.regularization(self.w)
                # self.training_errors.append(loss)
                # calculate param gradient, [n, m] --> [m, n] * [n. ] --> [m, ]
                param_gradient = x.T.dot(y_pred - y) + self.regularization.grad(self.param)
                # update param by moving against
                self.param -= self.learning_rate * param_gradient
        else:
            # x -> [n, m] y -> [n, ]
            # least square method
            # (X_T_X)^-1 * X_T * Y
            x = np.matrix(x)
            y = np.matrix(y)
            # [m, n] * [n, m] -> [m, m]
            X_T_X = x.T.dot(x)
            # [m, m] * [m, n] -> [m, n]
            X_T_X_I_X_T = X_T_X.I.dot(x.T)
            # [m, n] * [n, ] -> [m, ]
            X_T_X_I_X_T_X_T_y = X_T_X_I_X_T.dot(y)
            self.param = X_T_X_I_X_T_X_T_y

        return

    def predict(self, x):
        """
        Predict outcome as 0 or 1
        Args:
            x: array type dataset (n_samples, n_features)

        Returns:
            y_pred: (n_samples) each sample predict value
        """
        # [n, m] * [m, ] -> [n, ]
        y_pred = x.dot(self.param)
        return y_pred
