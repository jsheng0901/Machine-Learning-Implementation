import numpy as np
import math
from utils.data_operation import sigmoid


class LogisticRegression:
    """
    Logistic Regression classifier.
    Parameters:
    -----------
    learning_rate: float
        The step length that will be taken when following the negative gradient during
        training.
    n_iterations: int
        Max iteration times for gradient descent
    """
    def __init__(self, learning_rate=.1, n_iterations=4000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.param = None

    def initialize_parameters(self, x):
        """
        Initial parameters value
        Args:
            x: array type dataset (n_samples, n_features)

        Returns:
            None, initial self.params -> (n_features)
        """

        n_features = np.shape(x)[1]
        # Initialize parameters between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / math.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, (n_features,))
        # TODO: Initialize intercept b with 0
        # b = 0
        # self.param = np.insert(param, 0, b, axis=0)

    def fit(self, x, y):
        """
        Fit logistic regression with gradient descent
        Args:
            x: array type dataset (n_samples, n_features)
            y: array type dataset (n_samples)

        Returns:
            None, update self.param through gradient descant
        """

        self.initialize_parameters(x)
        # Tune parameters for n iterations
        for i in range(self.n_iterations):
            # Make a new prediction, X is [n, m], parameters is [m, ] --->  [n, ], use sigmoid trans to (0, 1) range
            y_pred = sigmoid(x.dot(self.param))
            # Move against the gradient of the loss function with
            # respect to the parameters to minimize the loss
            param_gradient = x.T.dot(y_pred - y)    # [n, m] --> [m, n] * [n. ] --> [m, ]
            self.param -= self.learning_rate * param_gradient

    def predict(self, x):
        """
        Predict outcome as 0 or 1
        Args:
            x: array type dataset (n_samples, n_features)

        Returns:
            y_pred: (n_samples) each sample predict label as 0 or 1
        """

        y_pred = np.round(sigmoid(x.dot(self.param))).astype(int)
        return y_pred

