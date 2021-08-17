import numpy as np
import math
from utils.data_operation import sigmoid


class LogisticRegression():
    """ Logistic Regression classifier.
    Parameters:
    -----------
    learning_rate: float
        The step length that will be taken when following the negative gradient during
        training.
    """
    def __init__(self, learning_rate=.1):
        self.param = None
        self.learning_rate = learning_rate
        self.sigmoid = sigmoid()

    def initialize_parameters(self, X):
        """initial parameters value"""
        n_features = np.shape(X)[1]
        # Initialize parameters between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / math.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, (n_features,))
        # TODO: Initialize intercept b with 0
        # b = 0
        # self.param = np.insert(param, 0, b, axis=0)

    def fit(self, X, y, n_iterations=4000):
        """fit logistic regression with gradient descent"""
        self.initialize_parameters(X)
        # Tune parameters for n iterations
        for i in range(n_iterations):
            # Make a new prediction, X is [n, m], parameters is [m, ] --->  [n, ]
            y_pred = self.sigmoid(X.dot(self.param))
            # Move against the gradient of the loss function with
            # respect to the parameters to minimize the loss
            param_gradient = X.T.dot(y_pred - y)    # [n, m] --> [m, n] * [n. ] --> [m, ]
            self.param -= self.learning_rate * param_gradient

    def predict(self, X):
        """predict outcome as 0 or 1"""

        y_pred = np.round(self.sigmoid(X.dot(self.param))).astype(int)
        return y_pred

