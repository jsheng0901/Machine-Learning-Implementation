import numpy as np
import progressbar
from utils.misc import bar_widgets
from decision_tree.decision_tree import DecisionTree
from utils.data_manipulation import to_categorical
from utils.loss_functions import CrossEntropy, SquareLoss


class XGBoostRegressionTree(DecisionTree):
    """
    Regression tree for XGBoost, this is sub class of decision tree, this is just one tree in XGBoost
    """

    def split(self, y):
        """
        y contains y_true in left half of the middle column and
        y_pred in the right half. Split and return the two matrices
        """

        col = int(np.shape(y)[1] / 2)
        y, y_pred = y[:, :col], y[:, col:]
        return y, y_pred

    def gain(self, y, y_pred):
        """
        calculate information gain by xgboost second order loss function taylor expansion
        single tree leaf node loss will be: -1/2 ( (Gi ** 2) / (Hi + lambda) ) + gama * 1

        y: array like contains y_true value
        y_pred: array like contains y predict value to (t - 1) tree
        """
        # TODO: here we remove the lambda and (gama * 1 (since this is one leaf) ), will added in future
        nominator = np.power((self.loss.gradient(y, y_pred)).sum(), 2)  # sum up each sample if drop in this branch
        denominator = self.loss.hess(y, y_pred).sum()
        return 0.5 * (nominator / denominator)

    def information_gain_by_taylor(self, y, y1, y2):
        """
        xgboost information gain calculate method, this is used for choose best feature and best split point when build
        tree, find maximum gain feature and split point

        y: np.array, original y value before split, contains both true value and predict value so far to (t - 1 ) tree
        y1: np.array, left true branch y value, contains both true value and predict value so far to (t - 1 ) tree
        y2: np.array, right false branch, contains both true value and predict value so far to (t - 1 ) tree
        """
        # split y into y_true value and y_pred value
        y, y_pred = self.split(y)
        y1, y1_pred = self.split(y1)
        y2, y2_pred = self.split(y2)
        # calculare each branch gain and calculate difference if we do the split
        left_gain = self.gain(y1, y1_pred)
        right_gain = self.gain(y2, y2_pred)
        parent_gain = self.gain(y, y_pred)
        return left_gain + right_gain - parent_gain

    def approximate_update(self, y):
        """
        this is function for calculate approximate leaf node value (weight): -Gi / (Hi + lambda)

        y: array like, contains y true and y predict
        """
        # y split into y, y_pred
        y, y_pred = self.split(y)
        # TODO: here we remove lambda, will added in future
        gradient = np.sum(self.loss.gradient(y, y_pred), axis=0)  # gi, sum up all sample
        hessian = np.sum(self.loss.hess(y, y_pred), axis=0)  # hi, sum up all sample
        approximation_weight = -gradient / hessian
        return approximation_weight

    def fit(self, X, y):
        """fit xgboost will call fit method in super class"""
        self._impurity_calculation = self.gain_by_taylor
        self._leaf_value_calculation = self.approximate_update
        super(XGBoostRegressionTree, self).fit(X, y)


class XGBoost(object):
    """
    The XGBoost object, this is whole collection of XGBoost tree build object.
    Parameters:
    -----------
    n_estimators: int
        The number of classification trees that are used.
    learning_rate: float
        The step length that will be taken when following the negative gradient during
        training.
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    regression: boolean
        True or false depending on if we're doing regression or classification.
     loss: loss object
        Loss function object to calculate taylor expansion first order and second order derivative
    """

    def __init__(self, n_estimators, learning_rate, min_samples_split, min_impurity, max_depth, regression, loss):
        self.n_estimators = n_estimators  # Number of trees
        self.learning_rate = learning_rate  # Step size for weight update
        self.min_samples_split = min_samples_split  # The minimum n of sampels to justify split
        self.min_impurity = min_impurity  # Minimum variance reduction (regression) to continue
        self.max_depth = max_depth  # Maximum depth for tree
        self.regression = regression
        self.loss = loss

        # create a progress bar
        self.bar = progressbar.ProgressBar(widgets=bar_widgets)
        # Log loss for classification
        # self.loss = LeastSquaresLoss()

        # Initialize regression trees
        self.trees = []
        for _ in range(n_estimators):
            tree = XGBoostRegressionTree(
                min_samples_split=self.min_samples_split,
                min_impurity=min_impurity,
                max_depth=self.max_depth,
                loss=self.loss)

            self.trees.append(tree)

    def fit(self, X, y):
        """bulid a xgboost regression tree for each estimators"""
        # y = to_categorical(y)
        # m = X.shape[0]
        # y = np.reshape(y, (m, -1))
        # TODO: initial value should use mean of y of all zero should be research
        y_pred = np.zeros(np.shape(y))
        # initial predict value is mean of y for each sample
        # y_pred = np.full(np.shape(y), np.mean(y, axis=0))
        for i in self.bar(range(self.n_estimators)):
            # concat y true value and previous tree y pred value horizontal, to calculate Gi, Hi in split and leaf value
            y_and_pred = np.concatenate((y, y_pred), axis=1)
            self.trees[i].fit(X, y_and_pred)
            update_pred = self.trees[i].predict(X)
            # update_pred = np.reshape(update_pred, (m, -1))
            # update new y_pred with previous value + new tree predict * learning rate
            y_pred += np.multiply(update_pred, self.learning_rate)

    def predict(self, X):
        """predict each sample value from X"""
        # consistent with train, the initial value is all zero
        y_pred = None
        # y_pred = np.array([])
        m = X.shape[0]
        # Make predictions
        for tree in self.trees:
            # Estimate gradient and update prediction
            update_pred = tree.predict(X)
            update_pred = np.reshape(update_pred, (m, -1))
            if y_pred is None:
                # consistent with train initial step
                y_pred = np.zeros_like(update_pred)
            y_pred += np.multiply(update_pred, self.learning_rate)

        # if it's classification tree, have to transfer to probability
        if not self.regression:
            # Turn into probability distribution (Softmax)
            y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
            # Set label to the value that maximizes probability
            y_pred = np.argmax(y_pred, axis=1)

        return y_pred


class XGBoostRegressor(XGBoost):
    """build XGBoost regression tree, this is sub class of XGBoost object"""

    def __init__(self, n_estimators=200, learning_rate=0.5, min_samples_split=2,
                 min_var_red=1e-7, max_depth=4):
        super(XGBoostRegressor, self).__init__(n_estimators=n_estimators,
                                               learning_rate=learning_rate,
                                               min_samples_split=min_samples_split,
                                               min_impurity=min_var_red,
                                               max_depth=max_depth,
                                               regression=True,
                                               loss=SquareLoss())


class XGBoostClassifier(XGBoost):
    """build XGBoost classifier tree, this is sub class of XGBoost object"""

    def __init__(self, n_estimators=200, learning_rate=.5, min_samples_split=2,
                 min_info_gain=1e-7, max_depth=2):
        super(XGBoostClassifier, self).__init__(n_estimators=n_estimators,
                                                learning_rate=learning_rate,
                                                min_samples_split=min_samples_split,
                                                min_impurity=min_info_gain,
                                                max_depth=max_depth,
                                                regression=False,
                                                loss=CrossEntropy())

    def fit(self, X, y):
        y = to_categorical(y)
        super(XGBoostClassifier, self).fit(X, y)
