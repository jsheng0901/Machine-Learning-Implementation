from abc import ABC

import numpy as np
from tqdm import tqdm

from model.supervised.decision_tree import DecisionTree
from model.supervised.gradient_boosting_desicion_tree import GradientBoosting
from utils.data_operation import calculate_unique_value, sigmoid
from utils.loss_functions import CrossEntropy, SquareLoss


class XGBoostRegressionTree(DecisionTree, ABC):
    """
    Regression tree for XGBoost, this is subclass of decision tree, this is one tree in XGBoost.
    We use CART decision tree as base weaker learner for XGB.
    Parameters:
    -----------
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    loss: loss object
        Loss function object to calculate first and second order gradient for tylor gain when split node.
    """

    def __init__(self, min_samples_split, min_impurity, max_depth, loss):
        super().__init__(min_samples_split, min_impurity, max_depth)
        self.loss = loss

    def split(self, y):
        """
        y contains y_true in left half of the middle column and
        y_pred in the right half. Split and return the two matrices
        Args:
            y: array type dataset, contains both true and predict value

        Returns:
            y: array type dataset, contains only true value
            y_pred: array type dataset, contains only predict value
        """

        col = int(np.shape(y)[1] / 2)
        y, y_pred = y[:, :col], y[:, col:]

        return y, y_pred

    def gain(self, y, y_pred):
        """
        Calculate information gain by xgboost second order loss function taylor expansion
        single tree leaf node loss will be: -1/2 ( (Gi ** 2) / (Hi + lambda) ) + gama * T -> (total leaves)
        Args:
            y: array type dataset, contains y true value
            y_pred: array type dataset, contains y predict value to (t - 1) tree

        Returns:
            value: float, information gain at this leaf
        """

        # TODO: here we remove the lambda and gama, will added in future
        # sum up each sample if drop in this leaf
        nominator = np.power((self.loss.gradient(y, y_pred)).sum(), 2)
        denominator = self.loss.hess(y, y_pred).sum()
        value = 0.5 * (nominator / denominator)

        return value

    def _impurity_calculation(self, y, y1, y2):
        """
        XGB information gain calculate method, this is used for choose best feature and best split point when build
        tree, find maximum gain feature and split point.
        Args:
            y: array type dataset, original y value before split, contains both true and predict value at t - 1 tree
            y1: array type dataset, left true branch y value, contains both true and predict value at t - 1 tree
            y2: array type dataset, right false branch, contains both true and predict value at t - 1 tree

        Returns:
            gain_by_taylor: float, total XGB defined information gain after this split.
        """

        # split y into y_true value and y_pred value
        y, y_pred = self.split(y)
        y1, y1_pred = self.split(y1)
        y2, y2_pred = self.split(y2)

        # calculate each branch gain and calculate difference when we do the split
        left_gain = self.gain(y1, y1_pred)
        right_gain = self.gain(y2, y2_pred)
        parent_gain = self.gain(y, y_pred)

        gain_by_taylor = left_gain + right_gain - parent_gain

        return gain_by_taylor

    def _leaf_value_calculation(self, y):
        """
        This is function for calculate approximate leaf node value (weight): -Gi / (Hi + lambda)
        Args:
            y: array type dataset, leaf node region y, contains y true and y predict

        Returns:
            approximation_weight: float, final predict leaf value, base on XGB defined leaf weight
        """

        # y split into y, y_pred
        y, y_pred = self.split(y)

        # TODO: here we remove lambda, will added in future
        # Gi, sum up all sample at this leaf
        gradient = np.sum(self.loss.gradient(y, y_pred), axis=0)

        # Hi, sum up all sample at this leaf
        hessian = np.sum(self.loss.hess(y, y_pred), axis=0)

        approximation_weight = -gradient / hessian

        return approximation_weight


class XGBoost(GradientBoosting, ABC):
    """
    The XGBoost object, this is whole collection of XGBoost tree build object.
    Parameters:
    -----------
    n_estimators: int
        The number of classification trees that are used.
    learning_rate: float
        The step length that will be taken when following the negative gradient during training.
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    loss: loss object
        Loss function object to calculate taylor expansion first order and second order derivative
    """

    def __init__(self, n_estimators, learning_rate, min_samples_split, min_impurity, max_depth, loss):
        super().__init__(n_estimators, learning_rate, min_samples_split, min_impurity, max_depth, loss)

        # Initialize regression trees, same as GBDT, no matter which task, always regression tree
        # Need to override in subclass here since parent class has same variable.
        self.trees = []
        for _ in range(n_estimators):
            tree = XGBoostRegressionTree(
                min_samples_split=self.min_samples_split,
                min_impurity=min_impurity,
                max_depth=self.max_depth,
                loss=self.loss)

            self.trees.append(tree)

    def fit(self, x, y):
        """
        Build a xgboost regression tree for each estimator
        Args:
            x: array type dataset (n_samples, n_features)
            y: array type dataset (n_samples)

        Returns:
            None, build multiple XGB regression decision tree
        """

        # reshape y into [n_sample, 1] for later concat
        n_sample = x.shape[0]
        y = np.reshape(y, (n_sample, -1))
        # initial first y predict value according to subclass method, same as GBDT.
        y_pred = self._initialize_predict(y)
        # reshape y_pred as same dim as y
        y_pred = np.reshape(y_pred, (n_sample, -1))
        for i in tqdm(range(self.n_estimators)):
            # concat y true value and previous tree y pred value horizontal, to calculate Gi, Hi in split and leaf value
            y_and_pred = np.concatenate((y, y_pred), axis=1)
            self.trees[i].fit(x, y_and_pred)
            update_pred = self.trees[i].predict(x)
            # update new y_pred with previous value + new tree predict * learning rate
            y_pred += np.multiply(update_pred, self.learning_rate)


class XGBoostingRegressor(XGBoost, ABC):
    """
    Build XGBoost regression tree, this is subclass of XGBoost object.
    Here we only implement square loss as loss function. Will do other later.
    """

    def __init__(self, n_estimators=200, learning_rate=0.5, min_samples_split=2, min_var_red=1e-7, max_depth=4):
        super().__init__(n_estimators=n_estimators,
                         learning_rate=learning_rate,
                         min_samples_split=min_samples_split,
                         min_impurity=min_var_red,
                         max_depth=max_depth,
                         loss=SquareLoss())

    def _initialize_predict(self, y):
        """
        Function to initial first tree named as F0 output value
        For regression task, initial predict value is mean of y for each sample.
        Args:
            y: array type dataset, leaf node region y

        Returns:
            initial_y: array type dataset same as input y
        """

        initial_y = np.full(np.shape(y), np.mean(y, axis=0))
        return initial_y

    def predict(self, x):
        """
        Predict each sample value from x. For regression task just use inside predict method from super class
        Args:
            x: array type dataset (n_samples, n_features)

        Returns:
            y_pred: array type dataset (n_samples), final predict value
        """

        y_pred = self._predict(x)

        return y_pred


class XGBoostingBinaryClassifier(XGBoost, ABC):
    """
    Build XGBoost binary classifier tree, this is subclass of XGBoost object.
    Here we only implement cross entropy loss as loss function. Will do other later.
    """

    def __init__(self, n_estimators=200, learning_rate=.5, min_samples_split=2, min_info_gain=1e-7, max_depth=2):
        super().__init__(n_estimators=n_estimators,
                         learning_rate=learning_rate,
                         min_samples_split=min_samples_split,
                         min_impurity=min_info_gain,
                         max_depth=max_depth,
                         loss=CrossEntropy())

    def _initialize_predict(self, y):
        """
        Function to initial first tree named as F0 output value
        For classification task, initial predict value is log odds of binary class
        Args:
            y: array type dataset, leaf node region y

        Returns:
            initial_y: array type dataset same as input y
        """

        # get all labels frequency
        unique_y_freq = calculate_unique_value(y)
        # since it's binary, only pos and neg labels, named as 0, 1 labels
        pos = unique_y_freq[0]
        neg = unique_y_freq[1]

        # calculate log odds: log(p / (1 - p))
        log_odds = np.log(pos / neg)
        # change to same shape as input y
        initial_y = np.full(np.shape(y), log_odds)
        return initial_y

    def predict(self, x):
        """
        Predict each sample value from x. For classification task use inside predict method from super class
        then apply sigmoid to probability and change to label as output.
        Args:
            x: array type dataset (n_samples, n_features)

        Returns:
            y_pred: array type dataset (n_samples), final predict value
        """

        _y_pred = self._predict(x)

        # turn into probability distribution, apply sigmoid
        y_pred = sigmoid(_y_pred)
        # set label to the value that threshold as 0.5
        y_pred = [1 if p >= 0.5 else 0 for p in y_pred]

        return y_pred
