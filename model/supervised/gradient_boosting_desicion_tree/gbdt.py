from abc import abstractmethod, ABC
from tqdm import tqdm

import numpy as np
from model.supervised.decision_tree import RegressionTree
from utils.data_operation import calculate_unique_value, sigmoid
from utils.loss_functions import CrossEntropy, SquareLoss


class GradientBoosting:
    """
    Super class of GradientBoostingClassifier and GradientBoostingRegressor.
    Uses a collection of regression trees that trains on predicting the gradient
    of the loss function.
    Parameters:
    -----------
    n_estimators: int
        The number of classification trees that are used.
    learning_rate: float
        The step length that will be taken when following the negative gradient during training.
    min_samples_split: int (smaller than stop split)
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float    (smaller than stop split)
        The minimum impurity required to split the tree further.
    max_depth: int         (bigger then stop split)
        The maximum depth of a tree.
    loss: loss object
        Loss function object to calculate negative gradient when fit residue
    """

    def __init__(self, n_estimators, learning_rate, min_samples_split, min_impurity, max_depth, loss):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.loss = loss

        # not matter classification or regression, both use regression to fit, classification have to applied logistic
        # function to transfer to probability after finish each tree calculate
        # to record each tree, we first create each initial tree object
        self.trees = []
        for i in range(self.n_estimators):
            self.trees.append(RegressionTree(min_samples_split=self.min_samples_split,
                                             min_impurity=self.min_impurity,
                                             max_depth=self.max_depth))

    @abstractmethod
    def _initialize_predict(self, y):
        """
        Function to initial first tree named as F0 output value
        If it's regression task, initial predict value is mean of y for each sample.
        If it's classification task, initial predict value is log odds of binary class
        Args:
            y: array type dataset, leaf node region y

        Returns:
            Customized initial predict value
        """
        return NotImplementedError()

    def fit(self, x, y):
        """
        Build a CART decision tree for each estimator
        Args:
            x: array type dataset (n_samples, n_features)
            y: array type dataset (n_samples)

        Returns:
            None, build multiple decision tree
        """

        # initial first y predict value according to subclass method
        y_pred = self._initialize_predict(y)
        # build each tree based on previous loss function negative gradient
        # TODO: 1. add sampling X method for each build tree, this can increase training speed and prevent overfiting
        #       2. add early stop method when build each tree, every time check if residual is smaller enough
        for i in tqdm(range(self.n_estimators)):
            # calculate negative gradient
            neg_gradient = self.loss.gradient(y, y_pred)
            self.trees[i].fit(x, neg_gradient)
            # update new y_pred with previous value + new tree predict negative gradient * learning rate
            y_pred += np.multiply(self.trees[i].predict(x), self.learning_rate)

    def _predict(self, x):
        """
        Predict each sample value from x, the output will be original regressor decision tree numerical output.
        If it's classification GBDT task, will do probability transfer in subclass.
        Args:
            x: array type dataset (n_samples, n_features)

        Returns:
            y_pred: array type dataset (n_samples), predict value
        """

        # additive each tree predict value * learning rate
        y_pred = None
        for tree in self.trees:
            each_tree_pred = np.multiply(self.learning_rate, tree.predict(x))
            y_pred = each_tree_pred if y_pred is None else y_pred + each_tree_pred

        return y_pred


class GradientBoostingRegressor(GradientBoosting, ABC):
    """
    Build gradient boosting regression tree, this is subclass of gradient boosting object.
    Here we only implement square loss as loss function. Will do other later.
    """

    def __init__(self, n_estimators=200, learning_rate=0.05, min_samples_split=2, min_var_red=1e-7, max_depth=4):
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


class GradientBoostingBinaryClassifier(GradientBoosting, ABC):
    """
    Build gradient boosting binary classifier tree, this is subclass of gradient boosting object.
    Here we only implement cross entropy loss as loss function. Will do other later.
    """

    def __init__(self, n_estimators=200, learning_rate=0.05, min_samples_split=2, min_info_gain=1e-7, max_depth=2):
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


class GradientBoostingMutilClassifier(GradientBoosting, ABC):
    """
    Build gradient boosting mutil classifier tree, this is subclass of gradient boosting object.
    Here we implement softmax loss as loss function.
    TODO: need finish mutil class classifier implementation details.
    """
    pass
