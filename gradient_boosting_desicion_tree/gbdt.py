import numpy as np
import progressbar
from decision_tree.decision_tree import RegressionTree
from utils.data_manipulation import to_categorical
from utils.loss_functions import CrossEntropy, SquareLoss
from utils.misc import bar_widgets


class GradientBoosting(object):
    """Super class of GradientBoostingClassifier and GradientBoostinRegressor.
    Uses a collection of regression trees that trains on predicting the gradient
    of the loss function.
    Parameters:
    -----------
    n_estimators: int
        The number of classification trees that are used.
    learning_rate: float
        The step length that will be taken when following the negative gradient during
        training.
    min_samples_split: int （smaller then stop split)
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float    (smaller then stop split)
        The minimum impurity required to split the tree further.
    max_depth: int         (bigger then stop split)
        The maximum depth of a tree.
    regression: boolean
        True or false depending on if we're doing regression or classification.
    """

    def __init__(self, n_estimators, learning_rate, min_samples_split,
                 min_impurity, max_depth, regression, loss):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.regression = regression
        self.loss = loss

        # create a progress bar
        self.progress_bar = progressbar.ProgressBar(widgets=bar_widgets)

        # create loss function for negative gradient
        # if self.regression:
        #     self.loss = SquareLoss()
        # else:
        #     self.loss = SotfMaxLoss()

        # not matter classification or regression, both use regression to fit, classification have to applied logistic
        # function to transfer to probability after finish each tree calculate
        # record each tree, we first create each initial tree object
        self.trees = []
        for i in range(self.n_estimators):
            self.trees.append(RegressionTree(min_samples_split=self.min_samples_split,
                                             min_impurity=self.min_impurity,
                                             max_depth=self.max_depth))

    def fit(self, X, y):
        """bulid a CART desicion tree for each estimators"""
        # fit first tree
        # self.trees[0].fit(X, y)
        # predict first tree each sampel target value
        # y_pred = self.trees[0].predict(X)

        # initial predict value is mean of y for each sample
        y_pred = np.full(np.shape(y), np.mean(y, axis=0))
        # build each tree based on previous loss function negative gradient
        # TODO: 1. add sampling X method for each build tree, this can increase training speed and prevent overfiting
        #       2. add early stop method when build each tree, every time check if residual is smaller enough
        for i in self.progress_bar(range(self.n_estimators)):
            # calculate negative gradient
            neg_gradient = self.loss.gradient(y, y_pred)
            self.trees[i].fit(X, neg_gradient)
            # update new y_pred with previous value + new tree predict negative gradient
            y_pred += np.multiply(self.trees[i].predict(X), self.learning_rate)

    def predict(self, X):
        """predict each sample value from X"""
        # additive each tree predict value * learning rate
        y_pred = np.array([])
        for tree in self.trees:
            each_tree_pred = np.multiply(self.learning_rate, tree.predict(X))
            y_pred = each_tree_pred if len(y_pred) == 0 else y_pred + each_tree_pred

        # if it's classification tree, have to transfer to probability
        if not self.regression:
            # Turn into probability distribution
            y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
            # Set label to the value that maximizes probability
            y_pred = np.argmax(y_pred, axis=1)

        return y_pred


class GradientBoostingRegressor(GradientBoosting):
    """build gradient boosting regression tree, this is sub class of gradient boosting object"""

    def __init__(self, n_estimators=200, learning_rate=0.5, min_samples_split=2,
                 min_var_red=1e-7, max_depth=4):
        super(GradientBoostingRegressor, self).__init__(n_estimators=n_estimators,
                                                        learning_rate=learning_rate,
                                                        min_samples_split=min_samples_split,
                                                        min_impurity=min_var_red,
                                                        max_depth=max_depth,
                                                        regression=True,
                                                        loss=SquareLoss())


class GradientBoostingClassifier(GradientBoosting):
    """build gradient boosting classifier tree, this is sub class of gradient boosting object"""

    def __init__(self, n_estimators=200, learning_rate=.5, min_samples_split=2,
                 min_info_gain=1e-7, max_depth=2):
        super(GradientBoostingClassifier, self).__init__(n_estimators=n_estimators,
                                                         learning_rate=learning_rate,
                                                         min_samples_split=min_samples_split,
                                                         min_impurity=min_info_gain,
                                                         max_depth=max_depth,
                                                         regression=False,
                                                         loss=CrossEntropy())

    def fit(self, X, y):
        y = to_categorical(y)
        super(GradientBoostingClassifier, self).fit(X, y)
