import numpy as np
from tqdm import tqdm

from decision_tree import RegressionTree, ClassificationCARTTree


class RandomForest:
    """
    Random Forest parent class. Uses a collection of trees that
    trains on random subsets of the data using a random subsets of the features.

    Parameters:
    -----------
    n_estimators: int
        The number of trees that are used.
    max_features: str or None
        The maximum number of features that the trees are allowed to use.
    min_samples_split: int  (smaller than stop split)
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float     (smaller than stop split)
        The minimum impurity required to split the tree further.
    max_depth: int          (bigger then stop split)
        The maximum depth of a tree.
    tree: single tree object
        Individual tree object to train
    """

    def __init__(self, n_estimators, max_features, min_samples_split, min_impurity, max_depth, tree):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.tree = tree

        self.trees = []
        # record each tree used features
        self.tree_features = []
        # build forest base on input single tree object
        for _ in range(self.n_estimators):
            self.trees.append(tree)

    def get_bootstrap_data(self, x, y):
        """
        Generate each tree train and label subset by bootstrap.
        Each tree training subset can have duplicate row due to replaced bootstrap
        Args:
            x: array type dataset (n_samples, n_features)
            y: array type dataset (n_samples)

        Returns:
            train_sets: array type dataset (n_estimators, ) build multiple decision tree
        """
        # get n_estimators subset by bootstrap
        n_samples = x.shape[0]
        # reshape [n_samples, ] -> [n_samples, 1]
        y = y.reshape(n_samples, 1)

        # combine x and y, [n_samples, n_features] + [n_samples, 1] -> [n_samples, n_features + 1]
        x_y = np.hstack((x, y))
        # random in-place shuffle whole train and label dataset
        np.random.shuffle(x_y)

        train_sets = []
        for _ in range(self.n_estimators):
            # row index can be duplicate, since we do replace bootstrap
            idx = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_x_y = x_y[idx, :]
            # get training subset
            bootstrap_x = bootstrap_x_y[:, :-1]
            # get label subset
            bootstrap_y = bootstrap_x_y[:, -1]
            train_sets.append([bootstrap_x, bootstrap_y])

        return train_sets

    def fit(self, x, y):
        """
        Build a CART decision tree for each estimator
        Args:
            x: array type dataset (n_samples, n_features)
            y: array type dataset (n_samples)

        Returns:
            None, build multiple decision tree
        """

        # every tree use random data set (bootstrap) and random feature
        # get random data subset
        sub_sets = self.get_bootstrap_data(x, y)

        # get number of total features
        n_features = x.shape[1]
        # if max features not provide, will use sqrt of total features
        if self.max_features is 'sqrt':
            self.max_features = int(np.sqrt(n_features))
        elif self.max_features is None:
            self.max_features = n_features

        for i in tqdm(range(self.n_estimators)):
            # get random train and label subset
            sub_x, sub_y = sub_sets[i]
            # get random features according to max features
            idx = np.random.choice(n_features, self.max_features, replace=True)
            # get train subset base on random choice features
            sub_x = sub_x[:, idx]
            # train each tree
            self.trees[i].fit(sub_x, sub_y)
            # record used features
            self.tree_features.append(idx)

    def _predict(self, x):
        """
        Predict each sample value from x, the output will be original each tree output for each sample.
        If it's classification task, will do majority vote in subclass.
        If it's regression task, will do mean in subclass
        Args:
            x: array type dataset (n_samples, n_features)

        Returns:
            y_pred: array type dataset (n_samples, n_estimators), predict value for each single tree
        """
        y_preds = []
        for i in range(self.n_estimators):
            # get tree used train features
            idx = self.tree_features[i]
            # get predict subset according to train features
            sub_x = x[:, idx]
            # get predict value for each sample
            y_pred = self.trees[i].predict(sub_x)
            # append into final result
            y_preds.append(y_pred)

        # transfer shape (n_estimators, n_samples) -> (n_samples, n_estimators)
        y_preds = np.array(y_preds).T

        return y_preds


class RandomForestRegressor(RandomForest):
    """
    Build random forest regression tree, this is subclass of random forest object.
    We will apply mean across all estimator for each sample.
    """

    def __init__(self, n_estimators=200, max_features='sqrt', min_samples_split=2, min_impurity=0,
                 max_depth=float("inf")):
        # build regression tree as single tree object for random forest
        self.tree = RegressionTree(min_samples_split=min_samples_split,
                                   min_impurity=min_impurity,
                                   max_depth=max_depth)
        super().__init__(n_estimators=n_estimators,
                         max_features=max_features,
                         min_samples_split=min_samples_split,
                         min_impurity=min_impurity,
                         max_depth=max_depth,
                         tree=self.tree)

    def predict(self, x):
        """
        Predict each sample value from x. For regression task will use mean across all estimator.
        Args:
            x: array type dataset (n_samples, n_features)

        Returns:
            y_pred: array type dataset (n_samples, ), final predict value
        """

        # get predict value for each sample and each tree -> (n_samples, n_estimators)
        y_preds = self._predict(x)

        # (n_samples, n_estimators) -> (n_samples, )
        y_pred = np.mean(y_preds, axis=1)

        return y_pred


class RandomForestClassifier(RandomForest):
    """
    Build random forest classifier tree, this is subclass of random forest object.
    We will apply majority vote across all estimator for each sample.
    """

    def __init__(self, n_estimators=200, max_features=None, min_samples_split=2, min_impurity=0,
                 max_depth=float("inf")):
        # build classification cart tree as single tree object for random forest
        self.tree = ClassificationCARTTree(min_samples_split=min_samples_split,
                                           min_impurity=min_impurity,
                                           max_depth=max_depth)
        super().__init__(n_estimators=n_estimators,
                         max_features=max_features,
                         min_samples_split=min_samples_split,
                         min_impurity=min_impurity,
                         max_depth=max_depth,
                         tree=self.tree)

    def predict(self, x):
        """
        Predict each sample value from x. For classification task will use majority vote across all estimator.
        Args:
            x: array type dataset (n_samples, n_features)

        Returns:
            y_pred: array type dataset (n_samples, ), final predict value
        """

        # get predict value for each sample and each tree -> (n_samples, n_estimators)
        y_preds = self._predict(x)

        y_pred = []
        for y_p in y_preds:
            # majority vote for all trees
            # np.bincount() get freq of each index, np.argmax() get max value index
            # (1, n_estimators) -> (1, )
            y_pred.append(np.bincount(y_p.astype('int')).argmax())

        return y_pred
