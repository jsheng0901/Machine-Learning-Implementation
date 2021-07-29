import numpy as np
from utils.data_manipulation import divide_matrix_on_feature
from utils.data_operation import calculate_variance, calculate_entropy, gini

class DecisionNode():
    """Class that represents a decision node or leaf in the decision tree
    Parameters:
    -----------
    feature_i: int
        Feature index which we want to use as the threshold measure.
    threshold: float
        The value that we will compare feature values at feature_i against to
        determine the prediction.
    value: float
        The class prediction if classification tree, or float value if regression tree.
    true_branch: DecisionNode
        Next decision node for samples where features value met the threshold.
    false_branch: DecisionNode
        Next decision node for samples where features value did not meet the threshold.
    """

    def __init__(self, feature_i=None, threshold=None,
                 value=None, true_branch=None, false_branch=None):
        self.feature_i = feature_i  # Index for the feature that is tested
        self.threshold = threshold  # Threshold value for feature
        self.value = value  # Value if the node is a leaf in the tree
        self.true_branch = true_branch  # 'Left' subtree
        self.false_branch = false_branch  # 'Right' subtree


# This is Super class of RegressionTree and ClassificationTree
class DecisionTree(object):
    """Super class of RegressionTree and ClassificationTree.
    Parameters:
    -----------
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    loss: function
        Loss function that is used for Gradient Boosting models to calculate impurity.
    """

    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=float("inf"), loss=None):
        self.root = None  # Root node in dec. tree
        # Minimum n of samples to justify split
        self.min_samples_split = min_samples_split
        # The minimum impurity to justify split
        self.min_impurity = min_impurity
        # The maximum depth to grow the tree to
        self.max_depth = max_depth
        # Function to calculate impurity (classification: info gain, gini, regression: variance reduction.)
        self._impurity_calculation = None
        # Function to determine prediction of y at leaf (classification: majority vote, regression: mean)
        self._leaf_value_calculation = None
        # If y is one-hot encoded (multi-dim) or not (one-dim)
        self.one_dim = None
        # If Gradient Boost, use loss function to calculate negative gradient (residues)
        self.loss = loss

    def _build_tree(self, X, y, current_depth=0):
        """
        Recursive method which builds the decision tree and splits X and y together
        on the feature of X which (based on impurity) best separates the data

        X: np.array
        y: np.array or list have to be same length as X
        current_depth: int, calculate current depth of tree to meet max_depth hyper-parameters
        """

        best_criteria = None  # Feature index and threshold to store in each step
        best_sets = None  # Subsets of the data to store in each step

        # Check if expansion of y is needed for concat
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        # Add y as last column of X, y have to be last column otherwise will influence feature_i index when split
        Xy = np.concatenate((X, y), axis=1)

        n_samples, n_features = np.shape(X)
        # when meet user set criteria then keep split tree node, this is known as pre-pruning
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # record which feature get largest impurity
            largest_impurity = 0
            # calculate the impurity (gini, variance reduction, information gain) for each feature
            for feature_i in range(n_features):
                # get all value of feature_i
                feature_values = np.expend_dims(X[:, feature_i], axis=1)
                # get unique value for finding which split point is best (binary split in CART)
                # no matter it's regression tree or classification tree, use same way to split feature
                # numerical: discrete => find unique value => find optimal split point
                # categorical:  find unique value => find optimal split point
                # TODO: need change numerical feature to discrete into few group first then find unique value
                # this will speed up finding optimal split point process
                unique_values = np.unique(feature_values)

                # Iterate through all unique values of feature column i and
                # calculate the impurity
                for threshold in unique_values:
                    # Divide X and y depending on if the feature value of X at index feature_i
                    # meets the threshold
                    Xy1, Xy2 = divide_matrix_on_feature(Xy, feature_i, threshold)

                    if len(Xy1) > 0 and len(Xy2) > 0:
                        # Select the y values of the two array
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        # Calculate impurity
                        impurity = self._impurity_calculation(y, y1, y2)

                        # If this threshold resulted in a higher information gain than previously
                        # recorded save the threshold value and the feature index
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],  # X of left subtree
                                "lefty": Xy1[:, n_features:],  # y of left subtree
                                "rightX": Xy2[:, :n_features],  # X of right subtree
                                "righty": Xy2[:, n_features:]  # y of right subtree
                            }
            # finish one tree node split and find optimal split feature and threshold for this node
            # check if largest larger then user setting hyper-parameters, known as pre-pruning
            if largest_impurity > self.min_impurity:
                # Build subtrees for the right and left branches
                cur_node = DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria["threshold"])
                cur_node.true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
                cur_node.false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
                return cur_node
                # true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
                # false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)

                # return DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria["threshold"],
                #                     true_branch=true_branch, false_branch=false_branch)
            else:
                # we meet leaf node and we have to stop recursive
                # calculate leaf value (majority vote or mean)
                leaf_value = self._leaf_value_calculation(y)
                # store leaf value into node
                return DecisionNode(value=leaf_value)

    def fit(self, X, y, loss=None):
        """ Build and train decision tree """
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(X, y)
        self.loss = None

    def predict_value(self, x, tree=None):
        """
        Do a recursive search down the tree and make a prediction of the data sample by the
        value of the leaf that we end up at

        x: one sample from test, mush have same order of features as fit X
        tree: Decision Tree node
        """
        # initial set
        if tree is None:
            tree = self.root

        # If we have a value, which means we already meet leaf node, stop recursive, return value as the prediction
        if tree.value is not None:
            return tree.value

        # Choose the feature that we will test
        feature_value = x[tree.feature_i]

        # Determine if we will go left or right branch
        # if it's numerical, same as divide, if it's categorical, same as equal
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
            else:
                branch = tree.false_branch
        else:
            if feature_value == tree.threshold:
                branch = tree.true_branch
            else:
                branch = tree.false_branch

        return self.predict_value(x, branch)

    def predict(self, X):
        """ Classify samples one by one and return the set of labels """
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred


class RegressionTree(DecisionTree):
    """build regression tree, this is sub class of decision tree object"""

    def _calculate_variance_reduction(self, y, y1, y2):
        """
        CART regression tree try to minimum variance on each sub tree, after feature A threshold applied split
        CART take most variance reduction feature threshold as best feature threshold

        y: np.array, original y value before split
        y1: np.array, left true branch y value
        y2: np.array, right false branch
        """
        var_tot = calculate_variance(y)
        var_1 = calculate_variance(y1)
        var_2 = calculate_variance(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)

        # Calculate the variance reduction
        # we want reduction maximum, which equal minimum this part (frac_1 * var_1 + frac_2 * var_2)
        variance_reduction = var_tot - (frac_1 * var_1 + frac_2 * var_2)

        return sum(variance_reduction)

    def _mean_of_y(self, y):
        """
        calculate mean of array like y for leaf node final value output
        y: np.array, leaf node region y
        """
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]

    def fit(self, X, y):
        """fit decision tree will call fit method in super class"""
        self._impurity_calculation = self._calculate_variance_reduction
        self._leaf_value_calculation = self._mean_of_y
        super(RegressionTree, self).fit(X, y)


class ClassificationTree(DecisionTree):
    """build classification tree, this is sub class of decision tree object"""
    def _calculate_information_gain(self, y, y1, y2):
        """
        ID3 classification tree try to minimum entropy on each sub tree, after feature A threshold applied split
        ID3 take most information gain feature threshold as best feature threshold

        y: np.array, original y value before split
        y1: np.array, left true branch y value
        y2: np.array, right false branch
        """
        # Calculate information gain
        p = len(y1) / len(y)
        entropy = calculate_entropy(y)
        info_gain = entropy - p * calculate_entropy(y1) - (1 - p) * calculate_entropy(y2)

        return info_gain

    def _calculate_gini_index_gain(self, y, y1, y2):
        """
        CART classification tree try to minimum gini index on each sub tree, after feature A threshold applied split
        CART take most gini index gain feature threshold as best feature threshold

        y: np.array, original y value before split
        y1: np.array, left true branch y value
        y2: np.array, right false branch
        """
        # Calculate gini index gain
        p = len(y1) / len(y)
        gini_index = gini(y)
        # we want maximum gain, which equal to minimum p * gini(y1) + (1 - p) * gini(y2), this is Gini index after split
        gini_index_gain = gini_index - (p * gini(y1) + (1 - p) * gini(y2))

        return gini_index_gain

    def _majority_vote(self, y):
        """
        calculate most frequency category in array like y for leaf node final value output
        y: np.array, leaf node region y
        """
        most_common = None
        max_count = 0
        for label in np.unique(y):
            # Count number of occurences of samples with label
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common

    def fit(self, X, y):
        """fit decision tree will call fit method in super class"""
        self._impurity_calculation = self._calculate_information_gain
        self._leaf_value_calculation = self._majority_vote
        super(RegressionTree, self).fit(X, y)
