from abc import abstractmethod, ABC

import numpy as np
from utils.data_manipulation import divide_matrix_on_feature
from utils.data_operation import calculate_variance, calculate_entropy, gini


class DecisionNode:
    """
    Class that represents a decision node or leaf in the decision tree
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
        self.true_branch = true_branch  # Left subtree
        self.false_branch = false_branch  # Right subtree


# This is Super class of RegressionTree and ClassificationTree
class DecisionTree:
    """
    Super class of RegressionTree and ClassificationTree.
    Parameters:
    -----------
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    """

    def __init__(self, min_samples_split=2, min_impurity=1e-7, max_depth=float("inf")):
        self.root = None  # Root node in decision tree

        # Minimum n of samples to justify split
        self.min_samples_split = min_samples_split
        # The minimum impurity to justify split
        self.min_impurity = min_impurity
        # The maximum depth to grow the tree to
        self.max_depth = max_depth
        # If y is one-hot encoded (multi-dim) or not (one-dim)
        self.one_dim = None

    @abstractmethod
    def _impurity_calculation(self, y, y1, y2):
        """
        Function to calculate impurity (classification: info gain, gini, regression: variance reduction.)
        Args:
            y: array type dataset, original y value before split
            y1: array type dataset, left true branch y value
            y2: array type dataset, right false branch

        Returns:
            Customized impurity output
        """
        return NotImplementedError()

    @abstractmethod
    def _leaf_value_calculation(self, y):
        """
        Function to determine prediction of y at leaf (classification: majority vote, regression: mean)
        Args:
            y: array type dataset, leaf node region y

        Returns:
            Customized leaf value output
        """
        return NotImplementedError()

    def _build_tree(self, x, y, current_depth=0):
        """
        Recursive method which builds the decision tree and splits x and y together
        on the feature of x which (based on impurity) best separates the data. Check all user
        setting hyperparameters before we go into recursive split dataset.
        Args:
            x: array type dataset (n_samples, n_features)
            y: array type dataset (n_samples)
            current_depth: int, calculate current depth of tree to meet max_depth hyperparameters

        Returns:
            cur_node: DecisionTree object, after build decision tree root node
        """

        best_criteria = None  # feature index and threshold to store in each step, initial as None at each recursive
        best_sets = None  # subsets of the data to store in each step, initial as None at each recursive

        # check if expansion of y is needed for concat
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        n_samples, n_features = np.shape(x)

        # add y as last column of x, y have to be last column otherwise will influence feature_i index when split
        xy = np.concatenate((x, y), axis=1)
        # record which feature get the largest impurity, initial as 0
        largest_impurity = 0
        # calculate the impurity (gini, variance reduction, information gain) for each feature
        for feature_i in range(n_features):
            # get all value of feature_i
            # feature_values = np.expand_dims(x[:, feature_i], axis=1)
            feature_values = x[:, feature_i]
            # get unique value for finding which split point is best (binary split in CART)
            # no matter it's regression tree or classification tree, use same way to split feature
            # numerical: discrete => group value => find unique group value => find optimal split point
            # categorical:  find unique value => find optimal split point
            # TODO: need change numerical feature to discrete into few group first then find unique value
            # this will speed up finding optimal split point process
            unique_values = np.unique(feature_values)

            # Iterate through all unique values of feature column i and
            # calculate the impurity
            for threshold in unique_values:
                # divide x and y depending on if the feature value of X at index feature_i
                # meets the threshold
                xy1, xy2 = divide_matrix_on_feature(xy, feature_i, threshold)

                if len(xy1) > 0 and len(xy2) > 0:
                    # select the y values of the two array
                    y1 = xy1[:, n_features:]
                    y2 = xy2[:, n_features:]

                    # calculate impurity
                    impurity = self._impurity_calculation(y, y1, y2)

                    # If this threshold resulted in a higher information gain than previously
                    # recorded save the threshold value, the feature index and best split left, right x, y dataset
                    if impurity > largest_impurity:
                        largest_impurity = impurity
                        best_criteria = {"feature_i": feature_i, "threshold": threshold}
                        best_sets = {
                            "left_x": xy1[:, :n_features],  # x of left subtree
                            "left_y": xy1[:, n_features:],  # y of left subtree
                            "right_x": xy2[:, :n_features],  # x of right subtree
                            "right_y": xy2[:, n_features:]  # y of right subtree
                        }

        # finish one tree node split and find optimal split feature and threshold for this node
        # check if impurity larger than user setting hyperparameters, known as pre-pruning
        # check if left and right subtrees meet user setting hyperparameters
        # get left and right sample size, if no best sets which means no more split set left right size as 0
        left_n_samples = best_sets["left_x"].shape[0] if best_sets else 0
        right_n_samples = best_sets["right_x"].shape[0] if best_sets else 0
        next_depth = current_depth + 1
        # must meet all user setting hyperparameters than we will keep split tree
        if largest_impurity > self.min_impurity and left_n_samples >= self.min_samples_split and \
                right_n_samples >= self.min_samples_split and next_depth <= self.max_depth:
            # build current node for the right and left branches root node
            cur_node = DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria["threshold"])
            # assign true and false two subtree to current node
            cur_node.true_branch = self._build_tree(best_sets["left_x"], best_sets["left_y"], next_depth)
            cur_node.false_branch = self._build_tree(best_sets["right_x"], best_sets["right_y"], next_depth)
            return cur_node
        else:
            # we meet leaf node, and we have to stop recursive split dataset
            # calculate leaf value (majority vote or mean)
            leaf_value = self._leaf_value_calculation(y)
            # store leaf value into node
            cur_node = DecisionNode(value=leaf_value)
            return cur_node

    def fit(self, x, y):
        """
        Build and train decision tree
        Args:
            x: array type dataset (n_samples, n_features)
            y: array type dataset (n_samples)

        Returns:
            None, build decision tree
        """

        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(x, y)

    def predict_value(self, x, tree=None):
        """
        Do a recursive search down the tree and make a prediction of the data sample by the
        value of the leaf that we end up at.
        Args:
            x: array type dataset (n_samples, n_features) one sample, mush have same ordered of features as fit x
            tree: DecisionTree node, default is None which know as traversal from root.

        Returns:
            left_value: float, final predict output value.
        """

        # initial set
        if tree is None:
            tree = self.root

        # If we have a value, which means we already meet leaf node, stop recursive, return value as the prediction
        if tree.value is not None:
            left_value = tree.value
            return left_value

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

        left_value = self.predict_value(x, branch)

        return left_value

    def predict(self, x):
        """
        Classify samples one by one and return the set of labels
        Args:
            x: array type dataset (n_samples, n_features)

        Returns:
            y_pred: array type dataset (n_samples), final predict value
        """

        y_pred = [self.predict_value(sample) for sample in x]

        return y_pred


class RegressionTree(DecisionTree, ABC):
    """
    Build regression tree, this is subclass of decision tree object
    """

    def _impurity_calculation(self, y, y1, y2):
        """
        CART regression tree try to minimum variance on each subtree, after feature A threshold applied split
        CART take most variance reduction feature threshold as best feature threshold
        Args:
            y: array type dataset, original y value before split
            y1: array type dataset, left true branch y value
            y2: array type dataset, right false branch

        Returns:
            variance_reduction: float, variance reduction for this split
        """

        var_tot = calculate_variance(y)
        var_1 = calculate_variance(y1)
        var_2 = calculate_variance(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)

        # Calculate the variance reduction
        # we want reduction maximum, which equal minimum this part (frac_1 * var_1 + frac_2 * var_2)
        variance_reduction = var_tot - (frac_1 * var_1 + frac_2 * var_2)

        return variance_reduction

    def _leaf_value_calculation(self, y):
        """
        Calculate mean of array like y for leaf node final value output
        Args:
            y: array type dataset, leaf node region y

        Returns:
            value: float, final predict leaf value
        """
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]


class ClassificationCARTTree(DecisionTree, ABC):
    """
    Build classification CART tree, this is subclass of decision tree object
    """

    def _impurity_calculation(self, y, y1, y2):
        """
        CART classification tree try to minimum gini index on each sub ree, after feature A threshold applied split
        CART take most gini index gain feature threshold as best feature threshold
        Args:
            y: array type dataset, original y value before split
            y1: array type dataset, left true branch y value
            y2: array type dataset, right false branch

        Returns:
            gini_index_gain: float, gini index gain after split
        """

        # Calculate gini index gain
        p = len(y1) / len(y)
        gini_index = gini(y)
        # we want maximum gain, which equal to minimum p * gini(y1) + (1 - p) * gini(y2), this is Gini index after split
        gini_index_gain = gini_index - (p * gini(y1) + (1 - p) * gini(y2))

        return gini_index_gain

    def _leaf_value_calculation(self, y):
        """
        Calculate most frequency category in array like y for leaf node final value output
        Args:
            y: array type dataset, leaf node region y

        Returns:
            most_common: int, most common label in that leaf
        """

        most_common = None
        max_count = 0
        for label in np.unique(y):
            # Count number of occurrences of samples with label
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common


class ClassificationID3Tree(DecisionTree, ABC):
    """
    Build classification ID3 tree, this is subclass of decision tree object
    """

    def _impurity_calculation(self, y, y1, y2):
        """
        ID3 classification tree try to minimum entropy on each subtree, after feature A threshold applied split
        ID3 take most information gain feature threshold as best feature threshold. Here ID3 tree we build BST, but
        in original paper it is N-search-tree.
        Args:
            y: array type dataset, original y value before split
            y1: array type dataset, left true branch y value
            y2: array type dataset, right false branch

        Returns:
            info_gain: float, information gain after split
        """

        # Calculate information gain
        p = len(y1) / len(y)
        entropy = calculate_entropy(y)
        info_gain = entropy - p * calculate_entropy(y1) - (1 - p) * calculate_entropy(y2)

        return info_gain

    def _leaf_value_calculation(self, y):
        """
        Calculate most frequency category in array like y for leaf node final value output
        Args:
            y: array type dataset, leaf node region y

        Returns:
            most_common: int, most common label in that leaf
        """

        most_common = None
        max_count = 0
        for label in np.unique(y):
            # Count number of occurrences of samples with label
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common
