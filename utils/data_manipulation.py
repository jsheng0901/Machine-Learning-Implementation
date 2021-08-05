import numpy as np


def to_categorical(x, n_col=None):
    """ One-hot encoding of nominal values """
    if not n_col:
        n_col = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_col))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot


def divide_matrix_on_feature(X, feature_i, threshold):
    """
    Divide dataset based on if sample value on feature index is larger than
    the given threshold or lower then, this is binary split function

    X: np.array, concat both X and y together, y have to concat as last column
    feature_i: index of which feature it's in X
    threshold: specific unique value in feature index i
    """
    # check threshold is number or object,
    # if it's number, split as bigger or not, if it's object, split as equal or not
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold
    # iterate each row in X,
    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])

    return X_1, X_2
