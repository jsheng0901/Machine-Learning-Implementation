import numpy as np
import math


def calculate_variance(X):
    """ Return the variance of the features in dataset X """
    mean = np.ones(np.shape(X)) * X.mean(0)
    n_samples = np.shape(X)[0]
    variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))

    return variance


def calculate_entropy(y):
    """ Calculate the entropy of label array y """
    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * log2(p)
    return entropy


def calculate_unique_value(y):
    """calculate unique value frequency in x array"""
    # return results {type1:type1_count, type2:type2_count,  ... typeN:typeN_count}

    results = {}
    for data in y:
        # data[-1] means dataType
        if data in results:
            results[data] += 1
        else:
            results[data] = 1
    return results


def gini(y):
    """calculate Gini Index value of region y"""

    length = len(y)
    results = calculate_unique_value(y)
    imp = 0
    for i in results.keys():
        imp += (results[i] / length) ** 2
    return 1 - imp


def sigmoid (x):
    """calculate sigmoid  value, known as 1 / (1 + exp(-x)"""

    return 1 / (1 + np.exp(-x))
