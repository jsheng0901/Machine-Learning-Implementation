import numpy as np


class KFoldCrossValidation:
    """
    Implement k-fold cross-validation. Naive way to implement not consider imbalance dataset.
    Parameters:
    -----------
    k: int
        The number of fold the algorithm will form.
    """

    def __init__(self, k):
        self.k = k

    def split(self, n):
        """
        Split n sample into k fold each fold contains train and validation indices
        Args:
            n: int, number of sample need to be split

        Returns:
            folds: list of tuple, each tuple are train and validation indices
        """
        # get fold size for each fold
        fold_size = n // self.k
        # create all indices for each sample
        indices = np.arange(n)
        # random shuffle the whole indices
        np.random.shuffle(indices)
        # create the output folds
        folds = []

        for i in range(self.k):
            # build validation fold indices first
            validation_indices = indices[i * fold_size: (i + 1) * fold_size]
            # build train fold by concat all other indices
            train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
            folds.append((train_indices, validation_indices))

        return folds


# TODO implement
class BayesianOptimization:
    pass
