import numpy as np
from utils.data_operation import euclidean_distance


class KNN:
    """
    K Nearest Neighbors classifier.
    Parameters:
    -----------
    k: int
        The number of closest neighbors that will determine the class of the
        sample that we wish to predict.
    """
    def __init__(self, k=5):
        self.k = k

    def vote(self, neighbor_labels):
        """ Return the most common class among the neighbor samples """
        counts = np.bincount(neighbor_labels.astype('int'))
        return counts.argmax()

    def fit(self, X, y):
        """ store X, y in global """
        self.X = X
        self.y = y

    def predict(self, X_test):
        y_pred = np.empty(X_test.shape[0])
        # Determine the class of each sample
        for i, test_sample in enumerate(X_test):
            # Sort the training samples by their distance to the test sample and get the K nearest
            idx = np.argsort([euclidean_distance(test_sample, x) for x in self.X])[:self.k]
            # Extract the labels of the K nearest neighboring training samples
            k_nearest_neighbors = np.array([self.y[i] for i in idx])
            # Label sample as the most common class label
            y_pred[i] = self.vote(k_nearest_neighbors)

        return y_pred