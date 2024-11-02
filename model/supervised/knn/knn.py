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

        # initial define global x and y
        self.x = None
        self.y = None

    def vote(self, neighbor_labels):
        """
        Return the most common class among the neighbor samples
        Args:
            neighbor_labels: list type (number of k)

        Returns:
            label: int, label value
        """
        # count each class occurrence
        counts = np.bincount(neighbor_labels.astype('int'))
        # get most frequent class label according to k neighborhoods
        label = counts.argmax()
        return label

    def fit(self, x, y):
        """
        Store x, y in global, since knn is lazy model
        Args:
            x: array type dataset (n_samples, n_features)
            y: array type dataset (n_samples)

        Returns:
            None
        """
        self.x = x
        self.y = y

    def predict(self, x_test):
        """
        Get label according to the neatest neighbors of each x test data.
        Args:
            x_test: array type dataset (n_samples, n_features)

        Returns:
            y_pred: array type dataset (n_samples), predict label for each sample
        """
        y_pred = np.empty(x_test.shape[0])
        # Determine the class of each sample
        for i, test_sample in enumerate(x_test):
            # Sort the training samples by their distance to the test sample and get the K nearest
            idx = np.argsort([euclidean_distance(test_sample, x) for x in self.x])[:self.k]
            # Extract the labels of the K the nearest neighboring training samples
            k_nearest_neighbors = np.array([self.y[i] for i in idx])
            # Label sample as the most common class label
            y_pred[i] = self.vote(k_nearest_neighbors)

        return y_pred