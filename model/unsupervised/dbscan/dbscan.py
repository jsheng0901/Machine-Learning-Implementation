import numpy as np
from sklearn.metrics import pairwise_distances


class DBSCAN:
    """
    Basin implementation of DBSCAN clustering method
    DBSCAN good for any shape of clusters, including non-convex dataset,
    better than KMeans when dataset have lots of noise data.
    Parameters:
    -----------
    metric: str
        The method of calculate distance metric between two points
    eps: float
        The max number that control how neighbor radius which will define point as core point
    min_samples: int
        The number of samples in a neighborhood for a point to be considered as core point
    """
    def __init__(self, metric: str = 'euclidean', eps: float = 0.1, min_samples: int = 5):
        self.metric = metric
        self.eps = eps
        self.min_samples = min_samples

    def _get_all_distance(self, X):
        """
        Get pairwise distance matrix
        Args:
            X: array type dataset (n_samples, n_features)
        Returns:
            distance_matrix: array type dataset (n_samples, n_samples)
        """
        distance_matrix = pairwise_distances(X)
        return distance_matrix

    def get_center_points(self, X):
        """
        Get all initial center points and each point neighbor points index
        Args:
            X: array type dataset (n_samples, n_features)
        Returns:
            centers: dict (key: sample index, value: list of neighbors index)
        """
        centers = {}
        n_samples = X.shape[0]
        distance_matrix = self._get_all_distance(X)
        for i in range(n_samples):
            # get one sample(i) distance between all others
            distance = distance_matrix[i, :]
            # check if this sample has enough neighbors inside eps radius
            index = np.where(distance <= self.eps)[0]
            if len(index) - 1 >= self.min_samples:
                # if this sample it's center, then store in dict, remember when counting remove itself
                centers[i] = index

        return centers

    def fit(self, X):
        """
        Run DBSCAN clustering for input dataset X
        Args:
            X: array type dataset (n_samples, n_features)
        Returns:
            list of labels size same as input X number of samples
        """

        # get all center point first, center point is point in eps radius has more than min_samples points
        centers = self.get_center_points(X)
        labels = {}
        n_samples = X.shape[0]
        initial_centers = centers.copy()

        cluster_id = 0
        # keep check overall sample visited or not
        unvisited = list(range(n_samples))

        # loop over all centers
        while bool(centers):
            # initial visited points which will be grouped as a cluster in one loop
            visited = []
            # get all unvisited centers
            cores = centers.keys()
            # random pick one center
            core = np.random.choice(cores)
            visited.append(core)
            # get all core's neighborhoods
            core_neighbor = centers[core]
            # remove core from unvisited
            unvisited.remove(core)
            # loop over all core's neighbors
            while bool(core_neighbor):
                q = core_neighbor[0]
                # add q into visited list
                visited.append(q)
                # remove q from neighborhoods
                core_neighbor.pop(0)
                # remove q from unvisited
                unvisited.remove(q)
                if q in initial_centers:
                    # remove core from centers which means we visited
                    centers.pop(q)
                    # if neighbor is center point than we combine those two centers
                    sample_combined = [sample for sample in initial_centers[q] if sample in unvisited]
                    core_neighbor.add(sample_combined)

            # assign visited points to one cluster
            labels[cluster_id] = visited
            cluster_id += 1

        # assign label to all points, noise point as -1
        final_labels = np.full(n_samples, -1)
        for k, v in labels.items():
            final_labels[v] = k

        return final_labels

    def fit_predict(self, X):
        """ predict labels for input X """
        # TODO
