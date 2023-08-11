import numpy as np
from utils.data_operation import euclidean_distance


class KMeans:
    """
    A simple clustering method that forms k clusters by iteratively reassigning
    samples to the closest centroids and after that moves the centroids to the center
    of the new formed clusters.
    Parameters:
    -----------
    k: int
        The number of clusters the algorithm will form.
    max_iterations: int
        The number of iterations the algorithm will run for if it does not converge before that.
    """
    def __init__(self, k=2, max_iterations=500):
        self.k = k
        self.max_iterations = max_iterations

    def init_random_centroids(self, x):
        """
        Initialize the centroids as k random samples of x
        Args:
            x: array type dataset (n_samples, n_features)

        Returns:
            centroids: array type (k, n_features)
        """
        n_samples, n_features = np.shape(x)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = x[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    def closest_centroid(self, sample, centroids):
        """
        Return the index of the closest centroid to the sample
        Args:
            sample: 1d array type (1, n_feature)
            centroids: array type (k, n_features)

        Returns:
            closest_i: int, the closest centroid index for sample point
        """
        # initial closest index and distance
        closest_i = 0
        closest_dist = float('inf')
        # loop through all centroids check which one is close to sample
        for i, centroid in enumerate(centroids):
            distance = euclidean_distance(sample, centroid)
            if distance < closest_dist:
                closest_i = i
                closest_dist = distance
        return closest_i

    def create_clusters(self, centroids, x):
        """
        Assign the samples to the closest centroids to create clusters
        Args:
            centroids: array type (k, n_features)
            x: array type dataset (n_samples, n_features)

        Returns:
            clusters: nested list, (k number list), each cluster contains sample index
        """

        clusters = [[] for _ in range(self.k)]      # create the number of clusters list
        for sample_i, sample in enumerate(x):       # loop through each sample and calculate distance to assign cluster
            centroid_i = self.closest_centroid(sample, centroids)      # get index of which cluster belong to
            clusters[centroid_i].append(sample_i)   # add sample index
        return clusters

    def calculate_centroids(self, clusters, x):
        """
        Calculate new centroids as the means of the samples in each cluster
        Args:
            clusters: nested list, (k number list), each cluster contains sample index
            x: array type dataset (n_samples, n_features)

        Returns:

        """
        # initial centroids with [k, n_features] size
        n_features = np.shape(x)[1]
        centroids = np.zeros((self.k, n_features))
        # loop through each cluster with sample index
        for i, cluster in enumerate(clusters):
            # get each cluster sample index and calculate all samples mean,
            # [n_sample_one_cluster, n_feature] -> mean -> [1, n_feature]
            centroid = np.mean(x[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    def get_cluster_labels(self, clusters, x):
        """
        Classify samples as the index of their clusters
        Args:
            clusters: nested list, (k number list), each list is one cluster contains sample index
            x: array type dataset (n_samples, n_features)

        Returns:
            output: array type (n_sample), each sample final cluster index
        """

        # One prediction for each sample
        # this function just transform sample in each cluster to each sample matched cluster
        output = np.zeros(np.shape(x)[0])
        # loop through each cluster and each sample index inside one cluster
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                # assign each sample inside which cluster with this cluster label
                output[sample_i] = cluster_i
        return output

    def predict(self, x):
        """
        Do K-Means clustering and return cluster indices
        Args:
            x: array type dataset (n_samples, n_features)

        Returns:
            output: array type (n_sample), each sample final cluster index
        """

        # Initialize centroids as k random samples from X
        centroids = self.init_random_centroids(x)

        # Iterate until convergence or for max iterations
        for _ in range(self.max_iterations):
            # Assign samples to the closest centroids (create clusters)
            clusters = self.create_clusters(centroids, x)
            # Save current centroids for convergence check
            prev_centroids = centroids
            # Calculate new centroids from the clusters
            centroids = self.calculate_centroids(clusters, x)
            # If no centroids have changed => convergence, which equal to no sample change assigned cluster
            diff = centroids - prev_centroids
            if not diff.any():
                break
        # Assign final label cluster id for each sample in x
        output = self.get_cluster_labels(clusters, x)

        return output
