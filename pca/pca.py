import numpy as np
from utils.data_operation import standardize_data
from scipy.linalg import svd


class PCA:
    def __init__(self, n_components, solver="eigen"):
        """
        Principal component analysis (PCA) implementation.
        Transforms a dataset of possibly correlated values into n linearly
        uncorrelated components. The components are ordered such that the first
        has the largest possible variance and each following component as the
        largest possible variance given the previous components. This causes
        the early components to contain most of the variability in the dataset.
        Parameters
        ----------
        n_components : int, number of dimension after reduce
        solver : str, default 'eigen', {'svd', 'eigen'}
        """
        self.solver = solver
        self.n_components = n_components
        self.components = None          # eigenvectors after pick corresponding top n_components eigenvalues

    def decompose(self, x):
        """
        Decompose X to get eigenvalues, eigenvectors from this formular X^T * X * W = lambda * W
        Args:
            x: array type dataset (n_samples, n_features)

        Returns:
            None, generate components which is W from above formular as sorted and selected eigenvectors
            W -> (n_features, n_components)
        """

        # Mean centering first
        x = standardize_data(x)

        if self.solver == "svd":
            _, eigenvalues, eigenvectors = svd(x)
        else:
            # eigen method
            # get covariance matrix first cov -> [n_feature, n_feature]
            covariance_matrix = np.cov(x.T)
            # eigenvectors -> [n_feature, n_feature], eigenvalues -> [n_feature,]
            # for each eigenvector we have a corresponding eigenvalues
            eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # sort eigenvalues and pick top n_component
        sorted_eigenvalues_id = eigenvalues.argsort()[::-1]
        # sort eigenvectors from corresponding max to min eigenvalues id
        eigenvectors = eigenvectors[:, sorted_eigenvalues_id]
        # get n_components eigenvectors
        self.components = eigenvectors[:, :self.n_components]

        # get variance ratio based on top n_components
        eigenvalues_squared = np.power(sorted(eigenvalues, reverse=True), 2)
        variance_ratio = eigenvalues_squared / eigenvalues_squared.sum()
        print("Explained variance ratio: %s" % (variance_ratio[0: self.n_components]))

    def fit(self, x):
        """
        Build PCA components
        Args:
            x: array type dataset (n_samples, n_features)

        Returns: None
        """
        self.decompose(x)

    def transform(self, x):
        """
        Transfer original x to new dimension axis x, x can be any size only dimension have to same as fit x
        Args:
            x: array type dataset (n_samples, n_features)

        Returns:
            X_trans: array type dataset (n_samples, n_components), after applied pca dimension reduction
        """
        # Mean centering first
        x = standardize_data(x)
        # check if W already create or not
        if self.components is None:
            self.fit(x)
        # (n_samples, n_features) * (n_features, n_components) -> (n_samples, n_components)
        x_trans = np.dot(x, self.components)

        return x_trans
