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

    def decompose(self, X):
        """
        Decompose X to get eigenvalues, eigenvectors from this formular X^T * X * W = lambda * W
        Args:
            X: array type dataset (n_samples, n_features)

        Returns:
            None, generate components which is W from above formular as eigenvectors
            W -> (n_features, n_components)
        """

        # Mean centering first
        X = standardize_data(X)

        if self.solver == "svd":
            _, eigenvalues, eigenvectors = svd(X)
        elif self.solver == "eigen":
            # get covariance matrix first cov -> [n_feature, n_feature]
            covariance_matrix = np.cov(X.T)
            # eigenvectors -> [n_feature, n_feature], eigenvectors -> [n_feature,]
            eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # sort eigenvalues and pick top n_component
        sorted_eigenvalues_id = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, sorted_eigenvalues_id]
        self.components = eigenvectors[:, :self.n_components]

        # get variance ratio based on top n_components
        eigenvalues_squared = np.power(sorted(eigenvalues, reverse=True), 2)
        variance_ratio = eigenvalues_squared / eigenvalues_squared.sum()
        print("Explained variance ratio: %s" % (variance_ratio[0: self.n_components]))

    def fit(self, X):
        """
        Build PCA components
        Args:
            X: array type dataset (n_samples, n_features)

        Returns: None
        """
        self.decompose(X)

    def transform(self, X):
        """
        Transfer original x to new dimension axis x, x can be any size only dimension have to same as fit x
        Args:
            X: array type dataset (n_samples, n_features)

        Returns:
            X_trans: array type dataset (n_samples, n_components), after applied pca dimension reduction
        """
        # Mean centering first
        X = standardize_data(X)
        # check if W already create or not
        if self.components is None:
            self.fit(X)

        X_trans = np.dot(X, self.components)

        return X_trans
