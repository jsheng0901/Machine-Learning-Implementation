import numpy as np
from scipy.linalg import svd
import logging


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
        self.mean = None                # mean of decompose x

    def decompose(self, X):
        """decompose X to get eigenvalues, eigenvectors"""

        # Mean centering first
        X = X.copy()
        X -= self.mean

        if self.solver == "svd":
            _, eigenvalues, eigenvectors = svd(X, full_matrices=True)
        elif self.solver == "eigen":
            # get covariance matrix first
            covariance_matrix = np.cov(X.T)
            eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # sort eigenvalues and pick top n_component
        sorted_eigenvalues_id = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, sorted_eigenvalues_id]
        self.components = eigenvectors[:, :self.n_components]

        # get variance ratio based on top n_components
        eigenvalues_squared = sorted(eigenvalues)[::-1] ** 2
        variance_ratio = eigenvalues_squared / eigenvalues_squared.sum()
        logging.info("Explained variance ratio: %s" % (variance_ratio[0: self.n_components]))

    def fit(self, X):
        """build PCA"""
        self.mean = np.mean(X, axis=0)
        self.decompose(X)

    def transform(self, X):
        """transfer original x to new dimension axis x, x can be any size only dimension have to same as fit x"""
        X = X.copy()
        X -= self.mean
        return np.dot(X, self.components)
