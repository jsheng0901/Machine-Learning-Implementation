import numpy as np
import math
from utils.data_operation import calculate_covariance_matrix


class GaussianMixtureModel:
    """
    A probabilistic clustering method for determining groupings among data samples.
    Parameters:
    -----------
    k: int
        The number of clusters the algorithm will form.
    max_iterations: int
        The number of iterations the algorithm will run for if it does
        not converge before that.
    tolerance: float
        If the difference of the results from one iteration to the next is
        smaller than this value we will say that the algorithm has converged.
    """

    def __init__(self, k=2, max_iterations=2000, tolerance=1e-8):
        self.k = k
        self.parameters = []
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.responsibilities = []
        self.sample_assignments = None
        self.responsibility = None

    def init_random_gaussians(self, X):
        """ Initialize gaussian randomly """
        n_samples = np.shape(X)[0]
        self.priors = (1 / self.k) * np.ones(self.k)        # initial priors as same weights P(Ck)
        for i in range(self.k):
            params = {}
            params["mean"] = X[np.random.choice(range(n_samples))]  # random pick one x from sample as mean
            params["cov"] = calculate_covariance_matrix(X)          # initial cov matrix is same
            self.parameters.append(params)

    def multivariate_gaussian(self, X, params):
        """ Calculate Likelihood for all points in one cluster """
        n_features = np.shape(X)[1]
        mean = params["mean"]
        covar = params["cov"]
        determinant = np.linalg.det(covar)
        likelihoods = np.zeros(np.shape(X)[0])      # gather each sample likelihoods result
        for i, sample in enumerate(X):
            d = n_features  # dimension
            # just gaussian distribution formula pdf for one sample
            coeff = (1.0 / (math.pow((2.0 * math.pi), d / 2) * math.sqrt(determinant)))
            exponent = math.exp(-0.5 * (sample - mean).T.dot(np.linalg.pinv(covar)).dot((sample - mean)))
            likelihoods[i] = coeff * exponent

        return likelihoods

    def get_likelihoods(self, X):
        """ Calculate the likelihood over all samples and all clusters P(Xi|Ck) """
        n_samples = np.shape(X)[0]
        likelihoods = np.zeros((n_samples, self.k))     # ex: 100 * 3
        for i in range(self.k):
            likelihoods[:, i] = self.multivariate_gaussian(X, self.parameters[i])
        return likelihoods

    def expectation(self, X):
        """ Calculate the responsibility P(Ck|Xi) and update each sample label, E step """
        # Calculate probabilities of X belonging to the different clusters
        weighted_likelihoods = self.get_likelihoods(X) * self.priors    # P(Xi|Ck) * P(Ck)
        sum_likelihoods = np.expand_dims(np.sum(weighted_likelihoods, axis=1), axis=1)  # ex: (100, 3) -> (100, 1)
        # Determine responsibility as P(Xi|Ck)*P(Ck)/P(Xi)
        self.responsibility = weighted_likelihoods / sum_likelihoods    # ex: (100, 3) / (100, 1) -> (100, 3)
        # Assign samples to cluster that has largest probability, here is label
        self.sample_assignments = self.responsibility.argmax(axis=1)    # ex: (100, 3) -> (100, 1)
        # Save value for convergence check, here is max probability in which cluster
        self.responsibilities.append(np.max(self.responsibility, axis=1))   # ex: (100, 3) -> (100, ), [(100, )]

    def maximization(self, X):
        """ Update the parameters and priors, M step """
        # Iterate through clusters and recalculate mean, covariance and priors
        for i in range(self.k):
            resp = np.expand_dims(self.responsibility[:, i], axis=1)
            # (sum(P(Ck|Xi) * Xi) / sum(P(Ck|Xi))) formula
            mean = (resp * X).sum(axis=0) / resp.sum()
            covariance = (X - mean).T.dot((X - mean) * resp) / resp.sum()
            self.parameters[i]["mean"], self.parameters[i]["cov"] = mean, covariance

        # Update weights, formula sum(P(Ck|Xi) / number_samples
        n_samples = np.shape(X)[0]
        self.priors = self.responsibility.sum(axis=0) / n_samples

    def converged(self, X):
        """ Check if it's converged, method: | likehood - last_likelihood | < tolerance """
        if len(self.responsibilities) < 2:  # which means just run one iteration
            return False
        # current each points max probability - previous each points max probability, and calculate norm of vector
        diff = np.linalg.norm(self.responsibilities[-1] - self.responsibilities[-2])
        # print ("Likelihood update: %s (tol: %s)" % (diff, self.tolerance))
        return diff <= self.tolerance

    def predict(self, X):
        """ Run GMM and return the cluster indices """
        # Initialize the gaussian randomly
        self.init_random_gaussians(X)

        # Run EM until convergence or for max iterations
        for _ in range(self.max_iterations):
            self.expectation(X)  # E-step
            self.maximization(X)  # M-step

            # Check convergence
            if self.converged(X):
                break

        # Make new assignments and return them
        self.expectation(X)
        return self.sample_assignments