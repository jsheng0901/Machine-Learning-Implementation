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
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # below are initial value for global use later
        self.parameters = []
        self.responsibilities = []
        self.sample_assignments = None
        self.responsibility = None
        self.priors = None

    def init_random_gaussian(self, x):
        """
        Initialize gaussian randomly
        Args:
            x: array type dataset (n_samples, n_features)

        Returns:
            None, initial param mean and cov for each class and add to parameters list
        """
        n_samples = np.shape(x)[0]
        # initial priors as same weights P(Ck), then each class has same weight
        self.priors = (1 / self.k) * np.ones(self.k)
        for i in range(self.k):
            # random pick one x from sample as mean, and initial cov matrix same for each class k
            params = {"mean": x[np.random.choice(range(n_samples))], "cov": calculate_covariance_matrix(x)}
            self.parameters.append(params)

    def multivariate_gaussian(self, x, params):
        """
        Calculate likelihood for all points in one cluster with params mean and covariance
        Args:
            x: array type dataset (n_samples, n_features)
            params: dictionary type with mean and cov keys ex: {mean: 1, cov: 2}

        Returns:
            likelihoods: array type (n_sample), each sample likelihood under this cluster params
        """
        n_features = np.shape(x)[1]
        n_sample = np.shape(x)[0]
        mean = params["mean"]
        covar = params["cov"]
        determinant = np.linalg.det(covar)
        # gather each sample likelihoods result, [n_sample,]
        likelihoods = np.zeros(n_sample)
        for i, sample in enumerate(x):
            d = n_features  # dimension
            # just gaussian distribution formula pdf for one sample
            coeff = (1.0 / (math.pow((2.0 * math.pi), d / 2) * math.sqrt(determinant)))
            exponent = math.exp(-0.5 * (sample - mean).T.dot(np.linalg.pinv(covar)).dot((sample - mean)))
            likelihoods[i] = coeff * exponent

        return likelihoods

    def get_likelihoods(self, x):
        """
        Calculate the likelihood over all samples and all clusters P(Xi|Ck)
        Args:
            x: array type dataset (n_samples, n_features)

        Returns:
            likelihoods: array type (n_sample, n_class -> k), likelihood of each sample to each class k
        """
        n_samples = np.shape(x)[0]
        # initial likelihoods output: [n_sample, n_class -> k]    ex: [100, 3]
        likelihoods = np.zeros((n_samples, self.k))
        for i in range(self.k):
            likelihoods[:, i] = self.multivariate_gaussian(x, self.parameters[i])
        return likelihoods

    def expectation(self, x):
        """
        Calculate the responsibility P(Ck|Xi) and update each sample label, E step,
        P(Ck|Xi) = P(Xi|Ck) * P(Ck) / sum(P(Xi|Cj) * P(Cj), along each class j)
        Args:
            x: array type dataset (n_samples, n_features)

        Returns:
            None, update sample_assignments (n_sample) and responsibilities (n_sample)
        """
        # Calculate probabilities of x belonged to the different clusters, P(Xi|Ck) * P(Ck)
        weighted_likelihoods = self.get_likelihoods(x) * self.priors
        # Calculate denominator of P(Ck|Xi), sum along each class, ex: (100, 3) -> (100, 1)
        sum_likelihoods = np.expand_dims(np.sum(weighted_likelihoods, axis=1), axis=1)
        # Determine responsibility as P(Xi|Ck)*P(Ck)/P(Xi), ex: (100, 3) / (100, 1) -> (100, 3)
        self.responsibility = weighted_likelihoods / sum_likelihoods
        # Assign samples to cluster that has the largest probability, here is how to label, ex: (100, 3) -> (100, 1)
        self.sample_assignments = self.responsibility.argmax(axis=1)
        # Save value for convergence check, here is max probability in which cluster, ex: (100, 3) -> (100, ), [(100, )]
        self.responsibilities.append(np.max(self.responsibility, axis=1))

    def maximization(self, x):
        """
        Update the parameters and priors, M step
        Args:
            x: array type dataset (n_samples, n_features)

        Returns:
            None
        """
        # Iterate through clusters and recalculate mean, covariance and priors
        for i in range(self.k):
            # get each class P(Ck|Xi) ex: [100, 1]
            resp = np.expand_dims(self.responsibility[:, i], axis=1)
            # (sum(P(Ck|Xi) * Xi) / sum(P(Ck|Xi))) formula
            # ex: [100, 1] * [100, n_feature] -> [n_feature] / sum all points and features -> [1]
            # mean -> [n_feature] for each k
            mean = (resp * x).sum(axis=0) / resp.sum()
            # [100, n_feature] - mean -> covariance matrix: [n_feature, n_feature]
            covariance = (x - mean).T.dot((x - mean) * resp) / resp.sum()
            self.parameters[i]["mean"], self.parameters[i]["cov"] = mean, covariance

        # Update weights, formula sum(P(Ck|Xi) / number_samples
        n_samples = np.shape(x)[0]
        # ex: [100, n_class] -> sum -> priors: [n_class ->k]
        self.priors = self.responsibility.sum(axis=0) / n_samples

    def converged(self, x):
        """
        Check if it's converged, since if each iteration update each point max likelihood diff is tiny, then converged
        method: each point max likelihood of which cluster, formular: | likelihood - last_likelihood | < tolerance
        Args:
            x: array type dataset (n_samples, n_features)

        Returns:
            True or False, boolean type, if converged or not
        """
        if len(self.responsibilities) < 2:  # which means just run one iteration
            return False
        # current each points max probability - previous each points max probability, and calculate norm of vector
        diff = np.linalg.norm(self.responsibilities[-1] - self.responsibilities[-2])

        return diff <= self.tolerance

    def fit(self, x):
        """
        Run GMM and return the cluster indices
        Args:
            x: array type dataset (n_samples, n_features)

        Returns:
            sample_assignments: (n_sample), each point assigned label according to max likelihood in which cluster
        """
        # Initialize the gaussian randomly
        self.init_random_gaussian(x)

        # Run EM until convergence or for max iterations
        for _ in range(self.max_iterations):
            self.expectation(x)  # E-step
            self.maximization(x)  # M-step

            # Check convergence
            if self.converged(x):
                break

        # Use last update parameters to make new assignments and return them
        self.expectation(x)

        return self.sample_assignments
