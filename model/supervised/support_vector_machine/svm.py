import progressbar
from utils.misc import bar_widgets
from utils.kernel import *


class SVM:
    """The Support Vector Machine classifier, solved in simple SMO method

        Parameters:
        -----------
        C: float
            Penalty term.
        kernel: function
            Kernel function. Can be either polynomial, rbf or linear.
        power: int
            The degree of the polynomial kernel. Will be ignored by the other
            kernel functions.
        gamma: float
            Used in the rbf kernel function.
        coef: float
            Bias term used in the polynomial kernel function.
        difference: float
            Threshold for check converge
        max_iter: int
            maximum loop times for optimized alpha
    """

    def __init__(self, C=1, kernel=None, power=4, gamma=None, coef=4, difference=1e-3, max_iter=100):

        self.C = C  # penalty term
        self.difference = difference  # check converge
        self.max_iter = max_iter  # loop times

        if kernel is None:
            self.kernel = LinearKernel()  # if no kernel, use linear classifier for svm
        else:
            self.kernel = kernel

        self.b = 0  # intercept
        self.alpha = None  # lagrange multipliers
        self.K = None  # features after transfer by kernel function
        self.bar = progressbar.ProgressBar(widgets=bar_widgets)  # progress bar

    def fit(self, X, y=None):
        """fit SVM model"""
        self.X = X
        self.Y = y
        self.n_samples, self.n_features = X.shape
        self.K = np.zeros((self.n_samples, self.n_samples))  # initial kernel dot product results (n, n) with all 0
        for i in range(self.n_samples):
            self.K[:, i] = self.kernel(self.X, self.X[i, :])  # transfer each sample by kernel function n --> m
        self.alpha = np.zeros(self.n_samples)  # initial lagrange multipliers with all 0 (n, )
        # self.sv_idx = np.arange(0, self.n_samples)
        self.train()

    def train(self):
        """train SVM used SMO method"""

        for iter in self.bar(range(self.max_iter)):
            alpha_prev = np.copy(self.alpha)  # deep copy current alpha for later check converge

            for j in range(self.n_samples):
                # random choose second lagrange multipliers
                i = self.random_index(j)
                # Error for current examples and i, j alphas
                error_i, error_j = self.error(i), self.error(j)

                # check if two alphas satisfy KKT rules, and choose break KKT rules most one self.alpha[j]
                if (self.Y[j] * error_j < -0.001 and self.alpha[j] < self.C) or (
                        self.Y[j] * error_j > 0.001 and self.alpha[j] > 0):

                    # jth lagrange multipliers eta which used for later update lagrange
                    eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]

                    if eta >= 0:
                        continue

                    # get bound of current alpha
                    L, H = self.get_bounds(i, j)
                    old_alpha_i, old_alpha_j = self.alpha[i], self.alpha[j]  # old alphas value
                    self.alpha[j] -= (self.Y[j] * (error_i - error_j)) / eta  # update second random choose alpha j

                    # according to constraint L, H, get second pick new alpha j value and update first alpha i value
                    self.alpha[j] = self.clip(self.alpha[j], H, L)
                    # need check formula again for update alpha new alpha i
                    self.alpha[i] = self.alpha[i] + self.Y[i] * self.Y[j] * (old_alpha_j - self.alpha[j])

                    # update intercept b
                    b1 = self.b - error_i - self.Y[i] * (self.alpha[i] - old_alpha_j) * self.K[i, i] - \
                         self.Y[j] * (self.alpha[j] - old_alpha_j) * self.K[i, j]
                    b2 = self.b - error_j - self.Y[j] * (self.alpha[j] - old_alpha_j) * self.K[j, j] - \
                         self.Y[i] * (self.alpha[i] - old_alpha_i) * self.K[i, j]
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = 0.5 * (b1 + b2)

            # check if converge
            diff = np.linalg.norm(self.alpha - alpha_prev)
            if diff < self.difference:
                break

    def predict_row(self, X):
        """calculate each sample predict sign (label)"""

        # calculate kernel(xi, xj) results
        kernel_ij = self.kernel(self.X, X)

        # calculate wx + b, w = alpha * y * kernel(x)
        return np.dot((self.alpha * self.Y).T, kernel_ij.T) + self.b

    def predict(self, X):
        """predict all sample labels"""
        n = X.shape[0]
        result = np.zeros(n)
        for i in range(n):
            result[i] = np.sign(self.predict_row(X[i, :]))  # in svm, positive means +1 label, negative means -1 label
        return result

    def random_index(self, first_alpha):
        """random pick one alpha lagrange multiplier must different with for j in loop"""
        i = first_alpha
        while i == first_alpha:
            i = np.random.randint(0, self.m - 1)
        return i

    def error(self, i):
        """predict error use predict value - true value"""
        return self.predict_row(self.X[i]) - self.Y[i]

    def get_bounds(self, i, j):
        """get alpha bounds L and H according to previous alpha value"""

        if self.Y[i] != self.Y[j]:
            # yi != yj which means alpha_i - alpha_j = k (constant), there line is parallel positive
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
        else:
            # opposite of above, parallel negative
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])

        return L, H

    def clip(self, alpha, H, L):
        """according to bounds clip the alpha get final alpha value, alpha mush in L, H bound"""
        if alpha > H:
            alpha = H
        elif alpha < L:
            alpha = L

        return alpha
