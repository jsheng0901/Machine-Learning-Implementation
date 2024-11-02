import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances


class Tsne:
    """
    Base implementation of t-sne dimension reduction algorithm
    T-sne is non-linear method dimension reduction
    T-sne good for catch local distribution than global distribution
    T-sne need long time when perplexity increase
    T-sne usually use for 2 or 3 dimension visualization
    Parameters:
    -----------
    n_components: int
        The dimension of the embedded space
    perplexity: float
        The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms.
        Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50.
        Different values can result in significantly different results.
    learning_rate: float
        The step distance of gradient descent
    n_iter: int
        Maximum number of iterations for the optimization gradient descent.
    """

    def __init__(self, n_components: int = 2, perplexity: float = 30, learning_rate: float = 200, n_iter: int = 1000):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.momentum = 0.9  # momentum control how previous gradient average influence current gradient

    def get_conditional_prob_matrix(self, distances, sigma, zero_index=None):
        """
        calculate conditional probability matrix p_j_i based on distance matrix (array)
        p_j_i = exp(-||xi - xj|| / 2*(sigma**2)) / sum(exp(-||xi - xk|| / 2*(sigma**2)), k)
        ||xi - xj|| is distance between xi and xj
        Args:
            distances: array
                distance array between point i to all others'
            sigma: float
                sigma value from binary search guess
            zero_index: int
                index of where i should be 0, since p_i_i is 0 in define
        Returns: (n_samples, n_samples)
            matrix with p_j_i conditional probability
        """
        # get p_j_i according to formular
        two_sig_sq = 2 * np.square(sigma)
        exp_x = np.exp(-distances / two_sig_sq)
        # check if it's calculate p_j_i inside binary search sigma method
        # or after find the best sigma inside get_p_join method
        if zero_index is None:
            # we fill all diagonal as 0 since p_i_i = 0, when zero index is None then exp_x is [n_sample, n_sample]
            np.fill_diagonal(exp_x, 0)
        else:
            exp_x[:, zero_index] = 0

        # add a tiny constant for stability of log we take later in calculate perplexity
        exp_x += 1e-8
        # get final conditional probability for each j given i as center
        p_j_i = exp_x / np.sum(exp_x, axis=1)

        return p_j_i

    def get_join_prob_matrix(self, conditional_prob):
        """
        get p join distribution probability matrix based on p conditional probability matrix
        p_ij = (p_j_i + p_i_j) / (2 * N)
        Args:
            conditional_prob: array (n_sample, n_sample)
                matrix with p_j_i conditional probability
        Returns: (n_sample, n_sample)
            matrix with p_ij join distribution probability
        """
        n = conditional_prob.shape[0]
        p_ij = (conditional_prob + conditional_prob.T) / (2 * n)

        return p_ij

    def get_perplexity(self, distances, sigma, zero_index):
        """
        calculate corresponding row (sample) perplexity defined in paper as Perp(pi) = 2 ** H(pi)
        H(pi) is entropy of pi point which is -sum(p_j_i * log(p_j_i, 2), j)
        Args:
            distances: array
                distance array between point i to all others'
            sigma: float
                sigma value from binary search guess
            zero_index: int
                index of where i should be 0, since p_i_i is 0 in define
        Returns: float
            perplexity value base on given sigma
        """
        # get conditional probability p_j_i
        conditional_prob = self.get_conditional_prob_matrix(distances, sigma, zero_index)
        # calculate entropy
        entropy = -np.sum(conditional_prob * np.log2(conditional_prob), 1)
        perplexity = 2 ** entropy

        return perplexity

    def binary_search_sigmas(self, distances, zero_index, tol=1e-10, max_iter=10000, lower=1e-20, upper=1000):
        """
        Use binary search find the best sigma which let perplexity close to user defined perplexity
        Args:
            distances: array
                distance array between point i to all others
            zero_index: int
                index of which row should be 0, since p_i_i is 0 in define
            tol: float
                threshold to control when should loop stop
            max_iter: int
                maximum iteration
            lower: float
                lower bound of search range
            upper: float
                upper bound of search range
        Returns: float
            optimal sigma value through binary search
        """
        guess_sigma = 10  # initial guess
        for i in range(max_iter):
            guess_sigma = (lower + upper) / 2
            perplexity = self.get_perplexity(distances, guess_sigma, zero_index)
            if perplexity > self.perplexity:
                upper = guess_sigma
            else:
                lower = guess_sigma
            if np.abs(guess_sigma, self.perplexity) <= tol:
                break
        return guess_sigma

    def find_optimal_sigmas(self, distance_matrix):
        """
        for each sample (each row) find optimal sigma value that close to use defined perplexity
        use binary search algorithm
        distance_matrix: array type (n_samples, n_samples)
            distance matrix between each two points
        Returns: array type (1, n_samples)
            each row optimal sigma value
        """
        sigmas = []
        n_samples = distance_matrix.shape[0]
        for i in range(n_samples):
            # find best sigma
            correct_sigma = self.binary_search_sigmas(distance_matrix[i, :], i)
            # append to sigmas list
            sigmas.append(correct_sigma)

        return np.array(sigmas)

    def get_p_joint(self, X):
        """
        Given a data matrix X, give the joint probability of gaussian distribution in high dimension
        In t-sne we define joint probability p_ij = (p_i_j + p_j+i) / (2 * n)
        p_i_j refer conditional probability P(i|j) -> when j is core point the probability of i is j's neighbor,
        n is number of sample size
        Args:
            X: array type dataset (n_samples, n_features)
        Returns:
            matrix with p_ij join probability (n_samples, n_samples)
        """
        # get euclidian distance matrix for each point
        distance_matrix = pairwise_distances(X)
        # get optimal sigma for each sample in X
        sigmas = self.find_optimal_sigmas(distance_matrix)
        # calculate conditional probability of each point P_j_i based on those optimal sigmas
        p_conditional_prob = self.get_conditional_prob_matrix(distance_matrix, sigmas)
        # convert conditional probability to join probability p_ij matrix
        p_join_prob = self.get_join_prob_matrix(p_conditional_prob)

        return p_join_prob

    def get_q_join(self, Y):
        """
        Given a matrix Y, give the joint probability of gaussian distribution in low dimension
        In t-sne we define joint probability q_ij = pow((1 + ||yi - yj||), -1) / sum(pow((1 + ||yk - yl||), -1), all)
        This is t-distribution refer as student distribution, the difference it's t-distribution has long tail
        which can solve crowding problem after high dimension projection to low dimension
        Args:
            Y: array type dataset (n_samples, n_components)
        Returns:
            q_join_prob: (n_samples, n_samples)
                matrix with q_ij join probability
            distance_matrix_inverse: (n_samples, n_samples)
                inverse of y distance matrix, which is pow((1 + ||yi - yj||), -1) this part
        """
        # get euclidian distance matrix for each point
        distance_matrix = pairwise_distances(Y)
        distance_matrix_inverse = np.power(1 + distance_matrix, -1)
        # fill all diagonal as 0 since q_i_i = 0 same as p_i_i
        np.fill_diagonal(distance_matrix_inverse, 0)
        q_join_prob = distance_matrix_inverse / np.sum(distance_matrix_inverse, axis=1)

        return q_join_prob, distance_matrix_inverse

    def get_gradient(self, p, q, y, distances):
        """
        get t-sne gradient with respect to current y. t-sne use gradient descent,
        in paper actually use stochastic gradient descent, here for easy we update all y point
        gradient = 4 * sum((p_ij - q_ij) * pow((1 + ||yi - yj||), -1) * (yi - yj), j)
        Args:
            p: array type (n_samples, n_samples)
                p join distribution from high dimension
            q: array type (n_samples, n_samples)
                q join distribution from low dimension
            y: array type (n_samples, n_components)
                current low dimension y value matrix
            distances: array type (n_samples, n_samples)
                current low dimension y distance matrix inverse this part: pow((1 + ||yi - yj||), -1), get from q_join
        Returns:
            y_gradient: array type (n_samples, n_components)
                current low dimension each y point gradient matrix
        """
        # get p, q difference (n_samples, n_samples)
        pq_diff = p - q
        # pq_diff expanded (n_samples, n_samples, 1)
        pq_diff_expanded = np.expand_dims(pq_diff, 1)
        # get y each point difference on each component (n_samples, n_samples, n_components)
        y_diff = np.expand_dims(y, 1) - np.expand_dims(y, 0)
        # expand y distance matrix for calculate (n_samples, n_samples, 1)
        distances_expanded = np.expand_dims(distances, 2)
        # get weighted y_diff (n_samples, n_samples, n_components)
        # weighted in here usually means, close point should have high weight
        y_diff_weight = y_diff * distances_expanded
        # final gradient (n_samples, n_components)
        y_gradient = 4 * np.sum(pq_diff_expanded * y_diff_weight, axis=1)

        return y_gradient

    def fit_transform(self, X):
        """
        Build t-sne model, fit into embedded space and return transformed output
        Args:
            X: array type dataset (n_samples, n_features)
        Returns:
            transformed output (n_samples, n_components)
        """
        n_sample = X.shape[0]
        # get high dimension matrix of joint probability p_ij first, this is fix value after X given
        p_join_prob = self.get_p_joint(X)
        # initial lower dimension embedded space
        y = np.random.normal(0, 0.0001, [n_sample, self.n_components])
        # initial pass value for momentum add into gradient descent, initial m1 == m2
        y_m1 = y.copy()
        y_m2 = y.copy()
        # start gradient descent loop
        for i in range(self.n_iter):
            # get low dimension join distribution matrix q_ij and y distance_matrix
            q_join_prob, y_distance_matrix = self.get_q_join(y)
            # estimate gradient with respect to current each point in y
            y_gradient = self.get_gradient(p_join_prob, q_join_prob, y, y_distance_matrix)

            # update y with gradient and momentum
            # y_t = y_t-1 - rate * y_gradient + momentum * (y_t-1 - y_t-2) t mean number of iter loop
            y = y - self.learning_rate * y_gradient + self.momentum * (y_m1 - y_m2)
            # update previous y value
            y_m2 = y_m1.copy()
            y_m1 = y.copy()

        return y
