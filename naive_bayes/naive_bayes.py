import numpy as np
import math


class NaiveBayes:
    """
    The Gaussian Naive Bayes classifier.
    will usually be used for text data, preform not better than LR when data size keep increasing, really relay on prior
    -----------
    """

    def __init__(self):
        # init class and each class each feature parameter
        self.classes = None
        self.parameters = []

        # initial define global x and y
        self.x = None
        self.y = None

    def fit(self, x, y):
        """
        learn x, y gaussian distribution parameters
        Args:
            x: array type dataset (n_samples, n_features)
            y: array type dataset (n_samples)

        Returns:
            None, calculate params: mean, var for each class each feature
        """

        self.x, self.y = x, y
        self.classes = np.unique(y)

        # Calculate the mean and variance of each feature for each class,
        # ex: [[{mean: , var: }, {mean: , var: }], [...]], length is class number, each class has number of feature {}
        for i, c in enumerate(self.classes):
            # Only select the rows where the label equals the given class
            x_where_c = x[np.where(y == c)]
            # init each feature parameters list
            feature_parameters_list = []
            # Add the mean and variance for each feature (column)
            for col in x_where_c.T:
                parameters = {"mean": col.mean(), "var": col.var()}
                feature_parameters_list.append(parameters)
            self.parameters.append(feature_parameters_list)

    def calculate_likelihood(self, mean, var, x):
        """
        Gaussian likelihood of the data x given mean and var
        Args:
            mean: float, mean of this feature under this class
            var: float, variance of this feature under this class
            x: float, this feature value

        Returns:
            likelihood: float
        """

        eps = 1e-4  # Added in denominator to prevent division by zero, this could be called smooth
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var + eps)))
        likelihood = coeff * exponent

        return likelihood

    def calculate_prior(self, c):
        """
        Calculate the prior of class c, ex: where class == c / total number of samples
        Args:
            c: int, class index number

        Returns:
            frequency: float, prior probability
        """

        frequency = np.mean(self.y == c)

        return frequency

    def classify(self, sample):
        """
        Classification using Bayes Rule P(Y|X) = P(X|Y)*P(Y)/P(X), or Posterior = Likelihood * Prior / Scaling Factor
        P(Y|X) - The posterior is the probability that sample x is of class y given the
                 feature values of x being distributed according to distribution of y and the prior.
        P(X|Y) - Likelihood of data X given class distribution Y.
                 Gaussian distribution (given by calculate_likelihood)
        P(Y)   - Prior (given by calculate_prior)
        P(X)   - Scales the posterior to make it a proper probability distribution.
                 This term is ignored in this implementation since it doesn't affect
                 which class distribution the sample is most likely to belong to.
        Classifies the sample as the class that results in the largest P(Y|X) (posterior)
        Args:
            sample: array type dataset (1, n_features)

        Returns:
            label: int, which class have the largest posterior probability
        """

        posteriors = []
        # Go through list of classes
        for i, c in enumerate(self.classes):
            # Initialize posterior as prior
            posterior = self.calculate_prior(c)
            # Naive assumption (independence):
            # P(x1,x2,x3|Y) = P(x1|Y)*P(x2|Y)*P(x3|Y)
            # Posterior is product of prior and likelihoods (ignoring scaling factor in here)
            # Loop through each feature params
            for feature_value, params in zip(sample, self.parameters[i]):
                # Likelihood of feature value given distribution of feature values given y
                likelihood = self.calculate_likelihood(params["mean"], params["var"], feature_value)
                posterior *= likelihood
            posteriors.append(posterior)

        # Return the class with the largest posterior probability
        label = self.classes[np.argmax(posteriors)]

        return label

    def predict(self, x):
        """
        Predict each sample label belong to which class
        Args:
            x: array type dataset (n_sample, n_features)

        Returns:
            y_pred: array type dataset (n_sample), each sample final label
        """

        y_pred = [self.classify(sample) for sample in x]  # loop through each sample

        return y_pred
