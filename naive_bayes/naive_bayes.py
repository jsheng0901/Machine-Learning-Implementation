import numpy as np
import math


class NaiveBayes:
    """
    The Gaussian Naive Bayes classifier.
    usually will be used for text data, preform not good than LR when data size keep increasing, really relay on prior
    """
    def fit(self, X, y):
        """learn X, y gaussian distribution parameters"""
        self.X, self.y = X, y
        self.classes = np.unique(y)
        self.parameters = []
        # Calculate the mean and variance of each feature for each class,
        # ex: [[{mean: , var: }, {mean: , var: }], [...]], length is class number, each class has number of feature {}
        for i, c in enumerate(self.classes):
            # Only select the rows where the label equals the given class
            X_where_c = X[np.where(y == c)]
            self.parameters.append([])
            # Add the mean and variance for each feature (column)
            for col in X_where_c.T:
                parameters = {"mean": col.mean(), "var": col.var()}
                self.parameters[i].append(parameters)

    def calculate_likelihood(self, mean, var, x):
        """ Gaussian likelihood of the data x given mean and var """
        eps = 1e-4  # Added in denominator to prevent division by zero, this could be called smooth
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var + eps)))
        return coeff * exponent

    def calculate_prior(self, c):
        """ Calculate the prior of class c, ex: where class == c / total number of samples """
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
        """
        posteriors = []
        # Go through list of classes
        for i, c in enumerate(self.classes):
            # Initialize posterior as prior
            posterior = self.calculate_prior(c)
            # Naive assumption (independence):
            # P(x1,x2,x3|Y) = P(x1|Y)*P(x2|Y)*P(x3|Y)
            # Posterior is product of prior and likelihoods (ignoring scaling factor in here)
            for feature_value, params in zip(sample, self.parameters[i]):   # loop through each feature params
                # Likelihood of feature value given distribution of feature values given y
                likelihood = self.calculate_likelihood(params["mean"], params["var"], feature_value)
                posterior *= likelihood
            posteriors.append(posterior)
        # Return the class with the largest posterior probability
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        """ Predict the class labels of the samples in X """
        y_pred = [self.classify(sample) for sample in X]    # loop through each sample
        return y_pred