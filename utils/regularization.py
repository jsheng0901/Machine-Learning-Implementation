import numpy as np


class Lasso:
    """
    Lasso penalty, gamma * abs(params)
    """
    def loss(self, gamma, params):
        norm_params = np.linalg.norm(params, ord=1)
        penalty = gamma * norm_params

        return penalty

    def grad(self, gamma, params):
        gradient_penalty = gamma * np.sign(params)

        return gradient_penalty


class Ridge:
    """
    Ridge penalty, (gamma / 2) * square(params)
    """
    def loss(self, gamma, params):
        norm_params = np.linalg.norm(params, ord=2)
        penalty = (gamma / 2) * (norm_params ** 2)

        return penalty

    def grad(self, gamma, params):
        gradient_penalty = gamma * params

        return gradient_penalty
