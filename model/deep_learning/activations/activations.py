from abc import abstractmethod, ABC

import numpy as np


class ActivationBase:
    def __init__(self, **kwargs):
        """
        Initialize the ActivationBase object
        """
        super().__init__()

    def __call__(self, z):
        """
        Apply the activation function to an input
        Args:
            z: array type (n_in, n_out)

        Returns:
            output: array type same as input z (n_in, n_out)
        """

        if z.ndim == 1:
            z = z.reshape(1, -1)
        output = self.fn(z)
        return output

    @abstractmethod
    def fn(self, z):
        """
        Apply the activation function to an input
        """
        raise NotImplementedError

    @abstractmethod
    def grad(self, x, **kwargs):
        """
        Compute the gradient of the activation function of the input
        """
        raise NotImplementedError


class Sigmoid(ActivationBase, ABC):
    def __init__(self):
        """
        A logistic sigmoid activation function.
        """
        super().__init__()

    def __str__(self):
        """
        Return a string representation of the activation function
        """
        return "Sigmoid"

    def fn(self, z):
        r"""
        Evaluate the logistic sigmoid, :math:`\sigma`, on the elements of input `z`.

        .. math::

            \sigma(x_i) = \frac{1}{1 + e^{-x_i}}
        """
        output = 1 / (1 + np.exp(-z))

        return output

    def gradient(self, x):
        r"""
        Evaluate the first derivative of the logistic sigmoid on the elements of `x`.

        .. math::

            \frac{\partial \sigma}{\partial x_i} = \sigma(x_i) (1 - \sigma(x_i))
        """
        fn_x = self.fn(x)
        gradient = fn_x * (1 - fn_x)

        return gradient

    def hess(self, x):
        r"""
        Evaluate the second derivative of the logistic sigmoid on the elements of `x`.

        .. math::

            \frac{\partial^2 \sigma}{\partial x_i^2} =
                \frac{\partial \sigma}{\partial x_i} (1 - 2 \sigma(x_i))
        """
        fn_x = self.fn(x)
        hess = fn_x * (1 - fn_x) * (1 - 2 * fn_x)

        return hess
