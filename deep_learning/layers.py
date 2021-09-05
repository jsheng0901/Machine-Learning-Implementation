import math
import numpy as np
import copy


class Layer(object):

    def set_input_shape(self, shape):
        """ Sets the shape that the layer expects of the input in the forward"""
        self.input_shape = shape

    def layer_name(self):
        """ The name of the layer. Used in model summary. """
        return self.__class__.__name__

    def parameters(self):
        """ The number of trainable parameters used by the layer """
        return 0

    def forward_pass(self, X, training):
        """ propagates the signal forward in the network """
        raise NotImplementedError()

    def backward_pass(self, accum_grad):
        """
        propagates the accumulated gradient backwards in the network.
        If the has trainable weights then these weights are also tuned in this method.
        As input (accum_grad) it receives the gradient with respect to the output of the layer and
        returns the gradient with respect to the output of the previous layer.
        """
        raise NotImplementedError()

    def output_shape(self):
        """ The shape of the output produced by forward_pass """
        raise NotImplementedError()


class Dense(Layer):
    """
    A fully-connected NN layer.
    Parameters:
    -----------
    n_units: int
        The number of neurons in the layer.
    input_shape: tuple
        The expected input shape of the layer. For dense layers a single digit specifying
        the number of features of the input. Must be specified if it is the first layer in
        the network.
    """
    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape      # ex: (8, 300), 8 is batch size, 300 is previous output size
        self.n_units = n_units              # will be the output size ex: (8, 300) * (300, 512) --> (8, 512)
        self.trainable = True
        self.W = None
        self.w0 = None

    def initialize(self, optimizer):
        """Initialize the weights with uniform distribution, (1 / sqrt(m), - 1 / sqrt(m)) with m is input size"""
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))
        self.w0 = np.zeros((1, self.n_units))           # bias initialize with all zero
        # Weight optimizers
        self.W_opt = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)

    def forward_pass(self, X, training=True):
        """ fully connected layer forward propagate, output: w * x + w0 """
        self.layer_input = X
        return X.dot(self.W) + self.w0

    def backward_pass(self, accum_grad):
        # Save weights used during forwards pass
        W = self.W

        if self.trainable:
            # Calculate gradient layer weights, from upstream gradient accum_grad
            grad_w = self.layer_input.T.dot(accum_grad)     # X * accum_grad
            grad_w0 = np.sum(accum_grad, axis=0, keepdims=True)     # sum of accum_grad

            # Update the layer weights according to different optimizer
            self.W = self.W_opt.update(self.W, grad_w)
            self.w0 = self.w0_opt.update(self.w0, grad_w0)

        # Return accumulated gradient for next layer
        # Calculated based on the weights used during the forward pass
        accum_grad = accum_grad.dot(W.T)
        return accum_grad

    def parameters(self):
        return np.prod(self.W.shape) + np.prod(self.w0.shape)