from abc import abstractmethod, ABC

import numpy as np
from deep_learning.initializers.initializers import WeightInitializer


class Layer:
    def __init__(self, optimizer=None):
        """
        An abstract base class inherited by all neural network layers.

        Parameters:
        -----------
        optimizer : str, object, or None
            The optimization strategy to use when performing gradient updates within the update method.
            If None, use the SGD optimizer with default parameters. Default is None.


        Attributes:
        -----------
        x : list
            Running list of inputs to the forward method
        gradients: dict
            Dictionary of loss gradients with regard to the layer parameters
        hyperparameters : dict
            Dictionary of layer hyperparameters
        parameters: dict
            Dictionary of layer parameters
        derived_variables : dict
            Dictionary of any intermediate values computed during forward/backward propagation.
        trainable: boolean
            Indicate should be trained or not. Default is true
        """
        self.x = []
        self.act_fn = None
        self.trainable = True
        self.optimizer = optimizer

        self.gradients = {}
        self.parameters = {}
        self.derived_variables = {}
        self.hyperparameters = {}

    # def set_input_shape(self, shape):
    #     """ Sets the shape that the layer expects of the input in the forward"""
    #     self.input_shape = shape

    @abstractmethod
    def _init_params(self, **kwargs):
        """
        Initial this layer params before training.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, z, **kwargs):
        """
        Perform a forward pass through the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def backward(self, out, **kwargs):
        """
        Perform a backward pass through the layer
        """
        raise NotImplementedError

    def layer_name(self):
        """
        The name of the layer. Used in model summary.
        """
        return self.__class__.__name__

    def parameters(self):
        """
        The number of trainable parameters used by the layer
        """
        return 0

    # def forward_pass(self, X, training):
    #     """ propagates the signal forward in the network """
    #     raise NotImplementedError()
    #
    # def backward_pass(self, accum_grad):
    #     """
    #     propagates the accumulated gradient backwards in the network.
    #     If the has trainable weights then these weights are also tuned in this method.
    #     As input (accum_grad) it receives the gradient with respect to the output of the layer and
    #     returns the gradient with respect to the output of the previous layer.
    #     """
    #     raise NotImplementedError()

    # def output_shape(self):
    #     """ The shape of the output produced by forward_pass """
    #     raise NotImplementedError()


class FullyConnected(Layer, ABC):
    def __init__(self, n_out, act_fn=None, init="glorot_uniform", optimizer=None):
        """
        A fully-connected (dense) layer.

        Notes:
        ------
        A fully connected layer computes the function
        
            y = f(wx + b)

        where f is the activation nonlinear, w and b are parameters of the layer, 
        and x is the minibatch of input examples.

        Parameters:
        -----------
        n_out : int
            The dimensionality of the layer output
        act_fn : str, object, or None
            The element-wise output nonlinear used in computing y. 
            If None, use the identity function f(x) = x. Default is None.
        init : str, from {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
            The weight initialization strategy. Default is 'glorot_uniform'.
        optimizer : str, object, or None
            The optimization strategy to use when performing gradient updates within the update method. 
            If None, use the SGD optimizer with default parameters. Default is None.

        Attributes:
        -----------
        n_in: int
            The dimensionality of the layer input
        is_initialized: boolean
            Indicate if it's already initialized
        """
        super().__init__(optimizer)

        # ex: x -> (8, 300), 8 is batch size, 300 is previous output size
        # run (8, 300) * (300, 512) --> (8, 512) in this layer
        # n_in -> 300   n_out -> 512
        self.init = init
        self.n_in = None
        self.n_out = n_out
        self.act_fn = act_fn()
        self.parameters = {"w": None, "b": None}
        self.is_initialized = False

    def _init_params(self):
        # perform input initial method, create initializer object
        init_weights = WeightInitializer(str(self.act_fn), mode=self.init)

        # ex: b -> (1, 512)
        b = np.zeros((1, self.n_out))
        # ex: w -> (300, 512)
        # since we implement __call__ inside this object, so we could call this instance like function
        w = init_weights((self.n_in, self.n_out))

        # initial layer parameters into prepared attributes
        self.parameters = {"w": w, "b": b}
        self.derived_variables = {"z": []}
        self.gradients = {"w": np.zeros_like(w), "b": np.zeros_like(b)}
        self.is_initialized = True

    def _fwd(self, x):
        """
        Actual computation of forward pass.
        Args:
            x: array type dataset (n_samples, n_in)

        Returns:
            y: array type dataset (n_samples, n_out)
        """
        # get layer parameters
        w = self.parameters["w"]
        b = self.parameters["b"]

        # run function, ex: (8, 300) * (300, 512) + (1, 512) -> (8, 512)
        # z -> (8, 512)
        z = x.dot(w) + b
        # y -> (8, 512) activation function don't change size
        y = self.act_fn(z)

        return y, z

    def _bwd(self, dy, x, z):
        """
        Actual computation of gradient of the loss of x, w, and b.
        Each layer derivative of x, w, b only related to same layer linear out z, layer input x, layer parameters w,
        and the gradient of lose function of same layer output y.
        Args:
            dy: array type dataset (n_samples, n_out)
            x: array type dataset (n_samples, n_in)
            z: array type dataset (n_samples, n_out)

        Returns:
            dx: array type dataset (n_samples, n_in)
            dw: array type dataset (n_in, n_out)
            db: array type dataset (1, n_out)
        """
        # get this layer parameters
        w = self.parameters["w"]

        # multiply not dot. -> (n_samples, n_out)
        # ex: dy -> (8, 512), z -> (8, 512), dz -> (8, 512)
        dz = np.multiply(dy, self.act_fn.grad(z))

        # (n_samples, n_out) dot (n_int, n_out).T -> (n_samples, n_in)
        # ex: (8, 512) * (300, 512).T -> (8, 300) same output size as input x (8, 300)
        dx = dz.dot(w.T)

        # (n_samples, n_in).T dot (n_samples, n_out) -> (n_in, n_out)
        # ex: (8, 300).T * (8, 512).T -> (300, 512)
        dw = x.T.dot(dz)

        # (n_samples, n_out) sum along all samples on same output column -> (1, n_out)
        # ex: (8, 512) -> (1, 512)
        db = dz.sum(axis=0, keepdims=True)

        return dx, dw, db

    def forward(self, x, retain_derived=True):
        """
        Compute the layer output on a single minibatch.

        Parameters:
        -----------
        x : array type dataset (n_samples, n_in)
            Layer input, representing the n_in -> dimensional features for a
            minibatch of n_samples -> samples.
        retain_derived : boolean
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If False, this suggests the layer
            will not be expected to backprop through this input. Default is True.

        Returns:
        --------
        y : array type dataset (n_samples, n_out)
            Layer output for each of the n_samples samples.
        """
        # if not initialized, then perform initialized first
        if not self.is_initialized:
            self.n_in = x.shape[1]
            self._init_params()

        # run real forward logic
        y, z = self._fwd(x)

        # store variables for later backprop
        if retain_derived:
            # store input x
            self.x.append(x)
            # store linea output z
            self.derived_variables["z"].append(z)

        return y

    def backward(self, dLdy, retain_grads=True):
        """
        Backprop from layer outputs to inputs.

        Parameters：
        -----------
        dLdy : array type dataset (n_samples, n_out) or list of arrays
            The gradient(s) of the loss of the layer output(s) y.
        retain_grads : boolean
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is True.

        Returns
        -------
        dLdx : array type dataset (n_samples, n_in) or list of arrays
            The gradient of the loss of the layer input(s) x.
        """
        # check if layer is frozen to not allow backprop
        assert self.trainable, "Layer is frozen"
        if not isinstance(dLdy, list):
            dLdy = [dLdy]

        # store each time dx
        dxs = []
        # make self.x copy
        x = self.x
        # for each pair loss gradient of layer output, x and layer linear output z
        # ex: zip((n_samples, n_out), (n_samples, n_in),  (n_samples, n_out))
        for dy, x, z in zip(dLdy, x, self.derived_variables['z']):
            dx, dw, db = self._bwd(dy, x, z)
            dxs.append(dx)

            # store this layer gradients of w and b
            if retain_grads:
                self.gradients["w"] += dw
                self.gradients["b"] += db

        # get output the gradient of the loss of this layer input x
        dLdx = dxs[0] if len(x) == 1 else dxs

        return dLdx

    @property
    def hyperparameters(self):
        """
        Return a dictionary containing the layer hyperparameters.
        """
        return {
            "layer": "FullyConnected",
            "init": self.init,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "act_fn": str(self.act_fn),
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }

    def parameters(self):
        """
        Return total parameters in this layer
        """
        return np.prod((self.n_in, self.n_out)) + self.n_out

# class Dense1(Layer, ABC):
#     """
#     A fully-connected NN layer.
#     Parameters:
#     -----------
#     n_units: int
#         The number of neurons in the layer.
#     input_shape: tuple
#         The expected input shape of the layer. For dense layers a single digit specifying
#         the number of features of the input. Must be specified if it is the first layer in
#         the network.
#     """
#
#     def __init__(self, n_units, input_shape=None):
#         self.layer_input = None
#         self.input_shape = input_shape  # ex: (8, 300), 8 is batch size, 300 is previous output size
#         self.n_units = n_units  # will be the output size ex: (8, 300) * (300, 512) --> (8, 512)
#         self.trainable = True
#         self.W = None
#         self.w0 = None
#
#     def initialize(self, optimizer):
#         """Initialize the weights with uniform distribution, (1 / sqrt(m), - 1 / sqrt(m)) with m is input size"""
#         limit = 1 / math.sqrt(self.input_shape[0])
#         self.W = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))
#         self.w0 = np.zeros((1, self.n_units))  # bias initialize with all zero
#         # Weight optimizers
#         self.W_opt = copy.copy(optimizer)
#         self.w0_opt = copy.copy(optimizer)
#
#     def forward_pass(self, X, training=True):
#         """ fully connected layer forward propagate, output: w * x + w0 """
#         self.layer_input = X
#         return X.dot(self.W) + self.w0
#
#     def backward_pass(self, accum_grad):
#         # Save weights used during forwards pass
#         W = self.W
#
#         if self.trainable:
#             # Calculate gradient layer weights, from upstream gradient accum_grad
#             grad_w = self.layer_input.T.dot(accum_grad)  # X * accum_grad
#             grad_w0 = np.sum(accum_grad, axis=0, keepdims=True)  # sum of accum_grad
#
#             # Update the layer weights according to different optimizer
#             self.W = self.W_opt.update(self.W, grad_w)
#             self.w0 = self.w0_opt.update(self.w0, grad_w0)
#
#         # Return accumulated gradient for next layer
#         # Calculated based on the weights used during the forward pass
#         accum_grad = accum_grad.dot(W.T)
#         return accum_grad
#
#     def parameters(self):
#         return np.prod(self.W.shape) + np.prod(self.w0.shape)
