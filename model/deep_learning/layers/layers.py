from abc import abstractmethod, ABC

import numpy as np
from model.deep_learning.initializers.initializers import WeightInitializer


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


class RNNCell(Layer, ABC):
    def __init__(self, n_out, act_fn="Tanh", init="glorot_uniform", optimizer=None):
        r"""
        A single step of a vanilla RNN.

        Notes
        -----
        At timestep `t`, the vanilla RNN cell computes, without each timestep output y = softmax(V * A_t)

        .. math::

            \mathbf{Z}^{(t)}  &=
                \mathbf{W}_{ax} \mathbf{X}^{(t)} + \mathbf{b}_{ax} +
                    \mathbf{W}_{aa} \mathbf{A}^{(t-1)} + \mathbf{b}_{aa} \\
            \mathbf{A}^{(t)}  &=  f(\mathbf{Z}^{(t)})

        where

        - :math:`\mathbf{X}^{(t)}` is the input at time `t`
        - :math:`\mathbf{A}^{(t)}` is the hidden state at timestep `t`
        - `f` is the layer activation function
        - :math:`\mathbf{W}_{ax}` and :math:`\mathbf{b}_{ax}` are the weights
          and bias for the input to hidden layer
        - :math:`\mathbf{W}_{aa}` and :math:`\mathbf{b}_{aa}` are the weights
          and biases for the hidden to hidden layer

        Parameters:
        -----------
        n_out : int
            The dimension of a single hidden state same as output on a given timestep
        act_fn : str, Activation object, or None
            The activation function for computing ``A[t]``. Default is `'Tanh'`. Recommend `'Relu'`.
        init : {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
            The weight initialization strategy. Default is `'glorot_uniform'`.
        optimizer : str, Optimizer object, or None
            The optimization strategy to use when performing gradient updates
            within the update method. If None, use the SGD optimizer with default parameters.
            Default is None.
        """
        super().__init__(optimizer)

        # ex: vocabulary size C = 8000 -> n_in = 8000 and a hidden layer size H = 100 -> n_out = 100
        # for each timestep x, we have:
        # x -> (1 * 8000), wx -> (8000, 100), bx -> (100, 1), wa -> (100, 100), ba -> (100, 1)
        # x -> (1, 100) = a -> (1, 100)
        # hidden layer output A is one dimension vector stand for this timestep x_t memory
        # for all input timesteps, we will have (t,
        self.init = init
        self.n_in = None
        self.n_out = n_out
        self.n_timesteps = None
        # self.act_fn = ActivationInitializer(act_fn)()
        self.act_fn = act_fn()
        self.parameters = {"w_aa": None, "w_ax": None, "ba": None, "bx": None}
        self.is_initialized = False

    def _init_params(self):
        # perform input initial method, create initializer object
        init_weights = WeightInitializer(str(self.act_fn), mode=self.init)

        # example from above:
        # wx -> (8000, 100), wa -> (100, 100), bx -> (100, 1), ba -> (100, 1)
        # bias dimension will be same as output dimension size, with one as expanded
        # since bias is performance on each feature in here it's hidden layer units
        wx = init_weights((self.n_in, self.n_out))
        wa = init_weights((self.n_out, self.n_out))
        ba = np.zeros((self.n_out, 1))
        bx = np.zeros((self.n_out, 1))

        # initial layer parameters into prepared attributes
        self.parameters = {"wa": wa, "wx": wx, "ba": ba, "bx": bx}

        self.gradients = {
            "wa": np.zeros_like(wa),
            "wx": np.zeros_like(wx),
            "ba": np.zeros_like(ba),
            "bx": np.zeros_like(bx),
        }

        self.derived_variables = {
            "a": [],
            "z": [],
            "n_timesteps": 0,
            "current_step": 0,
            "dLda_accumulator": None,
        }

        self.is_initialized = True

    def forward(self, x_t, retain_derived=True):
        """
        Compute the network output for a single timestep x_t.

        Parameters:
        -----------
        x_t : :py:class:`ndarray <numpy.ndarray>` of shape `(n_samples, n_in)`
            Input at timestep `t` consisting of `n_samples` samples each of
            dimensionality `n_in`.
        retain_derived : boolean
            Whether to retain the variables calculated during the forward pass
            for use later during backprop. If False, this suggests the layer
            will not be expected to backprop through this input. Default is True.

        Returns:
        --------
        a_t: :py:class:`ndarray <numpy.ndarray>` of shape `(n_samples, n_out)`
            The value of the hidden state at timestep `t` for each of the
            `n_samples` samples.
        """
        # if not initialized, then perform initialized first
        if not self.is_initialized:
            self.n_in = x_t.shape[1]
            self._init_params()

        # increment timestep
        self.derived_variables["n_timesteps"] += 1
        self.derived_variables["current_step"] += 1

        # Retrieve parameters
        ba = self.parameters["ba"]
        bx = self.parameters["bx"]
        wx = self.parameters["wx"]
        wa = self.parameters["wa"]

        # initialize the hidden state to zero
        # since first timestep no previous hidden state output pass in
        a_s = self.derived_variables["a"]
        if len(a_s) == 0:
            n_samples, n_in = x_t.shape
            a_0 = np.zeros((n_samples, self.n_out))
            a_s.append(a_0)

        # compute next hidden state, run function have:
        # ex: input part: [x_t -> (8, 8000)] * [wx -> (8000, 100)] + [bx -> (100, 1)] +
        # hidden part: [a_s -> (8, 100)] * [wa -> (100, 100)] + [bx -> (100, 1)]
        # here we're broadcasting ba, bx on n_samples dimension, from (1, 100) to (8, 100)
        # then z_t -> (8, 100) = a_t -> (8, 100), since activation function not change dimension
        z_t = (x_t.dot(wx) + bx.T) + (a_s[-1].dot(wa) + ba.T)
        a_t = self.act_fn(z_t)

        self.derived_variables["z"].append(z_t)
        self.derived_variables["a"].append(a_t)

        # store intermediate variables
        self.x.append(x_t)
        return a_t

    def backward(self, dLdat, retain_grads=True):
        """
        Backprop for a single timestep.

        Parameters:
        -----------
        dLdat : :py:class:`ndarray <numpy.ndarray>` of shape `(n_samples, n_out)`
            The gradient of the loss of the layer outputs (hidden states) at timestep `t`.
        retain_grads : boolean
            Whether to include the intermediate parameter gradients computed
            during the backward pass in the final parameter update. Default is True.

        Returns:
        --------
        dLdxt : :py:class:`ndarray <numpy.ndarray>` of shape `(n_samples, n_in)`
            The gradient of the loss of the layer inputs at timestep `t`.
        """
        assert self.trainable, "Layer is frozen"

        #  decrement current step
        self.derived_variables["current_step"] -= 1

        # extract context variables
        z_s = self.derived_variables["z"]
        a_s = self.derived_variables["a"]
        t = self.derived_variables["current_step"]
        da_acc = self.derived_variables["dLda_accumulator"]

        # initialize accumulator，for first hidden state
        if da_acc is None:
            da_acc = np.zeros_like(a_s[0])

        # get network weights for gradient calcs
        wx = self.parameters["wx"]
        wa = self.parameters["wa"]

        # compute gradient components at timestep t
        da = dLdat + da_acc
        # dLdz = dLdat * datdz
        dz = da * self.act_fn.grad(z_s[t])
        # ex: [dz -> (8, 100)] * [wx -> (8000, 100)] = dxt -> (8, 8000)
        dxt = dz.dot(wx.T)

        # update parameter gradients with signal from current step
        # ex: [a_s -> (8, 100)].T * [dz -> (8, 100)] = wa -> (100, 100)
        # dLdwa = dLdat * datdz * dzdwa
        # (dLdat * datdz) -> dz, dzdwa -> a_s[t]
        # here we only show one timestep, so we need keep sum through t
        self.gradients["wa"] += a_s[t].T.dot(dz)
        # ex: [x_t -> (8, 8000)].T * [dz -> (8, 100)] = wx -> (8000, 100)
        self.gradients["wx"] += self.x[t].T.dot(dz)
        # ex: [dz -> (8, 100)] -> sum on 8 = ba -> (1, 100).T -> (100, 1)
        self.gradients["ba"] += dz.sum(axis=0, keepdims=True).T
        # same as above ba
        self.gradients["bx"] += dz.sum(axis=0, keepdims=True).T

        # update accumulator variable for hidden state
        # [dz -> (8, 100)] * [wa -> (100, 100)].T = dLda -> (8, 100)
        self.derived_variables["dLda_accumulator"] = dz.dot(wa.T)

        return dxt

    @property
    def hyperparameters(self):
        """
        Return a dictionary containing the layer hyperparameters.
        """
        return {
            "layer": "RNNCell",
            "init": self.init,
            "n_in": self.n_in,
            "n_out": self.n_out,
            "act_fn": str(self.act_fn),
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }


class Embedding(Layer, ABC):
    def __init__(
        self, n_out, vocab_size, pool=None, init="glorot_uniform", optimizer=None,
    ):
        """
        An embedding layer.

        Notes:
        ------
        Equations::

            Y = W[x]

        This layer must be the first in a neural network as the gradients
        do not get passed back through to the inputs.

        Parameters:
        -----------
        n_out : int
            The dimensionality of the embeddings
        vocab_size : int
            The total number of items in the vocabulary. All integer indices
            are expected to range between 0 and `vocab_size - 1`.
        pool : {'sum', 'mean', None}
            If not None, apply this function to the collection of `n_in`
            encodings in each example to produce a single, pooled embedding.
            Default is None.
        init : {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'}
            The weight initialization strategy. Default is `'glorot_uniform'`.
        optimizer : str, :doc:`Optimizer <numpy_ml.neural_nets.optimizers>` object, or None
            The optimization strategy to use when performing gradient updates
            within the :meth:`update` method.  If None, use the :class:`SGD
            <numpy_ml.neural_nets.optimizers.SGD>` optimizer with
            default parameters. Default is None.

        Attributes:
        -----------
        x : list
            Running list of inputs to the :meth:`forward <numpy_ml.neural_nets.LayerBase.forward>` method since the last call to :meth:`update <numpy_ml.neural_nets.LayerBase.update>`. Only updated if the `retain_derived` argument was set to True.
        gradients : dict
            Dictionary of loss gradients with regard to the layer parameters.
        parameters : dict
            Dictionary of layer parameters
        hyperparameters : dict
            Dictionary of layer hyperparameters
        derived_variables : dict
            Dictionary of any intermediate values computed during
            forward/backward propagation.
        """
        super().__init__(optimizer)
        fstr = "'pool' must be either 'sum', 'mean', or None but got '{}'"
        assert pool in ["sum", "mean", None], fstr.format(pool)

        self.init = init
        self.pool = pool
        self.n_out = n_out
        self.vocab_size = vocab_size
        self.parameters = {"w": None}
        self.is_initialized = False
        self._init_params()

    def _init_params(self):
        init_weights = WeightInitializer("Affine(slope=1, intercept=0)", mode=self.init)

        # ex: v = 1000, embeddings size = 300, w -> (1000, 300)
        w = init_weights((self.vocab_size, self.n_out))

        self.parameters = {"w": w}
        self.derived_variables = {}
        self.gradients = {"w": np.zeros_like(w)}
        self.is_initialized = True

    @property
    def hyperparameters(self):
        """
        Return a dictionary containing the layer hyperparameters.
        """
        return {
            "layer": "Embedding",
            "init": self.init,
            "pool": self.pool,
            "n_out": self.n_out,
            "vocab_size": self.vocab_size,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }

    def lookup(self, ids):
        """
        Return the embeddings associated with the IDs in `ids`.

        Parameters:
        -----------
        ids : :py:class:`ndarray <numpy.ndarray>` of shape (`M`,)
            An array of `M` IDs to retrieve embeddings for.

        Returns:
        --------
        embeddings : :py:class:`ndarray <numpy.ndarray>` of shape (`M`, `n_out`)
            The embedding vectors for each of the `M` IDs.
        """
        return self.parameters["w"][ids]


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
