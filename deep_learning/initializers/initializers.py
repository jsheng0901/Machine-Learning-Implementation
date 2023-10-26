import re

import numpy as np
from deep_learning.utils.utils import glorot_uniform


class WeightInitializer:
    def __init__(self, act_fn_str, mode="glorot_uniform"):
        """
        A factory for weight initializers.

        Parameters:
        -----------
        act_fn_str : str
            The string representation for the layer activation function
        mode : str (default: 'glorot_uniform')
            The weight initialization strategy. Valid entries are {"he_normal",
            "he_uniform", "glorot_normal", glorot_uniform", "std_normal", "trunc_normal"}
        """
        if mode not in [
            "he_normal",
            "he_uniform",
            "glorot_normal",
            "glorot_uniform",
            "std_normal",
            "trunc_normal",
        ]:
            raise ValueError(f"Unrecognized initialization mode: {mode}")

        self.mode = mode
        self.act_fn = act_fn_str

        # choose initialized function according to select strategy
        if mode == "glorot_uniform":
            self._fn = glorot_uniform
        # TODO will implement other strategy later
        # elif mode == "glorot_normal":
        #     self._fn = glorot_normal
        # elif mode == "he_uniform":
        #     self._fn = he_uniform
        # elif mode == "he_normal":
        #     self._fn = he_normal
        # elif mode == "std_normal":
        #     self._fn = np.random.randn
        # elif mode == "trunc_normal":
        #     self._fn = partial(truncated_normal, mean=0, std=1)

    def _calc_glorot_gain(self):
        """
        Return the recommended gain value for the given non-linearity function.
        Values from:
        https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.calculate_gain
        """
        # other activation function share same default gain value as 1
        gain = 1.0
        act_str = self.act_fn.lower()
        if act_str == "tanh":
            gain = 5.0 / 3.0
        elif act_str == "relu":
            gain = np.sqrt(2)
        elif "leaky relu" in act_str:
            r = r"leaky relu\(alpha=(.*)\)"
            alpha = re.match(r, act_str).groups()[0]
            gain = np.sqrt(2 / 1 + float(alpha) ** 2)
        return gain

    def __call__(self, weight_shape):
        """
        Initialize weights according to the specified strategy.
        Args:
            weight_shape : tuple (n_in, n_out)
            The dimensions of the weight matrix/volume.

        Returns:
            w : array type of shape weight_shape (n_in, n_out)
            The initialized weights.
        """

        if "glorot" in self.mode:
            # get gain value according to different activation funtion in this layer
            gain = self._calc_glorot_gain()
            # get weight according to which initializer strategy choose
            w = self._fn(weight_shape, gain)
        # TODO will implement other strategy later
        # elif self.mode == "std_normal":
        #     W = self._fn(*weight_shape)
        # else:
        #     W = self._fn(weight_shape)
        return w
