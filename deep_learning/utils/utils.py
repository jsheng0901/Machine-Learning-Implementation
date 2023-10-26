import numpy as np


def calc_fan(weight_shape):
    """
    Compute the fan-in and fan-out for a weight matrix/volume.

    Parameters:
    -----------
    weight_shape : tuple
        The dimensions of the weight matrix/volume. The final 2 entries must be n_in, n_out.

    Returns:
    --------
    fan_in: int
        The number of input units in the weight tensor
    fan_out: int
        The number of output units in the weight tensor
    """
    if len(weight_shape) == 2:
        fan_in, fan_out = weight_shape
    elif len(weight_shape) in [3, 4]:
        in_ch, out_ch = weight_shape[-2:]
        kernel_size = np.prod(weight_shape[:-2])
        fan_in, fan_out = in_ch * kernel_size, out_ch * kernel_size
    else:
        raise ValueError(f"Unrecognized weight dimension: {weight_shape}")
    return fan_in, fan_out


def glorot_uniform(weight_shape, gain=1.0):
    """
    Initialize network weights w using the Glorot uniform initialization
    strategy.

    Notes:
    ------
    The Glorot uniform initialization strategy initializes weights using draws
    from ``Uniform(-b, b)`` where:

    .. math::

        b = \\text{gain} \sqrt{\\frac{6}{\\text{fan_in} + \\text{fan_out}}}

    The motivation for Glorot uniform initialization is to choose weights to ensure that the variance of the layer
    outputs are approximately equal to the variance of its inputs. Since if layer become deeper, the variance of w
    gradient will explode or vanish, then gradients could vanish or explode. So keep input output variance close will
    increase coverage and avoid gradient vanish and explode.
    link:
    1. https://chaithanyakumars.medium.com/understanding-the-difficulty-of-training-deep-feed-forward-neural-networks-e4545690b4d5
    2. http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

    This initialization strategy was primarily developed for deep networks with
    tanh and logistic sigmoid non-linearity activation function.

    Parameters:
    -----------
    weight_shape: tuple
        The dimensions of the weight matrix/volume.
    gain: float, default is 1
        The value for initial gain value, default is 1 since most activation function default is 1
    Returns
    -------
    w : array type of shape weight_shape
        The initialized weights.
    """
    # get in and out units number
    fan_in, fan_out = calc_fan(weight_shape)
    # get b value according to math formular
    b = gain * np.sqrt(6 / (fan_in + fan_out))
    # get weight from uniform distribution
    w = np.random.uniform(-b, b, size=weight_shape)
    return w
