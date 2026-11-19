"""Activation functions applied to the weighted sum z of a layer."""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Squash each value of z between 0 and 1.

    Work on a single number or element-wise on a numpy array. Values are
    clipped to [-500, 500] so that np.exp cannot overflow.
    """

    clipped_z = np.clip(z, -500, 500)
    sig = 1 / (1 + np.exp(-clipped_z))
    return sig


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """Return the slope of the sigmoid at each value of z.

    Computed from the sigmoid itself: sigmoid(z) * (1 - sigmoid(z)).
    """

    sig = sigmoid(z)
    d_sigmoid = sig * (1 - sig)
    return d_sigmoid


def relu(z: np.ndarray) -> np.ndarray:
    """Return max(0, z) element-wise: negative values become 0.

    z is not modified: return a new array of the same shape.
    """

    relu_z = np.maximum(z, 0)
    return relu_z


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """Return the slope of ReLU at each value of z: 1 if z > 0, else 0.

    In the backward pass, it cuts the error of the neurons that were off.
    """

    d_relu_z = (z > 0).astype(int)
    return d_relu_z


def softmax(z: np.ndarray) -> np.ndarray:
    """Turn each row of scores of z into probabilities that sum to 1.

    z has the shape (patients, classes). The maximum of each row is
    subtracted before np.exp to avoid an overflow; it does not change the
    result, since softmax only depends on the gaps between the scores.
    """

    shifted = z - np.max(z, axis=1, keepdims=True)
    exps = np.exp(shifted)
    softmax_z = exps / np.sum(exps, axis=1, keepdims=True)
    return softmax_z


# name -> (activation, derivative). The softmax has no derivative here:
# combined with the cross-entropy, its error is directly (P - Y) / N.
ACTIVATIONS = {
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "softmax": (softmax, None)
}
