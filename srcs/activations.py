"""Activation functions applied to the weighted sum z of a layer."""

import numpy as np


def sigmoid(score) -> np.ndarray:
    """Squash each value of score between 0 and 1.

    Work on a single number or element-wise on a numpy array. Values are
    clipped to [-500, 500] so that np.exp cannot overflow.
    """

    clipped_score = np.clip(score, -500, 500)
    sig = 1 / (1 + np.exp(-clipped_score))
    return sig


def rectified_linear_unit(z: np.ndarray) -> np.ndarray:
    """Return max(0, z) element-wise: negative values become 0.

    z is not modified: return a new array of the same shape.
    """

    relu_z = np.maximum(z, 0)
    return relu_z


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
