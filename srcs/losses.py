"""Loss functions measuring the error of the network."""

import numpy as np


def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    """Return the categorical cross-entropy between probs and y.

    probs are the softmax outputs and y the one-hot states, both of shape
    (patients, states). For each patient, the one-hot keeps only
    -log(probability of the true state); the result is the mean over the
    patients. probs are clipped to avoid log(0).
    """

    clipped_probs = np.clip(probs, 1e-15, 1 - (1e-15))
    all_loss = y * np.log(clipped_probs)
    sum_per_line = np.sum(all_loss, axis=1)
    J = -np.mean(sum_per_line)
    return J
