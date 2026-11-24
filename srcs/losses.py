"""Loss and accuracy, measuring the quality of the predictions."""

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


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    """Return the proportion of patients given their true state.

    probs are the softmax outputs and y the one-hot states, both of shape
    (patients, states). The predicted state of a patient is the column of
    its highest probability. The result is between 0 and 1; it is only
    displayed, the network does not learn from it.
    """

    max_probs = np.argmax(probs, axis=1)
    max_y = np.argmax(y, axis=1)
    acc = (max_probs == max_y).mean()
    return acc
