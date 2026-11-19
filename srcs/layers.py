"""Fully connected layer of the multilayer perceptron."""

import numpy as np


class DenseLayer():
    """Fully connected layer: each neuron receives every input value.

    weights has the shape (n_inputs, n_neurons): one column per neuron.
    bias has the shape (n_neurons,): one value per neuron.
    """

    def __init__(self, n_inputs: int, n_neurons: int, activation, rng):
        """Create random weights (He initialization) and zero biases.

        The weights are drawn from a normal distribution centered on 0,
        with a standard deviation of sqrt(2 / n_inputs).
        activation is the function applied to the weighted sum.
        """

        self.weights = rng.normal(loc=0, scale=(np.sqrt(2 / n_inputs)),
                                  size=(n_inputs, n_neurons))
        self.bias = np.zeros(n_neurons)
        self.activation = activation

    def forward(self, x: np.ndarray):
        """Return activation(x @ weights + bias).

        x has the shape (patients, n_inputs) and the result
        (patients, n_neurons). x and z are kept for the backpropagation.
        """

        self.x = x
        self.z = x @ self.weights + self.bias
        activ = self.activation(self.z)
        return activ
