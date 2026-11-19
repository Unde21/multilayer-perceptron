"""Fully connected layer of the multilayer perceptron."""

import numpy as np
import sys
from srcs.activations import (ACTIVATIONS)


class DenseLayer():
    """Fully connected layer: each neuron receives every input value.

    weights has the shape (n_inputs, n_neurons): one column per neuron.
    bias has the shape (n_neurons,): one value per neuron.
    """

    def __init__(self, n_inputs: int, n_neurons: int, activation_name, rng):
        """Create random weights (He initialization) and zero biases.

        The weights are drawn from a normal distribution centered on 0,
        with a standard deviation of sqrt(2 / n_inputs).
        activation_name is a key of ACTIVATIONS ("relu", "sigmoid",
        "softmax"): it gives the activation function and its derivative.
        The name is kept to save the model.
        """

        self.weights = rng.normal(loc=0, scale=(np.sqrt(2 / n_inputs)),
                                  size=(n_inputs, n_neurons))
        self.bias = np.zeros(n_neurons)

        self.activation_name = activation_name
        try:
            self.activation, self.derivative = ACTIVATIONS[activation_name]
        except KeyError:
            print(f"Error: invalid activation '{activation_name}'"
                  f" (expected: {', '.join(ACTIVATIONS)}).",
                  file=sys.stderr)
            sys.exit(1)

    def forward(self, x: np.ndarray):
        """Return activation(x @ weights + bias).

        x has the shape (patients, n_inputs) and the result
        (patients, n_neurons). x and z are kept for the backpropagation.
        """

        self.x = x
        self.z = x @ self.weights + self.bias
        output = self.activation(self.z)
        return output

    def backward(self, d_output: np.ndarray):
        """Compute the gradients of the layer from the next layer's error.

        d_output is the error on the output of this layer, of shape
        (patients, n_neurons). For the softmax output layer, it is already
        the error on z, (P - Y) / N, so the derivative step is skipped.
        Store d_weights and d_bias (same shapes as weights and bias), then
        return the error to send to the previous layer, of shape
        (patients, n_inputs). The weights are not modified here.
        """

        if self.derivative is not None:
            d_z = d_output * self.derivative(self.z)
        else:
            d_z = d_output

        self.d_weights = self.x.T @ d_z
        self.d_bias = np.sum(d_z, axis=0)
        return (d_z @ self.weights.T)
