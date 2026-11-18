""""""

import numpy as np
from srcs.activations import rectified_linear_unit


class DenseLayer():
    """"""

    def __init__(self, n_inputs: int, n_neurons: int, activation, rng):
        """"""

        self.weights = rng.normal(loc=0, scale=(np.sqrt(2 / n_inputs)), size=(n_inputs, n_neurons))
        self.bias = np.zeros(n_neurons)
        self.activation = activation


    def forward(self, x: np.ndarray):
        """"""

        self.x = x
        self.z = x @ self.weights + self.bias
        activ = self.activation(self.z)
        return activ




# def main ():
    
#     rng = np.random.default_rng(42)
    
#     layer = DenseLayer(30, 24, rectified_linear_unit, rng)
#     print("shape: ", layer.weights.shape)
#     print("mean: ", layer.weights.mean())
#     print("min / max: ", layer.weights.min(), layer.weights.max())
    
    
#     layer = DenseLayer(3, 2, rectified_linear_unit, rng)
#     layer.weights = np.array([[1, -1], [0, 2], [1, 0]])
#     layer.bias = np.array([-2, -1])
#     X = np.array([[1, 2, 0], [0, 1, 3]])
#     print("A1: ", layer.forward(X))


# if __name__ == "__main__":
#     main()
