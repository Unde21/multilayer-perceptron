"""Check the backpropagation against numerical gradients."""

import sys
import numpy as np
from train import (create_network, forward_pass, backward_pass, make_rng)
from srcs.data import (load_data)
from srcs.preprocessing import (fit_preprocessing, prepare_data)
from srcs.losses import (cross_entropy)
from srcs.display import (BOLD, GREEN, DIM, RED, color)

THRESHOLD = 1e-7


def forward_loss(layers: list, x_train: np.ndarray,
                 y_train: np.ndarray) -> float:
    """Return the loss of the network after a full forward pass."""

    probs = forward_pass(layers, x_train)
    loss = cross_entropy(probs, y_train)
    return loss


def gradient_check(seed: int) -> tuple[int, float]:
    """Compare every gradient of the network with its numerical value.

    Each weight and bias is moved by eps in both directions to measure the
    slope of the loss. Return the number of parameters checked and the
    largest difference with the backpropagation.
    """

    df_train = load_data("datasets/data_train.csv")
    rng = make_rng(seed)

    params = fit_preprocessing(df_train)
    x_train, y_train = prepare_data(df_train, params)
    layers = create_network(rng, x_train)

    probs = forward_pass(layers, x_train[:8])
    backward_pass(layers, probs, y_train[:8])

    eps = 1e-5
    max_value = 0
    count = 0
    for layer in layers:
        for values, grads in [(layer.weights, layer.d_weights),
                              (layer.bias, layer.d_bias)]:
            for index in np.ndindex(values.shape):
                real = grads[index]
                original = values[index]
                values[index] = original + eps
                loss_plus = forward_loss(layers, x_train[:8], y_train[:8])
                values[index] = original - eps
                loss_minus = forward_loss(layers, x_train[:8], y_train[:8])
                values[index] = original
                numeric = (loss_plus - loss_minus) / (2 * eps)
                delta = abs(real - numeric)
                count += 1
                max_value = np.maximum(max_value, delta)

    return count, max_value


def main():
    """Run the gradient check on several networks and print a report."""

    seeds = range(5)
    passed = 0

    print(color("Gradient checking", BOLD))
    print(color(f"backpropagation vs numerical, threshold {THRESHOLD:.0e}\n",
                DIM))

    for seed in seeds:
        count, max_value = gradient_check(seed)
        success = max_value < THRESHOLD
        passed += success
        status = color("[PASS]", GREEN) if success else color("[FAIL]", RED)
        print(f"  {status} seed {seed}   {count} parameters   "
              f"max diff {max_value:.2e}")

    success = passed == len(seeds)
    print(color(f"\n{passed}/{len(seeds)} passed", BOLD,
                GREEN if success else RED))
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
