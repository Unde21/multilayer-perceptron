"""Train a multilayer perceptron and evaluate it on a validation set."""

from srcs.data import (load_data)
from srcs.preprocessing import (fit_preprocessing, prepare_data)
from srcs.layers import DenseLayer
from srcs.losses import (cross_entropy, accuracy)
from srcs.display import (BOLD, DIM, color)
import sys
import numpy as np
import argparse


def make_rng(seed: int):
    """Return the random generator shared by the whole training.

    The same seed gives the same weights initialization and shuffles.
    """

    rng = np.random.default_rng(seed)
    return rng


def create_network(rng, x_train: np.ndarray) -> list:
    """Build the layers of the network, from input to output.

    Two ReLU hidden layers of 24 neurons, then a softmax output layer
    with one neuron per state. The input size is the number of features
    of x_train.
    """

    layers = [
        DenseLayer(x_train.shape[1], 24, "relu", rng),
        DenseLayer(24, 24, "relu", rng),
        DenseLayer(24, 2, "softmax", rng),
    ]

    return layers


def forward_pass(layers: list, values: np.ndarray) -> np.ndarray:
    """Feedforward: pass values through each layer, in order.

    The output of a layer is the input of the next one. Return the
    probabilities of the output layer, of shape (patients, states).
    """

    for layer in layers:
        values = layer.forward(values)

    return values


def backward_pass(layers: list, probs: np.ndarray, y: np.ndarray) -> None:
    """Backpropagation: send the error back through each layer, in reverse.

    The error starts at (probs - y) / patients on the output layer, then
    each layer receives the error returned by the next one. Every layer
    stores its gradients; the weights are not modified here.
    """

    error = (probs - y) / len(y)
    for layer in reversed(layers):
        error = layer.backward(error)


def parse_args() -> argparse.Namespace:
    """Parse and check the command line arguments.

    Exit with an argparse error if a value is out of range.
    """

    parser = argparse.ArgumentParser(
        description="Train a multilayer perceptron on the training set,"
                    " evaluate it\non the validation set and save the model"
                    " to weights.json.",
        epilog="examples:\n"
               "  python3 train.py datasets/data_train.csv"
               " datasets/data_valid.csv\n"
               "  python3 train.py datasets/data_train.csv"
               " datasets/data_valid.csv --batch 8\n"
               "  python3 train.py datasets/data_train.csv"
               " datasets/data_valid.csv --epochs 100"
               " --learning_rate 0.03 --seed 42",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("dataset_train",
                        help="path to the training .csv file.")
    parser.add_argument("dataset_valid",
                        help="path to the validation .csv file.")
    parser.add_argument("--batch", type=int, default=None,
                        metavar="SIZE",
                        help="number of patients per update of the weights:\n"
                             "  (none)  batch gradient descent, all patients"
                             " (default)\n"
                             "  1       stochastic gradient descent (SGD)\n"
                             "  n       mini-batch gradient descent,"
                             " n patients")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed, to get the same weights"
                             " initialization\n"
                             "and shuffles on every run\n"
                             "(default: none, they change on each run).")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        metavar="RATE",
                        help="size of each weights update step"
                             " (default: 0.01).")
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of passes over the whole training set"
                             " (default: 100).")

    args = parser.parse_args()
    if args.batch is not None and args.batch < 1:
        parser.error("--batch must be >= 1.")
    if args.learning_rate <= 0:
        parser.error("--learning_rate must be > 0.")
    if args.epochs < 1:
        parser.error("--epochs must be >= 1.")

    return args


def print_epoch(metrics: dict, epoch: int, epochs: int) -> None:
    """Print the results of one epoch on a single line.

    metrics holds "loss" and "acc" for the training set, "val_loss" and
    "val_acc" for the validation set; the accuracies are between 0 and 1
    and displayed in percent. The epoch number is padded with zeros to
    the width of epochs, the total number of epochs.
    """

    size = len(str(epochs))
    print(color(f"epoch {epoch:0{size}d}/{epochs}", DIM),
          f"- loss: {metrics['loss']:.4f}",
          color(f"- val_loss: {metrics['val_loss']:.4f}", BOLD),
          f"- acc: {metrics['acc']:.2%}",
          color(f"- val_acc: {metrics['val_acc']:.2%}", BOLD))


def main():
    """Parse the arguments, prepare both datasets, build the network
    and train it with gradient descent on the whole training set.

    Each epoch displays the loss and the accuracy on both sets. The
    validation set is only measured: it never updates the weights, and
    its forward pass comes after the backward pass, since it overwrites
    the x and z kept by the layers.
    """

    args = parse_args()
    df_train, df_valid = (load_data(args.dataset_train),
                          load_data(args.dataset_valid))

    if args.batch is not None and args.batch > df_train.shape[0]:
        args.batch = df_train.shape[0]

    params = fit_preprocessing(df_train)
    x_train, y_train = prepare_data(df_train, params)
    x_valid, y_valid = prepare_data(df_valid, params)

    rng = make_rng(args.seed)
    layers = create_network(rng, x_train)

    np.set_printoptions(suppress=True, precision=4)

    print(f"x_train shape : {x_train.shape}")
    print(f"x_valid shape : {x_valid.shape}\n")

    for epoch in range(1, args.epochs + 1):
        probs = forward_pass(layers, x_train)
        loss = cross_entropy(probs, y_train)
        backward_pass(layers, probs, y_train)
        for layer in layers:
            layer.update(args.learning_rate)
        val_probs = forward_pass(layers, x_valid)
        val_loss = cross_entropy(val_probs, y_valid)
        acc = accuracy(probs, y_train)
        val_acc = accuracy(val_probs, y_valid)
        metrics = {
            "loss": loss,
            "val_loss": val_loss,
            "acc": acc,
            "val_acc": val_acc
        }
        print_epoch(metrics, epoch, args.epochs)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nperceptron training canceled.")
        sys.exit(130)
