"""Train a multilayer perceptron and evaluate it on a validation set."""

from srcs.data import (load_data)
from srcs.preprocessing import (fit_preprocessing, prepare_data)
from srcs.layers import DenseLayer
from srcs.losses import (cross_entropy, accuracy)
from srcs.display import (BOLD, DIM, color)
from srcs.plots import (plot_history)
from srcs.activations import (ACTIVATIONS)
import sys
import numpy as np
import argparse


def make_rng(seed: int):
    """Return the random generator shared by the whole training.

    The same seed gives the same weights initialization and shuffles.
    """

    rng = np.random.default_rng(seed)
    return rng


def create_network(rng, x_train: np.ndarray, hidden_sizes: list,
                   activation: str) -> list:
    """Build the layers of the network, from input to output.

    hidden_sizes gives the number of neurons of each hidden layer, in
    order (for example [24, 24]), and activation their activation name
    ("relu" or "sigmoid"). The input size of each layer is the output
    size of the previous one, starting from the number of features of
    x_train. A softmax output layer with one neuron per state is added
    at the end.
    """

    layers = []

    previous_size = x_train.shape[1]
    for i in hidden_sizes:
        layers.append(DenseLayer(previous_size, i, activation, rng))
        previous_size = i
    layers.append(DenseLayer(previous_size, 2, "softmax", rng))
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
               " --learning_rate 0.03 --seed 42\n"
               "  python3 train.py datasets/data_train.csv"
               " datasets/data_valid.csv --layer 24 24 24",
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
    parser.add_argument("--layer", type=int, nargs="+", default=[24, 24],
                        metavar="NEURONS",
                        help="number of neurons of each hidden layer,"
                             " from input to output\n"
                             "(default: 24 24, two hidden layers of 24"
                             " neurons).\n"
                             "The input and output layers are set by the"
                             " data:\n"
                             "one neuron per feature, one per state.")
    hidden_activations = [name for name, (_, derivative)
                          in ACTIVATIONS.items() if derivative is not None]
    parser.add_argument("--activation", default="relu",
                        choices=hidden_activations,
                        help="activation function of the hidden layers"
                             " (default: relu).\n"
                             "The output layer always uses softmax.")

    args = parser.parse_args()
    if args.batch is not None and args.batch < 1:
        parser.error("--batch must be >= 1.")
    if args.learning_rate <= 0:
        parser.error("--learning_rate must be > 0.")
    if args.epochs < 1:
        parser.error("--epochs must be >= 1.")
    for layer in args.layer:
        if layer > 1000 or layer < 1:
            parser.error(f"--layer: invalid size {layer}, each hidden layer"
                         " must have between 1 and 1000 neurons.")
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


def print_hyperparams(hyperparam: dict, requested_batch: int) -> None:
    """Print the settings of the training, before the first epoch.

    requested_batch is the value given to --batch, or None. If it was
    larger than the training set, the reduced size is pointed out.
    Without a seed, "random" is displayed.
    """

    warning = ''
    if (requested_batch is not None
            and requested_batch > hyperparam["batch_size"]):
        warning = f"(reduced from {requested_batch})"

    print(f"{'epochs':<15}: {hyperparam['epochs']}")
    print(f"{'batch size':<15}: {hyperparam['batch_size']} {warning}")
    print(f"{'learning rate':<15}: {hyperparam['learning_rate']}")
    print(f"{'hidden layers':<15}: {hyperparam['hidden_sizes']}")
    print(f"{'activation':<15}: {hyperparam['activation']}")

    seed = f"{hyperparam['seed']}"
    if hyperparam["seed"] is None:
        seed = 'random'
    print(f"{'seed':<15}: {seed}\n")


def train_epoch(layers: list, x: np.ndarray, y: np.ndarray,
                hyperparams: dict, rng) -> None:
    """Train the network on every patient once, batch by batch.

    The patients are shuffled, then split in batches of "batch_size"
    patients (the last one may be smaller). For each batch: forward
    pass, backward pass, then update of the weights with "learning_rate".
    x and y follow the same order, so each patient keeps its state.
    """

    order = rng.permutation(x.shape[0])
    for start in range(0, x.shape[0], hyperparams["batch_size"]):
        batch_indices = order[start: start + hyperparams["batch_size"]]
        x_batch, y_batch = x[batch_indices], y[batch_indices]
        probs_batch = forward_pass(layers, x_batch)
        backward_pass(layers, probs_batch, y_batch)
        for layer in layers:
            layer.update(hyperparams["learning_rate"])


def main():
    """Parse the arguments, prepare both datasets, build the network
    and train it with gradient descent.

    Without --batch, or if it is too large, each batch is the whole
    training set. After each epoch, the loss and the accuracy are
    measured on both sets with the same weights, then displayed. The
    validation set never updates the weights. The learning curves are
    plotted at the end.
    """

    args = parse_args()
    df_train, df_valid = (load_data(args.dataset_train),
                          load_data(args.dataset_valid))

    requested_batch = args.batch
    if args.batch is None or args.batch > df_train.shape[0]:
        args.batch = df_train.shape[0]

    params = fit_preprocessing(df_train)
    x_train, y_train = prepare_data(df_train, params)
    x_valid, y_valid = prepare_data(df_valid, params)

    rng = make_rng(args.seed)
    layers = create_network(rng, x_train, args.layer, args.activation)

    print(f"x_train shape : {x_train.shape}")
    print(f"x_valid shape : {x_valid.shape}\n")

    hyperparams = {
        "batch_size": args.batch,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "seed": args.seed,
        "hidden_sizes": args.layer,
        "activation": args.activation
    }
    print_hyperparams(hyperparams, requested_batch)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_epoch(layers, x_train, y_train, hyperparams, rng)
        probs = forward_pass(layers, x_train)
        loss = cross_entropy(probs, y_train)
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
        history.append(metrics)
        print_epoch(metrics, epoch, args.epochs)

    plot_history(history)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nperceptron training canceled.")
        sys.exit(130)
