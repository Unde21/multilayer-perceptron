"""Train a multilayer perceptron and evaluate it on a validation set."""

from srcs.data import (load_data)
from srcs.preprocessing import (fit_preprocessing, prepare_data)
import sys
import pandas as pd
import numpy as np
import argparse


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


def main():
    """Parse the arguments, then load and prepare both datasets."""

    args = parse_args()
    df_train, df_valid = (load_data(args.dataset_train),
                          load_data(args.dataset_valid))

    if args.batch is not None and args.batch > df_train.shape[0]:
        args.batch = df_train.shape[0]

    params = fit_preprocessing(df_train)
    # print(f"params:\n{params}")
    x_train, y_train = prepare_data(df_train, params)
    # print(f"x_train:\n{x_train}, y_train:\n{y_train}")
    x_valid, y_valid = prepare_data(df_valid, params)
    # print(f"x_valid:\n{x_valid}, y_valid:\n{y_valid}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nperceptron training canceled.")
        sys.exit(130)
