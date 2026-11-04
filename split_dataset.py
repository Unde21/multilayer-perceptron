"""Split the dataset into a training set and a validation set."""

from srcs.utils import (load_data)
import sys
import argparse
# import pandas as pd


def main():
    """Parse the command line arguments and load the dataset."""

    parser = argparse.ArgumentParser(
        description="Split a dataset into a training set and a validation"
                    " set.",
        epilog="examples:\n"
               "  python3 split_dataset.py datasets/data.csv\n"
               "  python3 split_dataset.py datasets/data.csv --ratio 0.75\n"
               "  python3 split_dataset.py datasets/data.csv --seed 42",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("dataset", help="path to the dataset .csv file.")
    parser.add_argument("--ratio", type=float, default=0.8,
                        help="share of the rows used for training, strictly"
                             " between 0 and 1\n"
                             "(default: 0.8, so 80%% train and 20%%"
                             " validation).")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed, to get the same split on every"
                             " run\n"
                             "(default: none, the split changes on each run).")

    args = parser.parse_args()
    if not 0 < args.ratio < 1:
        parser.error("--ratio must be strictly between 0 and 1.")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be a non-negative integer.")
    df = load_data(args.dataset)
    print(df)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nsplitting dataset canceled.")
        sys.exit(130)
