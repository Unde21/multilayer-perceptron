""""""

from srcs.utils import (load_data)
import sys
import argparse
import pandas as pd


def main():
    """"""

    parser = argparse.ArgumentParser(description="split the dataset")
    parser.add_argument("dataset", help="path to the dataset .csv file.")
    parser.add_argument("--ratio", type=float, default=0.8,
                        help="ratio train / validate")
    parser.add_argument("--seed", type=int, default=None,
                        help="the seed.")

    args = parser.parse_args()

    df = load_data(args.dataset)
    print(df)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nsplitting dataset canceled.")
        sys.exit(130)
