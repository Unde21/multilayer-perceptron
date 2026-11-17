"""Split the dataset into a training set and a validation set."""

from srcs.data import (load_data, LABEL)
import sys
import argparse
import pandas as pd
import numpy as np


def split_data(df: pd.DataFrame, ratio: float,
               seed: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into shuffled training and validation sets.

    The ratio is applied to each diagnosis separately, so both sets keep the
    proportions of M and B of the original dataset. Print an error message
    and exit with status 1 if one of the sets is empty.
    """

    rng = np.random.default_rng(seed)
    malignant = df[df[LABEL] == "M"]
    benign = df[df[LABEL] == "B"]

    train_malignant = malignant.sample(frac=ratio, random_state=rng)
    train_benign = benign.sample(frac=ratio, random_state=rng)

    valid_malignant = malignant.drop(train_malignant.index)
    valid_benign = benign.drop(train_benign.index)

    df_train = pd.concat([train_malignant, train_benign])
    df_valid = pd.concat([valid_malignant, valid_benign])

    df_train = df_train.sample(frac=1, random_state=rng)
    df_valid = df_valid.sample(frac=1, random_state=rng)

    if df_train.empty:
        print(f"Error: empty training set with ratio {ratio} on {len(df)} "
              "row(s): increase --ratio or use more rows.", file=sys.stderr)
        sys.exit(1)
    if df_valid.empty:
        print(f"Error: empty validation set with ratio {ratio} on {len(df)} "
              "row(s): decrease --ratio or use more rows.", file=sys.stderr)
        sys.exit(1)

    return df_train, df_valid


def save_csv(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to a csv file, without header nor index.

    Print an error message and exit with status 1 if the file cannot be
    written."""

    try:
        df.to_csv(path, index=False, header=False)
        print(f"> saved {len(df)} rows to '{path}'")
    except (OSError) as error:
        print(f"Error: cannot write '{path}' : "
              f"{(error.strerror or str(error)).lower()}.",
              file=sys.stderr)
        sys.exit(1)


def main():
    """Load the dataset, split it and save both sets to csv files."""

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

    df_train, df_valid = split_data(df, args.ratio, args.seed)
    # print(f"Train: {df_train.shape}\n{df_train["diagnosis"].value_counts()}"
    #       f"\n{df_train["diagnosis"].value_counts(normalize=True) * 100}\n")
    # print(f"Valid: {df_valid.shape}\n{df_valid["diagnosis"].value_counts()}"
    #       f"\n{df_valid["diagnosis"].value_counts(normalize=True) * 100}")
    save_csv(df_train, "datasets/data_train.csv")
    save_csv(df_valid, "datasets/data_valid.csv")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nsplitting dataset canceled.")
        sys.exit(130)
