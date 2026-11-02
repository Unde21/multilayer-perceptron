""""""

import sys
import pandas as pd

LABEL = "diagnosis"
MEASURES = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension"
]
STATS = [
    "mean",
    "se",
    "worst"
]
FEATURES = [f"{m}_{s}" for s in STATS for m in MEASURES]
COLUMNS = ["id", LABEL] + FEATURES


def load_data(path: str) -> pd.DataFrame:
    """Function that receive Dataset in csv format,
    and return it as Pandas DataFrame if none error occured."""

    try:
        if not isinstance(path, str):
            raise TypeError("path must be a string.")
        if not path.lower().endswith(".csv"):
            raise ValueError("file extension must be '.csv'.")

        df = pd.read_csv(path, header=None, names=COLUMNS)
        if df.empty:
            raise ValueError("csv file is empty.")
        check_dataset(df)

        return df

    except (FileNotFoundError, PermissionError,
            IsADirectoryError) as error:
        print(f"Error: cannot read '{error.filename}': "
              f"{(error.strerror or str(error)).lower()}.", file=sys.stderr)
        sys.exit(1)

    except pd.errors.EmptyDataError:
        print("Error: csv file is empty.", file=sys.stderr)
        sys.exit(1)

    except (TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


def check_dataset(df: pd.DataFrame) -> None:
    """Raise a ValueError if the dataset content is invalid."""

    labels = df[LABEL]
    invalid = ~labels.isin(["M", "B"])
    if invalid.any():
        lines = (df[invalid].index + 1).tolist()
        raise ValueError(f"invalid label (expected 'M' or 'B') "
                         f"at line(s): {lines}.")
