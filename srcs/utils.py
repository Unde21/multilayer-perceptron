"""Load and validate the breast cancer dataset."""

import sys
import pandas as pd
import numpy as np


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
    """Load a headerless csv dataset into a named DataFrame.

    Print an error message and exit with status 1 if the file cannot be
    read or if its content is invalid.
    """

    try:
        if not isinstance(path, str):
            raise TypeError("path must be a string.")
        if not path.lower().endswith(".csv"):
            raise ValueError("file extension must be '.csv'.")

        df = pd.read_csv(path, header=None)
        if df.shape[1] != len(COLUMNS):
            raise ValueError(f"invalid csv format: expected {len(COLUMNS)}"
                             f" columns, got {df.shape[1]}.")
        df.columns = COLUMNS
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

    except pd.errors.ParserError as error:
        print(f"Error: invalid csv format: {error}".strip(), file=sys.stderr)
        sys.exit(1)

    except (TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


def check_dataset(df: pd.DataFrame) -> None:
    """Check that features are finite numbers and labels are M or B.

    Raise a ValueError naming the offending column or lines otherwise.
    """

    for feature in FEATURES:
        if not pd.api.types.is_numeric_dtype(df[feature]):
            raise ValueError(f"non-numeric value in '{feature}'.")
    bad = ~np.isfinite(df[FEATURES])
    bad_rows = bad.any(axis=1)
    if bad_rows.any():
        lines = (df[bad_rows].index + 1).tolist()
        raise ValueError("missing or infinite value"
                         f" in features at line: {lines}.")

    labels = df[LABEL]
    invalid = ~labels.isin(["M", "B"])
    if invalid.any():
        lines = (df[invalid].index + 1).tolist()
        raise ValueError(f"invalid label (expected 'M' or 'B') "
                         f"at line(s): {lines}.")
