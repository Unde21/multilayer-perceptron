"""Turn a loaded dataset into arrays ready for the network.

The parameters (normalization statistics and label order) are learned
on the training set only, then applied the same way to every dataset.
"""

import pandas as pd
import numpy as np
import math
from srcs.stats import (ft_count, ft_mean, ft_std)
from srcs.data import (LABEL, FEATURES)


PRECISION = 1e-12


def calc_mean_std(df: pd.DataFrame) -> dict:
    """Compute the mean and standard deviation of each column of df.

    A standard deviation close to 0 is replaced by 1, so that a constant
    column is normalized to 0 instead of being divided by 0.
    Return a dict {column: {"mean": mean, "std": std}}.
    """

    norm_params = {}

    for col in df.columns:
        values = pd.to_numeric(df[col], errors='coerce').tolist()

        if not values:
            continue

        count = ft_count(values)
        mean = ft_mean(values, count)
        std = ft_std(values, mean, count)
        if math.isnan(std) or std < PRECISION:
            std = 1
        norm_params[col] = {
            "mean": mean,
            "std": std
        }

    return norm_params


def get_states(df: pd.DataFrame) -> list:
    """Return the sorted list of the states found in the label column.

    Call it on the training set only: the order of this list sets which
    output neuron stands for which state.
    """

    label_state = sorted(df[LABEL].dropna().unique())

    return label_state


def encode_states(df: pd.DataFrame, label_state: list) -> np.ndarray:
    """One-hot encode the label column of df, following label_state order.

    Return an array of shape (rows, len(label_state)): 1 in the column of
    the patient's state, 0 elsewhere.
    """

    true_states = df[LABEL].to_numpy()
    return (true_states[:, None] == label_state).astype(int)


def fit_features(df: pd.DataFrame,
                 features=FEATURES) -> dict:
    """Compute the normalization parameters of the features of df.

    Return a dict {feature: {"mean": mean, "std": std}}, computed on the
    training set and reused on the validation set and by the prediction.
    """

    features_df = df[features]
    norm_params = calc_mean_std(features_df)
    return norm_params


def normalize(df: pd.DataFrame, norm_params: dict) -> pd.DataFrame:
    """Apply the z-score (x - mean) / std to each column of df.

    The mean and standard deviation come from norm_params, computed on the
    training set. df is not modified: return a normalized copy.
    """

    df_norm = df.copy()
    for col in df_norm:
        df_norm[col] = (df_norm[col] - norm_params[col]["mean"])
        df_norm[col] /= norm_params[col]["std"]
    return df_norm


def fit_preprocessing(df_train: pd.DataFrame) -> dict:
    """Learn the preprocessing parameters from the training set.

    Return a dict with "norm_params" (mean and std of each feature) and
    "label_state" (the sorted labels, which set the output order).
    """

    label_state = get_states(df_train)
    norm_params = fit_features(df_train)
    params = {
        "norm_params": norm_params,
        "label_state": label_state
    }
    return params


def prepare_data(df: pd.DataFrame, params: dict) -> tuple[np.ndarray,
                                                          np.ndarray]:
    """Turn df into the network inputs and expected outputs.

    Normalize the features and one-hot encode the labels with the params
    returned by fit_preprocessing. Return (x, y) of shapes
    (rows, len(FEATURES)) and (rows, len(label_state)).
    """

    features_df = df[FEATURES]
    y = encode_states(df, params["label_state"])
    x_norm = normalize(features_df, params["norm_params"]).to_numpy()

    return x_norm, y
