"""Save the trained network to a JSON file, to be used by predict."""

import sys
import json


def save_model(params: dict, layers: list, path: str) -> None:
    """Write everything predict needs to redo the same computations.

    For each layer, in order: its weights, its bias and its activation
    name (the layer sizes are given by the shape of the weights). Also
    the normalization parameters of the training set and the order of
    the states, which tells which output column is which state. The
    numpy arrays are converted to lists, since JSON cannot store them.
    Exit with an error message if the file cannot be written.
    """

    all_layers = [
        {
            "weights": layer.weights.tolist(),
            "bias": layer.bias.tolist(),
            "activation": layer.activation_name
        }
        for layer in layers
    ]

    data = {
        "norm_params": params["norm_params"],
        "label": params["label_state"],
        "layers": all_layers
    }

    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"> saving model '{path}' to disk...")

    except (PermissionError, IsADirectoryError, OSError) as error:
        print(f"Error: cannot write '{error.filename}': "
              f"{(error.strerror or str(error)).lower()}.",
              file=sys.stderr)
        sys.exit(1)
