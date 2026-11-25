"""Learning curves displayed at the end of the training."""

import matplotlib.pyplot as plt
import sys


def set_ax(ax, lst: list, label: str, epochs: range, validation: bool) -> None:
    """Draw one curve on ax: the values of lst at each epoch.

    The validation curve is dashed, to tell it from the training curve.
    """

    if validation is False:
        ax.plot(epochs, lst, label=label)
    else:
        ax.plot(epochs, lst, label=label, linestyle='--', alpha=0.5)


def set_axes_style(title: str, ax, y_label: str) -> None:
    """Set the title, the axis labels, the legend and the grid of ax.

    Must be called after the curves are drawn, since the legend is built
    from their labels.
    """

    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xlabel("epochs", fontsize=12, fontstyle='italic')
    ax.set_ylabel(y_label, fontsize=12, fontstyle='italic')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)


def plot_history(history: list) -> None:
    """Display the learning curves: the loss and the accuracy per epoch.

    history holds one metrics dictionary per epoch ("loss", "val_loss",
    "acc", "val_acc"). Each graph compares the training set with the
    validation set. The program waits until the window is closed.
    """

    loss_values = [record["loss"] for record in history]
    val_loss_values = [record["val_loss"] for record in history]
    acc_values = [record["acc"] for record in history]
    val_acc_values = [record["val_acc"] for record in history]

    epochs = range(1, len(loss_values) + 1)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    set_ax(axes[0], loss_values, "training loss", epochs, False)
    set_ax(axes[0], val_loss_values, "validation loss", epochs, True)

    set_ax(axes[1], acc_values, "training accuracy", epochs, False)
    set_ax(axes[1], val_acc_values, "validation accuracy", epochs, True)

    set_axes_style("Loss per epoch", axes[0], "loss")
    set_axes_style("Accuracy per epoch", axes[1], "accuracy")

    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nPlot display closed.")
        sys.exit(130)
