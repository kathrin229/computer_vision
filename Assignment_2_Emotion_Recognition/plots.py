import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_train_val(epochs, loss_train, loss_val, metric, IMG_DIR):
    sns.set_style("darkgrid")
    epochs_twice = np.tile(epochs, 2)
    hue_train = ['training' for l in loss_train]
    hue_val = ['validation' for v in loss_val]
    hue = hue_train + hue_val
    loss = loss_train + loss_val
    df = pd.DataFrame({'loss': loss, 'epochs': epochs_twice, 'hue': hue})
    ax = sns.lineplot(x='epochs', y='loss', hue='hue', data=df)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(str(metric))
    ax.set_xticks(epochs)
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=[h for h in handles if handles.index(h) != 0])  # remove subtitle for hue entries from legend
    if metric == "Cross Entropy":
        name = "ce"
    else:
        name = "acc"
        ax.set_yticks(np.linspace(0, 1, 11))
    plt.savefig(f"{IMG_DIR}_{name}_plot.png")
    plt.show()


def print_confusion_matrix(confusion_matrix, class_names, figsize=(10,7), fontsize=14):
    # reference: https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Note that due to returning the created figure object, when this funciton is called in a
    notebook the figure willl be printed twice. To prevent this, either append ; to your
    function call, or modify the function by commenting out the return expression.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """

    df_cm = pd.DataFrame(
            confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # Note that due to returning the created figure object, when this funciton is called in a notebook
    # the figure will be printed twice. To prevent this, either append ; to your function call, or
    # modify the function by commenting out this return expression.
    return fig
