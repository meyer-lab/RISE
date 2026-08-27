"""
Figure S6
"""

from RISE.plotting import plot_labels_pacmap, plot_wp_pacmap

from ..imports import import_thomson_factors
from .common import getSetup, subplotLabel


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    ax, f = getSetup((12, 12), (6, 4))
    subplotLabel(ax)

    X = import_thomson_factors()

    plot_labels_pacmap(X, "Cell Type", ax[0])
    plot_labels_pacmap(X, "Cell Type2", ax[1])

    ax[2].axis("off")
    ax[3].axis("off")

    for i in range(1, 21):
        plot_wp_pacmap(X, i, ax[i + 3], cbarMax=0.3)

    return f
