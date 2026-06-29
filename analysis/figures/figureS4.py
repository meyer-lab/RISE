"""
Figure S4
"""

from .common import getSetup, subplotLabel


def makeFigure():
    ax, f = getSetup((6, 3), (1, 2))
    subplotLabel(ax)

    # X = import_thomson()
    # percentList = np.arange(0.0, 8.0, 5.0)
    # plot_fms_percent_drop(X, ax[0], percentList=percentList, runs=3)

    # ranks = list(range(1, 3))
    # plot_fms_diff_ranks(X, ax[1], ranksList=ranks, runs=3)

    return f
