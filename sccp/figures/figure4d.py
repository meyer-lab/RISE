"""
Lupus: Prediction accuracy for all two
pair logistic regression combinations
"""

from anndata import read_h5ad
import itertools
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from .common import subplotLabel, getSetup
from .commonFuncs.plotLupus import samples_only_lupus
from ..factorization import correct_conditions
from .commonFuncs.plotGeneral import rotate_xaxis, rotate_yaxis
from matplotlib.axes import Axes
import anndata
import pandas as pd


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((6, 6), (1, 1))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")

    status = samples_only_lupus(X)

    pair_logistic_regression(correct_conditions(X), status, ax[0])

    return f


def pair_logistic_regression(X: anndata.AnnData, status_df: pd.DataFrame, ax: Axes):
    """Plot factor weights for donor SLE prediction"""
    lrmodel = LogisticRegression(penalty=None)
    y = preprocessing.label_binarize(
        status_df.SLE_status, classes=["Healthy", "SLE"]
    ).flatten()
    all_comps = np.arange(X.shape[1])
    acc = np.zeros((X.shape[1], X.shape[1]))

    for comps in itertools.product(all_comps, all_comps):
        if comps[0] >= comps[1]:
            compFacs = X[:, [comps[0], comps[1]]]
            LR_CoH = lrmodel.fit(compFacs, y)
            acc[comps[0], comps[1]] = LR_CoH.score(compFacs, y)
            acc[comps[1], comps[0]] = acc[comps[0], comps[1]]

    mask = np.triu(np.ones_like(acc, dtype=bool))

    for i in range(len(mask)):
        mask[i, i] = False

    sns.heatmap(
        data=acc,
        vmin=0.5,
        vmax=1,
        xticklabels=all_comps + 1,
        yticklabels=all_comps + 1,
        mask=mask,
        cbar_kws={"label": "Prediction Accuracy"},
        ax=ax,
    )

    ax.set(xlabel="Component", ylabel="Component")
    rotate_xaxis(ax, rotation=0)
    rotate_yaxis(ax, rotation=0)
