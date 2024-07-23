"""
Lupus: Logistic regression weights for SLE and/or ancestry
"""

from anndata import read_h5ad
import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from .commonFuncs.plotLupus import samples_only_lupus
from ..factorization import correct_conditions
from ..logisticReg import logistic_regression
from matplotlib.axes import Axes
import anndata
from .commonFuncs.plotGeneral import cell_count_perc_df, rotate_xaxis


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((5, 4), (2, 1))

    # Add subplot labels
    subplotLabel(ax)

    X = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")

    df = cell_count_perc_df(X, celltype="Cell Type2", status=True)
    sns.boxplot(
        data=df,
        x="Cell Type",
        y="Cell Type Percentage",
        hue="SLE_status",
        showfliers=False,
        ax=ax[0],
    )
    rotate_xaxis(ax[0])

    X.uns["Pf2_A"] = correct_conditions(X)

    samples_only_df = samples_only_lupus(X)

    logreg_weights_status, logreg_score_status = logreg_weights_scores(
        X, samples_only_df, "SLE_status"
    )
    plot_logreg_weights_status(logreg_weights_status, logreg_score_status, ax[1])

    return f


def logreg_weights_scores(
    X: anndata.AnnData, y: pd.Series, prediction: str
) -> pd.DataFrame:
    """Fit logistic regression model, return coefficients of that model"""
    status = y[prediction]
    cond_factors = np.array(X.uns["Pf2_A"])
    lr = logistic_regression("accuracy").fit(cond_factors, status)
    cmp_col = [i for i in range(1, cond_factors.shape[1] + 1)]

    df = pd.DataFrame({"Component": cmp_col, "Weight": lr.coef_.flatten()})

    return df, lr.score(cond_factors, status)


def plot_logreg_weights_status(
    logreg_weights_df: pd.DataFrame, logreg_predaccuracy: float, ax: Axes
):
    """Plots logistic regression weights for predicting by status"""
    sns.barplot(
        data=logreg_weights_df,
        x="Component",
        y="Weight",
        color="k",
        errorbar=None,
        ax=ax,
    )
    ax.set(
        ylim=[-10, 10],
        title="LR Prediction Accuracy: " + str(np.round(logreg_predaccuracy, 3)),
    )
