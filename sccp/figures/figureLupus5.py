"""
Lupus: Plot logistic regression weights for SLE and/or ancestry
"""

from anndata import read_h5ad
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from .common import subplotLabel, getSetup
from .commonFuncs.plotLupus import getSamplesObs
from ..factorization import correct_conditions


def getCompContribs(X: np.ndarray, y: pd.Series) -> pd.DataFrame:
    """Fit logistic regression model, return coefficients of that model"""
    lr = LogisticRegressionCV(
        random_state=0, max_iter=100000, penalty="l1", solver="saga"
    ).fit(X, y)

    cmp_col = [i for i in range(1, X.shape[1] + 1)]
    df = pd.DataFrame({"Component": cmp_col, "Weight": lr.coef_.flatten()})

    return df, lr.score(X, y)


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((5, 4), (2, 1))

    # Add subplot labels
    subplotLabel(ax)

    data = read_h5ad("/opt/andrew/lupus/lupus_fitted_ann.h5ad")
    data.uns["Pf2_A"] = correct_conditions(data)

    df_y = getSamplesObs(data.obs)
    X = np.array(data.uns["Pf2_A"])

    (
        dfWeights,
        score,
    ) = getCompContribs(X, df_y["SLE_status"])
    sns.barplot(
        data=dfWeights,
        x="Component",
        y="Weight",
        color="k",
        errorbar=None,
        ax=ax[0],
    )

    ax[0].set(
        ylim=[-10, 10],
        title="Logistic Regression: Prediction Accuracy - " + str(np.round(score, 3)),
    )

    df_y["ancestry"] = df_y["ancestry"] == "European"
    dfAnc, _ = getCompContribs(X, df_y["ancestry"])

    dfWeights["Predicting"] = "SLE Status"
    dfAnc["Predicting"] = "Euro-Ancestry"
    combinedWeights = pd.concat([dfWeights, dfAnc])

    sns.barplot(
        data=combinedWeights,
        x="Component",
        y="Weight",
        hue="Predicting",
        errorbar=None,
        ax=ax[1],
    )

    return f
