import seaborn as sns
import anndata
import numpy as np
from matplotlib.axes import Axes
from sklearn.metrics import RocCurveDisplay
from ...logisticReg import (
    predaccuracy_ranks_lupus,
    roc_lupus_fourtbatch,
)


def samples_only_lupus(X: anndata.AnnData):
    """Obtain samples once only with corresponding observations"""
    samples = X.obs
    df_samples = samples.drop_duplicates(subset="condition_unique_idxs")
    df_samples = df_samples.sort_values("condition_unique_idxs")

    return df_samples


def plot_accuracy_ranks_lupus(
    X: anndata.AnnData, ranks: np.ndarray, ax: Axes, error_metric="roc_auc"
):
    """Plots results from Pf2 test of various ranks using defined error metric and logistic reg"""
    pred_accuracy_df = predaccuracy_ranks_lupus(
        X, samples_only_lupus(X), ranks, error_metric
    )

    sns.lineplot(
        data=pred_accuracy_df,
        x="Component",
        y=error_metric,
        ax=ax,
    )
    sns.scatterplot(
        data=pred_accuracy_df,
        x="Component",
        y=error_metric,
        ax=ax,
    )
    ax.set(ylim=[-0.05, 1.05])


def plot_roc_fourthbatch(X: anndata.AnnData, ax: Axes):
    """Plots ROC curve for prediction"""
    y_test, sle_decisions = roc_lupus_fourtbatch(X, samples_only_lupus(X))

    RocCurveDisplay.from_predictions(
        y_test, sle_decisions, pos_label=True, plot_chance_level=True, ax=ax
    )
