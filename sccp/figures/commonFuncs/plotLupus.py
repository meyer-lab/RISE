import seaborn as sns
import pandas as pd
from matplotlib.axes import Axes
from sklearn.metrics import RocCurveDisplay
from ...logisticReg import (
    predaccuracy_ranks_lupus,
    roc_lupus_fourtbatch,
)


def samples_only_lupus(X) -> pd.DataFrame:
    """Obtain samples once only with corresponding observations"""
    samples = X.obs
    df_samples = samples.drop_duplicates(subset="condition_unique_idxs")
    df_samples = df_samples.sort_values("condition_unique_idxs")
    return df_samples


def plot_accuracy_ranks_lupus(
    X, ranks, ax: Axes, error_metric="roc_auc", palette="tab10"
):
    """Plots results from Pf2 test of various ranks using defined error metric and logistic reg"""
    pred_accuracy_df = predaccuracy_ranks_lupus(
        X, samples_only_lupus(X), ranks, error_metric
    )
    sns.lineplot(
        data=pred_accuracy_df,
        x="Component",
        y=error_metric,
        palette=palette,
        ax=ax,
    )
    sns.scatterplot(
        data=pred_accuracy_df,
        x="Component",
        y=error_metric,
        palette=palette,
        legend=False,
        ax=ax,
    )
    ax.set(ylim=[-0.05, 1.05])


def investigate_comp(X, comp: int, obs_column: str, ax: Axes, threshold: float = 0.05):
    """Makes barplots of the percentages of each observation column (obs_column) that are represented in the top
    contributors to a certain component (comp). Top contributors are determined by having a contribution above `threshold`
    """
    component_string = f"Cmp. {comp}"
    cmp_n = pd.DataFrame(
        {
            component_string: X.obsm["weighted_projections"][:, comp - 1],
            obs_column: X.obs[obs_column],
        }
    )

    # get just the ones that are "super" positive
    counts_all = (
        cmp_n.groupby(by=obs_column, observed=False)
        .count()
        .reset_index()
        .rename({component_string: "count"}, axis=1)
    )
    cmp_n = cmp_n[cmp_n[component_string] > threshold]

    counts = (
        cmp_n.groupby(by=obs_column, observed=False)
        .count()
        .reset_index()
        .rename({component_string: "count"}, axis=1)
    )

    pcts = pd.concat(
        (counts[obs_column], counts["count"] / counts_all["count"]), axis=1
    ).rename({"count": "percent"}, axis=1)
    pcts["percent"] = pcts["percent"] * 100

    sns.barplot(pcts, x=obs_column, y="percent", color="k", errorbar=None, ax=ax)
    ax.tick_params(axis="x", rotation=90)
    ax.set_title(
        obs_column
        + " Percentages, Threshold: "
        + str(threshold)
        + " for comp "
        + str(comp)
    )


def plot_roc_fourthbatch(X, ax):
    """Plots ROC curve for prediction"""
    y_test, sle_decisions = roc_lupus_fourtbatch(X, samples_only_lupus(X))

    RocCurveDisplay.from_predictions(
        y_test, sle_decisions, pos_label=True, plot_chance_level=True, ax=ax
    )
