import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.axes import Axes
from sklearn.metrics import RocCurveDisplay, auc
from sklearn.model_selection import StratifiedGroupKFold
from ...logisticReg import (
    predaccuracy_ranks_lupus,
    logistic_regression,
    roc_lupus_fourtbatch,
)
from sklearn.metrics import roc_auc_score


def samples_only_lupus(X) -> pd.DataFrame:
    """Obtain samples once only with corresponding observations"""
    samples = X.obs
    df_samples = samples.drop_duplicates(subset="condition_unique_idxs")
    df_samples = df_samples.sort_values("condition_unique_idxs")
    return df_samples


def plot_predaccuracy_ranks_lupus(
    X, ranks, ax: Axes, error_metric="accuracy", palette="tab10"
):
    """Plots results from Pf2 test of various ranks using defined error metric and logistic reg"""
    pred_accuracy_df = predaccuracy_ranks_lupus(X, samples_only_lupus(X), ranks, error_metric)
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


def plot_roc_allbatches_lupus(
    X,
    ax: Axes,
    pred_group="SLE_status",
    cv_group="Processing_Cohort",
    n_splits=4,
):
    cond_factors = X.uns["Pf2_A"]
    condition_labels_all = samples_only_lupus(X)

    condition_labels = condition_labels_all[pred_group]

    sgkf = StratifiedGroupKFold(n_splits=n_splits)

    # get labels for the group that you want to do cross validation by
    group_cond_labels = condition_labels_all[cv_group]

    log_reg = logistic_regression(scoring="roc_auc")

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for fold, (train, test) in enumerate(
        sgkf.split(
            cond_factors, condition_labels.to_numpy(), group_cond_labels.to_numpy()
        )
    ):
        # adding escape option for the second fold (@ index 1) because it has no SLE cases.
        # otherwise we just get NA for our mean and NA for that fold. which isn't super helpful
        # this if statement shouldn't be used with other data
        if fold == 1:
            continue
        log_reg.fit(cond_factors[train], condition_labels.to_numpy()[train])
        viz = RocCurveDisplay.from_estimator(
            log_reg,
            cond_factors[test],
            condition_labels.to_numpy()[test],
            name=f"ROC fold {fold}",
            alpha=0.3,
            lw=1,
            ax=ax,
            plot_chance_level=(fold == n_splits - 1),
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Mean ROC curve with variability",
    )
    ax.axis("square")
    ax.legend(loc="lower right")


def plot_roc_fourthbatch(X, ax):
    """Plots ROC curve for prediction"""
    y_test, sle_decisions = roc_lupus_fourtbatch(X, samples_only_lupus(X))
    
    roc_auc = roc_auc_score(y_test, sle_decisions)
    print("ROC AUC: ", roc_auc)

    RocCurveDisplay.from_predictions(
        y_test, sle_decisions, pos_label=True, plot_chance_level=True, ax=ax
    )
