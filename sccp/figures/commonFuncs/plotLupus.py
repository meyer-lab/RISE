import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.axes import Axes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, auc
from sklearn.model_selection import StratifiedGroupKFold


def plotPf2RankTest(
    rank_test_results, ax: Axes, error_metric="accuracy", palette="tab10"
):
    """Plots results from Pf2 test of various ranks using defined error metric and logistic reg"""
    sns.lineplot(
        data=rank_test_results,
        x="rank",
        y=error_metric,
        hue="penalty",
        palette=palette,
        ax=ax,
    )
    sns.scatterplot(
        data=rank_test_results,
        x="rank",
        y=error_metric,
        hue="penalty",
        palette=palette,
        legend=False,
        ax=ax,
    )
    ax.set_title(error_metric + " by Hyperparameter input")
    ax.set(ylim=[-0.05, 1.05])


def plotCmpRegContributions(contribs, predicting: str, ax: Axes):
    """Plots weights of components in logistic regression from `getCompContribs`"""
    sns.barplot(
        data=contribs, x="Component", y="Weight", color="k", errorbar=None, ax=ax
    )
    ax.tick_params(axis="x", rotation=90)
    ax.set_title("Weight of Pf2 Cmps in Logsitic Regression: Predicting " + predicting)


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
        cmp_n.groupby(by=obs_column)
        .count()
        .reset_index()
        .rename({component_string: "count"}, axis=1)
    )
    cmp_n = cmp_n[cmp_n[component_string] > threshold]

    counts = (
        cmp_n.groupby(by=obs_column)
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


def plot2DSeparationByComp(df, pair: tuple, predict: str, ax: Axes):
    """
    Plots the separation of some observation variable (hue) that is contained in
    the input merged dataframe across two components, passed as two strings in a tuple (x_y)
    that denote the names of the columns to be used for the x and y axes.
    """
    sns.scatterplot(
        data=df, x=f"Cmp. {pair[0]}", y=f"Cmp. {pair[1]}", hue=predict, ax=ax
    )


def plotROCAcrossGroups(
    A_matrix,
    group_labs,
    ax: Axes,
    pred_group="SLE_status",
    cv_group="Processing_Cohort",
    penalty_type="l1",
    solver="saga",
    penalty=50,
    n_splits=4,
):
    condition_labels_all = group_labs

    condition_labels = group_labs[pred_group]

    sgkf = StratifiedGroupKFold(n_splits=n_splits)

    # get labels for the group that you want to do cross validation by
    group_cond_labels = condition_labels_all[cv_group]
    # set up log reg specs
    log_reg = LogisticRegression(
        random_state=0, max_iter=5000, penalty=penalty_type, solver=solver, C=penalty
    )

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for fold, (train, test) in enumerate(
        sgkf.split(A_matrix, condition_labels.to_numpy(), group_cond_labels.to_numpy())
    ):
        # adding escape option for the second fold (@ index 1) because it has no SLE cases.
        # otherwise we just get NA for our mean and NA for that fold. which isn't super helpful
        # this if statement shouldn't be used with other data
        if fold == 1:
            continue
        log_reg.fit(A_matrix[train], condition_labels.to_numpy()[train])
        viz = RocCurveDisplay.from_estimator(
            log_reg,
            A_matrix[test],
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
