from typing import Optional
from anndata import AnnData
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as sch
from matplotlib.patches import Patch


def plotConditionsFactors(
    data: AnnData, ax, reorder=bool, cond_group_labels: Optional[pd.Series] = None
):
    """Plots parafac2 factors."""
    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    yt = pd.Series(np.unique(data.obs["Condition"]))
    X = np.array(data.uns["Pf2_A"])

    controls = yt.str.contains("CTRL")

    X = np.log10(X)
    X -= np.median(X[controls], axis=0)
    X /= np.std(X[controls], axis=0)

    if reorder:
        ind = reorder_table(X)
        X = X[ind]
        yt = yt.iloc[ind]
        if cond_group_labels is not None:
            cond_group_labels = cond_group_labels.iloc[ind]
            ind = cond_group_labels.argsort()
            cond_group_labels = cond_group_labels.iloc[ind]
            X = X[ind]
            yt = yt.iloc[ind]

    xticks = [f"Cmp. {i}" for i in np.arange(1, X.shape[1] + 1)]
    sns.heatmap(
        data=X,
        xticklabels=xticks,
        yticklabels=yt,
        ax=ax,
        center=0,
        cmap=cmap,
    )

    if cond_group_labels is not None:
        # add little boxes to denote SLE/healthy rows
        ax.tick_params(
            axis="y", which="major", pad=20, length=0
        )  # extra padding to leave room for the row colors
        # get list of colors for each label:
        colors = sns.color_palette(
            n_colors=pd.Series(cond_group_labels).nunique()
        ).as_hex()
        lut = {}
        legend_elements = []
        for index, group in enumerate(pd.Series(cond_group_labels).unique()):
            lut[group] = colors[index]
            legend_elements.append(Patch(color=colors[index], label=group))
        row_colors = pd.Series(cond_group_labels).map(lut)
        for iii, color in enumerate(row_colors):
            ax.add_patch(
                plt.Rectangle(
                    xy=(-0.05, iii),
                    width=0.05,
                    height=1,
                    color=color,
                    lw=0,
                    transform=ax.get_yaxis_transform(),
                    clip_on=False,
                )
            )
        # add a little legend
        ax.legend(handles=legend_elements, bbox_to_anchor=(0.18, 1.07))

    ax.set_title("Components by Condition")
    ax.tick_params(axis="y", rotation=0)


def plotFactors(
    data: AnnData,
    axs: list,
    reorder=tuple(),
    trim=True,
    cond_group_labels: Optional[pd.Series] = None,
):
    """Plots parafac2 factors."""
    assert len(axs) == 3

    pd.set_option("display.max_rows", None)
    rank = data.uns["Pf2_A"].shape[1]
    xticks = [f"Cmp. {i}" for i in np.arange(1, rank + 1)]
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    for i in range(3):
        # The single cell mode has a square factors matrix
        if i == 0:
            plotConditionsFactors(
                data,
                axs[0],
                reorder=(0 in reorder),
                cond_group_labels=cond_group_labels,
            )
            continue
        elif i == 1:
            X = data.uns["Pf2_B"]
            yt = [f"Cell State {i}" for i in np.arange(1, rank + 1)]
            title = "Components by Cell State"
        else:
            X = data.varm["Pf2_C"]
            yt = data.var.index.values
            title = "Components by Gene"

        X = np.array(X)

        if (i == 2) and (trim is True):
            max_weight = np.max(np.abs(X), axis=1)
            kept_idxs = max_weight > 0.08
            X = X[kept_idxs]
            yt = yt[kept_idxs]

        if i in reorder:
            ind = reorder_table(X)
            X = X[ind]
            yt = [yt[ii] for ii in ind]

        X = X / np.max(np.abs(X))

        sns.heatmap(
            data=X,
            xticklabels=xticks,
            yticklabels=yt,
            ax=axs[i],
            center=0,
            cmap=cmap,
            vmin=-1,
            vmax=1,
        )

        axs[i].set_title(title)
        axs[i].tick_params(axis="y", rotation=0)


def reorder_table(projs: np.ndarray) -> np.ndarray:
    """Reorder a table's rows using heirarchical clustering"""
    assert projs.ndim == 2
    Z = sch.linkage(projs, method="complete", optimal_ordering=True)
    return sch.leaves_list(Z)


def plotWeight(weight: np.ndarray, ax):
    """Plots weights from Pf2 model"""
    df = pd.DataFrame(data=np.transpose([weight]), columns=["Value"])
    df["Value"] = df["Value"] / np.max(df["Value"])
    df["Component"] = [f"Cmp. {i}" for i in np.arange(1, len(weight) + 1)]
    sns.barplot(data=df, x="Component", y="Value", ax=ax)
    ax.tick_params(axis="x", rotation=90)
