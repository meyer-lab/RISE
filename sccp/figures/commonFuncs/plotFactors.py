from typing import Optional
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.cluster.hierarchy as sch
from matplotlib.patches import Patch


def plotFactors(
    factors,
    data,
    axs,
    reorder=tuple(),
    trim=tuple(),
    cond_group_labels: Optional[pd.Series] = None,
):
    """Plots parafac2 factors."""
    pd.set_option("display.max_rows", None)
    rank = factors[0].shape[1]
    xticks = [f"Cmp. {i}" for i in np.arange(1, rank + 1)]
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    for i in range(3):
        # The single cell mode has a square factors matrix
        if i == 0:
            yt = np.unique(data.obs["Condition"])
            title = "Components by Condition"
        elif i == 1:
            yt = [f"Cell State {i}" for i in np.arange(1, rank + 1)]
            title = "Components by Cell State"
        else:
            yt = data.var.to_numpy()
            title = "Components by Gene"

        X = factors[i]

        if i in trim:
            max_weight = np.max(np.abs(X), axis=1)
            kept_idxs = max_weight > 0.08
            X = X[kept_idxs]
            yt = [yt[ii] for ii in kept_idxs]

        if i in reorder:
            X, ind = reorder_table(X)
            yt = [yt[ii] for ii in ind]
            if i == 0 and not (cond_group_labels is None):
                cond_group_labels = cond_group_labels.iloc[ind]

        X = X / np.max(np.abs(X))

        if i == 0:
            vmin = 0
        else:
            vmin = -1

        sns.heatmap(
            data=X,
            xticklabels=xticks,
            yticklabels=yt,
            ax=axs[i],
            center=0,
            cmap=cmap,
            vmin=vmin,
            vmax=1,
        )

        if i == 0 and not (cond_group_labels is None):
            # add little boxes to denote SLE/healthy rows
            axs[i].tick_params(
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
                axs[i].add_patch(
                    plt.Rectangle(
                        xy=(-0.05, iii),
                        width=0.05,
                        height=1,
                        color=color,
                        lw=0,
                        transform=axs[i].get_yaxis_transform(),
                        clip_on=False,
                    )
                )
            # add a little legend
            axs[i].legend(handles=legend_elements, bbox_to_anchor=(0.18, 1.07))

        axs[i].set_title(title)
        axs[i].tick_params(axis="y", rotation=0)


def reorder_table(projs: np.ndarray):
    """Reorder a table's rows using heirarchical clustering"""
    assert projs.ndim == 2
    Z = sch.linkage(projs, method="centroid", optimal_ordering=True)
    index = sch.leaves_list(Z)
    return projs[index, :], index


def plotWeight(weight: np.ndarray, ax):
    """Plots weights from Pf2 model"""
    df = pd.DataFrame(data=np.transpose([weight]), columns=["Value"])
    df["Value"] = df["Value"] / np.max(df["Value"])
    df["Component"] = [f"Cmp. {i}" for i in np.arange(1, len(weight) + 1)]
    sns.barplot(data=df, x="Component", y="Value", ax=ax)
    ax.tick_params(axis="x", rotation=90)
