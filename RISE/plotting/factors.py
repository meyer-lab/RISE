import anndata
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from scipy.spatial.distance import pdist

from .seriate import seriate

cmap = sns.diverging_palette(240, 10, as_cmap=True)


def plot_condition_factors(
    data: anndata.AnnData,
    ax: Axes,
    cond: str = "Condition",
    log_transform: bool = True,
    cond_group_labels: pd.Series | None = None,
    ThomsonNorm=False,
    color_key=None,
    group_cond=False,
):
    """Plot condition factors as a heatmap showing how conditions contribute to
    components.

    This visualization shows how each experimental condition (rows) contributes to
    each RISE component (columns). High values indicate strong association between
    a condition and a component's pattern. Log transformation and normalization
    help reveal relative differences across conditions.

    Parameters
    ----------
    data : anndata.AnnData
        AnnData object with RISE decomposition results. Must contain:
        - data.uns["Pf2_A"]: Condition factors (n_conditions, rank)
        - data.obs[cond]: Condition labels for each cell
    ax : matplotlib.axes.Axes
        Matplotlib axes object to plot on.
    cond : str, optional (default: "Condition")
        Name of column in data.obs containing condition labels.
    log_transform : bool, optional (default: True)
        If True, applies log10 transformation to condition factors before plotting.
        This helps visualize differences when values span orders of magnitude.
    cond_group_labels : pandas.Series, optional (default: None)
        Series mapping conditions to group labels for colored row annotations.
        Useful for grouping related conditions (e.g., drug classes, patient cohorts).
    ThomsonNorm : bool, optional (default: False)
        If True, normalizes factors using only control conditions (those
        containing 'CTRL').
    color_key : list, optional (default: None)
        Custom colors for condition group labels. If None, uses default palette.
    group_cond : bool, optional (default: False)
        If True and cond_group_labels provided, sorts conditions by group.
    """
    pd.set_option("display.max_rows", None)
    yt = pd.Series(np.unique(data.obs[cond]))
    X = np.array(data.uns["Pf2_A"])

    if log_transform is True:
        X = np.log10(X)

    if ThomsonNorm is True:
        controls = yt.str.contains("CTRL")
        XX = X[controls]
    else:
        XX = X

    X -= np.median(XX, axis=0)
    X /= np.std(XX, axis=0)

    if log_transform is False:
        X -= np.min(X, axis=0)

    ind = reorder_table(X)
    X = X[ind]
    yt = yt.iloc[ind]

    if cond_group_labels is not None:
        cond_group_labels = cond_group_labels.iloc[ind]
        if group_cond is True:
            ind = cond_group_labels.argsort()
            cond_group_labels = cond_group_labels.iloc[ind]
            X = X[ind]
            yt = yt.iloc[ind]
        ax.tick_params(axis="y", which="major", pad=20, length=0)
        if color_key is None:
            colors = sns.color_palette(
                n_colors=pd.Series(cond_group_labels).nunique()
            ).as_hex()
        else:
            colors = color_key
        lut = {}
        legend_elements = []
        for index, group in enumerate(pd.unique(cond_group_labels)):
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
        ax.legend(handles=legend_elements, bbox_to_anchor=(0.18, 1.07))

    xticks = np.arange(1, X.shape[1] + 1)
    sns.heatmap(
        data=X,
        xticklabels=xticks,
        yticklabels=yt,
        ax=ax,
        center=0,
        cmap=cmap,
    )
    ax.tick_params(axis="y", rotation=0)
    ax.set(xlabel="Component")


def plot_eigenstate_factors(data: anndata.AnnData, ax: Axes):
    """Plot eigen-state factors as a heatmap showing cell state patterns.

    Eigen-state factors represent the underlying cell state patterns across components.
    Each row represents an eigen-state (a summary of similar cells), and each column
    represents a component. High values indicate strong association between a cell
    state pattern and a component.

    Parameters
    ----------
    data : anndata.AnnData
        AnnData object with RISE decomposition results. Must contain:
        - data.uns["Pf2_B"]: Eigen-state factors (rank, rank)
    ax : matplotlib.axes.Axes
        Matplotlib axes object to plot on.
    """
    rank = data.uns["Pf2_B"].shape[1]
    xticks = np.arange(1, rank + 1)
    X = data.uns["Pf2_B"]
    X = X / np.max(np.abs(np.array(X)))
    yt = np.arange(1, rank + 1)

    sns.heatmap(
        data=X,
        xticklabels=xticks,
        yticklabels=yt,
        ax=ax,
        center=0,
        cmap=cmap,
        vmin=-1,
        vmax=1,
    )
    ax.set(xlabel="Component")


def plot_gene_factors(data: anndata.AnnData, ax: Axes, weight=0.08, trim=True):
    """Plot gene factors as a heatmap showing which genes contribute to each component.

    This visualization reveals coordinated gene modules by showing which genes (rows)
    are highly weighted in each component (columns). The weight parameter filters out
    genes with low contributions, focusing on the most important genes for
    interpretation.

    Parameters
    ----------
    data : anndata.AnnData
        AnnData object with RISE decomposition results. Must contain:
        - data.varm["Pf2_C"]: Gene factors (n_genes, rank)
    ax : matplotlib.axes.Axes
        Matplotlib axes object to plot on.
    weight : float, optional (default: 0.08)
        Minimum absolute weight threshold for including genes. Genes with maximum
        absolute weight below this value across all components are filtered out.
        Higher values show fewer, more important genes.
    trim : bool, optional (default: True)
        If True, filters genes based on the weight parameter. If False, shows all genes.
    """
    rank = data.varm["Pf2_C"].shape[1]
    X = np.array(data.varm["Pf2_C"])
    yt = data.var.index.values

    if trim is True:
        max_weight = np.max(np.abs(X), axis=1)
        kept_idxs = max_weight > weight
        X = X[kept_idxs]
        yt = yt[kept_idxs]

    ind = reorder_table(X)
    X = X[ind]
    X = X / np.max(np.abs(X))
    yt = [yt[ii] for ii in ind]
    xticks = np.arange(1, rank + 1)

    sns.heatmap(
        data=X,
        xticklabels=xticks,
        yticklabels=yt,
        ax=ax,
        center=0,
        cmap=cmap,
        vmin=-1,
        vmax=1,
    )
    ax.set(xlabel="Component")


def reorder_table(projs: np.ndarray):
    """Reorder a table's rows using heirarchical clustering"""
    return seriate(pdist(projs))
