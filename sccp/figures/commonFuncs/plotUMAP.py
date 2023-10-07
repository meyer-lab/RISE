import seaborn as sns
from matplotlib import pyplot as plt
import umap
import numpy as np
from umap.plot import _get_embedding, _select_font_color, _datashade_points, _get_metric


def points(
    umap_object,
    labels=None,
    values=None,
    cmap="Blues",
    color_key=None,
    color_key_cmap="Spectral",
    width=1200,
    height=1200,
    show_legend=True,
    ax=None,
    alpha=None,
):
    """Copied from umap.plot.points. This just always uses datashader."""
    if labels is not None and values is not None:
        raise ValueError(
            "Conflicting options; only one of labels or values should be set"
        )

    if alpha is not None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0 and 1 inclusive")

    if isinstance(umap_object, umap.UMAP):
        points = _get_embedding(umap_object)
    else:
        points = umap_object

    if points.shape[1] != 2:
        raise ValueError("Plotting is currently only implemented for 2D embeddings")

    assert ax is not None

    # Datashader uses 0-255 as the range for alpha, with 255 as the default
    if alpha is not None:
        alpha = alpha * 255
    else:
        alpha = 255

    background = "white"

    ax = _datashade_points(
        points,
        ax,
        labels,
        values,
        cmap,
        color_key,
        color_key_cmap,
        background,
        width,
        height,
        show_legend,
        alpha,
    )

    ax.set(xticks=[], yticks=[])
    ax.text(
        0.99,
        0.01,
        "UMAP: metric={}, n_neighbors={}, min_dist={}".format(
            _get_metric(umap_object), umap_object.n_neighbors, umap_object.min_dist
        ),
        transform=ax.transAxes,
        horizontalalignment="right",
        color=_select_font_color(background),
    )

    return ax


def plotCondUMAP(conds, decomp, totalconds, umappoints, axs: list[plt.Axes]):
    """Scatterplot of UMAP visualization weighted by condition"""
    for i, cond in enumerate(conds):
        condList = np.where(np.asarray(totalconds == cond), cond, " Other Conditions")
        points(
            umappoints,
            labels=condList,
            ax=axs[i],
            color_key_cmap="tab20",
            show_legend=True,
        )
        axs[i].set(
            title=decomp + "-Based Decomposition", ylabel="UMAP2", xlabel="UMAP1"
        )


def plotGeneUMAP(
    genes: list[str],
    decomp: str,
    umappoints,
    dataDF,
    axs: list[plt.Axes]
):
    """Scatterplot of UMAP visualization weighted by gene"""
    cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)

    for i, ax in enumerate(axs):
        geneList = dataDF[:, genes[i]].X
        geneList = geneList / np.max(np.abs(geneList))
        psm = plt.pcolormesh([[0, 1], [0, 1]], cmap=cmap)
        plot = points(umappoints, values=geneList, cmap=cmap, ax=ax)
        plt.colorbar(psm, ax=plot)
        axs[i].set(
            title=f"{genes[i]}-{decomp}-Based Decomposition",
            ylabel="UMAP2",
            xlabel="UMAP1",
        )


def plotCmpUMAP(
    cmp: int, factors: np.ndarray, umappoints: umap.UMAP, allP: np.ndarray, ax: plt.Axes
):
    """Scatterplot of UMAP visualization weighted by
    projections for a component and cell state"""
    weightedProjs = (allP @ factors)[:, cmp - 1]
    weightedProjs = weightedProjs / np.max(np.abs(weightedProjs)) * 2.0

    cmap = sns.diverging_palette(240, 10, as_cmap=True, s=100)
    plot = points(umappoints, values=weightedProjs, cmap=cmap, ax=ax)

    psm = plt.pcolormesh([[-1, 1], [-1, 1]], cmap=cmap)
    plt.colorbar(psm, ax=plot, label="Cell Specific Weight")

    ax.set(ylabel="UMAP2", xlabel="UMAP1", title="Cmp. " + str(cmp))


def plotUMAP_obslabel(labels, umappoints, ax: plt.Axes):
    """Scatterplot of UMAP visualization labeled by cell type or other obs column"""
    points(umappoints, labels=labels, color_key_cmap="Paired", ax=ax)
    ax.set(
        ylabel="UMAP2",
        xlabel="UMAP1",
        title="Pf2-Based Decomposition: Label " + str(labels.name),
    )


def plotLabelAllUMAP(conditions, umappoints, ax: plt.Axes):
    """Scatterplot of UMAP visualization weighted by condition or cell type"""
    points(umappoints, labels=conditions, ax=ax, color_key_cmap="tab20", show_legend=True)
    ax.set(title="Pf2-Based Decomposition", ylabel="UMAP2", xlabel="UMAP1")


def plotCellTypeUMAP(umappoints, data, ax: plt.Axes):
    """Plots UMAP labeled by cell type"""
    points(umappoints, labels=data["Cell Type"].values, ax=ax)
    ax.set(ylabel="UMAP2", xlabel="UMAP1")


def plotCmpPerCellType(weightedprojs, cmp, ax: plt.Axes, outliers=True):
    """Boxplot of weighted projections for one component across cell types"""
    cmpName = "Cmp. " + str(cmp)
    sns.boxplot(
        data=weightedprojs[[cmpName, "Cell Type"]],
        x=cmpName,
        y="Cell Type",
        showfliers=outliers,
        ax=ax,
    )
    maxvalue = np.max(np.abs(ax.get_xticks()))
    ax.set(xlim=(-maxvalue, maxvalue), xlabel="Cell Specific Weight")
    ax.set_title(cmpName)
