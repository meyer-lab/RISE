import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.colors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import datashader as ds
import datashader.transfer_functions as tf
from matplotlib.patches import Patch
import anndata


def _get_extent(points: np.ndarray) -> tuple[float, float, float, float]:
    """Compute bounds on a space with appropriate padding"""
    min_xy = np.nanmin(points, axis=0)
    assert min_xy.size == 2
    max_xy = np.nanmax(points, axis=0)

    mins = np.round(min_xy - 0.05 * (max_xy - min_xy))
    maxs = np.round(max_xy + 0.05 * (max_xy - min_xy))
    return (mins[0], maxs[0], mins[1], maxs[1])


def _to_hex(arr):
    return [matplotlib.colors.to_hex(c) for c in arr]


def points(
    points: np.ndarray,
    ax: Axes,
    labels=None,
    values=None,
    cmap=None,
    width: int = 1200,
    height: int = 1200,
    show_legend: bool = True,
    alpha=255,
) -> Axes:
    """Use datashader to plot points"""
    assert points.shape[1] == 2

    extent = _get_extent(points)
    canvas = ds.Canvas(
        plot_width=width,
        plot_height=height,
        x_range=(extent[0], extent[1]),
        y_range=(extent[2], extent[3]),
    )
    data = pd.DataFrame(points, columns=("x", "y"))

    legend_elements = None

    # Color by labels
    if labels is not None:
        assert labels.shape[0] == points.shape[0]
        if cmap is None:
            cmap = "Spectral"
        indices = np.argsort(labels)
        points = points[indices, :]
        labels = labels[indices]
        data = pd.DataFrame(points, columns=("x", "y"))

        data["label"] = pd.Categorical(labels)
        aggregation = canvas.points(data, "x", "y", agg=ds.count_cat("label"))

        unique_labels = np.unique(labels)

        num_labels = unique_labels.shape[0]
        color_key = _to_hex(plt.get_cmap(cmap)(np.linspace(0, 1, num_labels)))
        legend_elements = [
            Patch(facecolor=color_key[i], label=k) for i, k in enumerate(unique_labels)
        ]
        result = tf.shade(
            aggregation,
            color_key=color_key,
            how="eq_hist",
            alpha=alpha,
            min_alpha=255,
        )

    # Color by values
    elif values is not None:
        if cmap is None:
            cmap = "Blues"

        assert values.shape[0] == points.shape[0]

        min_val, max_val = np.min(values), np.max(values)
        bin_size = (max_val - min_val) / 255.0
        data["val_cat"] = pd.Categorical(
            np.round((values - min_val) / bin_size).astype(np.int16)
        )
        aggregation = canvas.points(data, "x", "y", agg=ds.count_cat("val_cat"))
        color_key = _to_hex(plt.get_cmap(cmap)(np.linspace(0, 1, 256)))
        result = tf.shade(
            aggregation,
            color_key=color_key,
            how="eq_hist",
            alpha=alpha,
            min_alpha=255,
        )

    # Color by density (default datashader option)
    else:
        print("yup")
        aggregation = canvas.points(data, "x", "y", agg=ds.count())
        result = tf.shade(aggregation, cmap=plt.get_cmap(cmap), alpha=alpha, how="log")

    result = tf.set_background(result, "white")

    img_rev = result.data[::-1]
    mpl_img = np.dstack(
        [img_rev & 0x0000FF, (img_rev & 0x00FF00) >> 8, (img_rev & 0xFF0000) >> 16]
    )

    ax.imshow(mpl_img)

    if show_legend and legend_elements is not None:
        ax.legend(handles=legend_elements)
    elif show_legend:
        psm = plt.pcolormesh([[0, 256], [0, 256]], cmap=cmap)
        plt.colorbar(psm, ax=ax)

    ax.set(xticks=[], yticks=[])
    return ax


def plotGeneUMAP(gene: str, decompType: str, X: anndata.AnnData, ax: Axes):
    """Scatterplot of UMAP visualization weighted by gene"""
    geneList = X[:, gene].X.toarray().flatten()
    geneList = np.clip(geneList, None, np.quantile(geneList, 0.99))
    cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
    points(X.obsm["embedding"], values=geneList, cmap=cmap, ax=ax)
    ax.set(title=f"{gene}-{decompType}-Based Decomposition")


def plotCmpUMAP(X: anndata.AnnData, cmp: int, ax: Axes):
    """Scatterplot of UMAP visualization weighted by
    projections for a component and cell state"""
    weightedProjs = X.obsm["weighted_projections"][:, cmp - 1]
    print(np.max(weightedProjs))
    print(np.min(weightedProjs))

    cmap = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
    points(X.obsm["embedding"], values=weightedProjs, cmap=cmap, ax=ax)
    ax.set(title="Cmp. " + str(cmp))


def plotLabelsUMAP(X: anndata.AnnData, labelType: str, ax: Axes, condition=None):
    """Scatterplot of UMAP visualization weighted by condition or cell type"""
    labs = X.obs[labelType]

    if condition is not None:
        labs = np.array([c if c in condition else "Other" for c in labs])

    points(
        X.obsm["embedding"],
        labels=labs,
        ax=ax,
        cmap="tab20",
        show_legend=True,
    )


def plotCmpPerCellType(X: anndata.AnnData, cmp: int, ax: Axes, outliers: bool = False):
    """Boxplot of weighted projections for one component across cell types"""
    XX = X.obsm["weighted_projections"][:, cmp - 1]
    cmpName = f"Cmp. {cmp}"

    df = pd.DataFrame({cmpName: XX, "Cell Type": X.obs["Cell Type"]})

    sns.boxplot(
        data=df,
        x=cmpName,
        y="Cell Type",
        showfliers=outliers,
        ax=ax,
    )
    maxvalue = np.max(np.abs(ax.get_xticks()))
    ax.set(
        xticks=np.linspace(-maxvalue, maxvalue, num=5), xlabel="Cell Specific Weight"
    )
    ax.set_title(cmpName)
