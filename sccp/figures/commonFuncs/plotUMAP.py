import seaborn as sns
import matplotlib.colors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
from matplotlib.patches import Patch
import anndata


def _red(x):
    return (x & 0xFF0000) >> 16


def _green(x):
    return (x & 0x00FF00) >> 8


def _blue(x):
    return x & 0x0000FF


def _get_extent(points):
    """Compute bounds on a space with appropriate padding"""
    min_x = np.nanmin(points[:, 0])
    max_x = np.nanmax(points[:, 0])
    min_y = np.nanmin(points[:, 1])
    max_y = np.nanmax(points[:, 1])

    extent = (
        np.round(min_x - 0.05 * (max_x - min_x)),
        np.round(max_x + 0.05 * (max_x - min_x)),
        np.round(min_y - 0.05 * (max_y - min_y)),
        np.round(max_y + 0.05 * (max_y - min_y)),
    )

    return extent


def _to_hex(arr):
    return [matplotlib.colors.to_hex(c) for c in arr]


def _embed_datashader_in_an_axis(datashader_image, ax):
    img_rev = datashader_image.data[::-1]
    mpl_img = np.dstack([_blue(img_rev), _green(img_rev), _red(img_rev)])
    ax.imshow(mpl_img)
    return ax


def _datashade_points(
    points,
    ax=None,
    labels=None,
    values=None,
    cmap="Blues",
    color_key=None,
    color_key_cmap="Spectral",
    background="white",
    width=800,
    height=800,
    show_legend=True,
    alpha=255,
):
    """Use datashader to plot points"""
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
        if labels.shape[0] != points.shape[0]:
            raise ValueError(
                "Labels must have a label for "
                "each sample (size mismatch: {} {})".format(
                    labels.shape[0], points.shape[0]
                )
            )

        data["label"] = pd.Categorical(labels)
        aggregation = canvas.points(data, "x", "y", agg=ds.count_cat("label"))
        if color_key is None and color_key_cmap is None:
            result = tf.shade(aggregation, how="eq_hist", alpha=alpha, min_alpha=255)
        elif color_key is None:
            unique_labels = np.unique(labels)
            num_labels = unique_labels.shape[0]
            color_key = _to_hex(
                plt.get_cmap(color_key_cmap)(np.linspace(0, 1, num_labels))
            )
            legend_elements = [
                Patch(facecolor=color_key[i], label=k)
                for i, k in enumerate(unique_labels)
            ]
            result = tf.shade(
                aggregation,
                color_key=color_key,
                how="eq_hist",
                alpha=alpha,
                min_alpha=255,
            )
        else:
            legend_elements = [
                Patch(facecolor=color_key[k], label=k) for k in color_key.keys()
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
        if values.shape[0] != points.shape[0]:
            raise ValueError(
                "Values must have a value for "
                "each sample (size mismatch: {} {})".format(
                    values.shape[0], points.shape[0]
                )
            )
        unique_values = np.unique(values)
        if unique_values.shape[0] >= 256:
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
        else:
            data["val_cat"] = pd.Categorical(values)
            aggregation = canvas.points(data, "x", "y", agg=ds.count_cat("val_cat"))
            color_key_cols = _to_hex(
                plt.get_cmap(cmap)(np.linspace(0, 1, unique_values.shape[0]))
            )
            color_key = dict(zip(unique_values, color_key_cols))
            result = tf.shade(
                aggregation,
                color_key=color_key,
                how="eq_hist",
                alpha=alpha,
                min_alpha=255,
            )

    # Color by density (default datashader option)
    else:
        aggregation = canvas.points(data, "x", "y", agg=ds.count())
        result = tf.shade(aggregation, cmap=plt.get_cmap(cmap), alpha=alpha, how="log")

    if background is not None:
        result = tf.set_background(result, background)

    if ax is not None:
        _embed_datashader_in_an_axis(result, ax)
        if show_legend and legend_elements is not None:
            ax.legend(handles=legend_elements)
        return ax
    else:
        return result


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

    return ax


def plotGeneUMAP(gene: str, decompType: str, X: anndata.AnnData, ax: Axes):
    """Scatterplot of UMAP visualization weighted by gene"""
    geneList = X[:, X.var_names.isin([gene])].X.flatten()
    geneList = geneList + np.min(geneList)
    geneList /= np.max(geneList)
    geneList = np.clip(geneList, None, np.quantile(geneList, 0.99))
    cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
    plot = points(X.obsm["embedding"], values=geneList, cmap=cmap, ax=ax)
    psm = plt.pcolormesh([[0, 1], [0, 1]], cmap=cmap)
    colorbar = plt.colorbar(psm, ax=plot)
    ax.set(
        title=f"{gene}-{decompType}-Based Decomposition",
        ylabel="UMAP2",
        xlabel="UMAP1",
    )


def plotCondUMAP(condition: str, decompType: str, XX: anndata.AnnData, ax: Axes):
    """Scatterplot of UMAP visualization weighted by condition"""
    X = XX.copy()
    X.obs["Condition"] = np.array(
        [c if c in condition else " Other Conditions" for c in X.obs["Condition"]]
    )
    points(
        X.obsm["embedding"],
        labels=X.obs["Condition"],
        ax=ax,
        color_key_cmap="tab20",
        show_legend=True,
    )
    ax.set(title=f"{decompType}-Based Decomposition", ylabel="UMAP2", xlabel="UMAP1")


def plotCmpUMAP(X, cmp: int, ax: Axes):
    """Scatterplot of UMAP visualization weighted by
    projections for a component and cell state"""
    weightedProjs = X.obsm["weighted_projections"]
    weightedProjs = weightedProjs[:, cmp - 1]
    weightedProjs = weightedProjs / np.max(np.abs(weightedProjs)) * 2.0

    cmap = sns.diverging_palette(240, 10, as_cmap=True, s=100)
    plot = points(X.obsm["embedding"], values=weightedProjs, cmap=cmap, ax=ax)

    psm = plt.pcolormesh([[-1, 1], [-1, 1]], cmap=cmap)
    plt.colorbar(psm, ax=plot, label="Cell Specific Weight")
    ax.set(ylabel="UMAP2", xlabel="UMAP1", title="Cmp. " + str(cmp))


def plotUMAP_obslabel(labels, umappoints: np.ndarray, ax: Axes):
    """Scatterplot of UMAP visualization labeled by cell type or other obs column"""
    points(umappoints, labels=labels, color_key_cmap="Paired", ax=ax)
    ax.set(
        ylabel="UMAP2",
        xlabel="UMAP1",
        title="Pf2-Based Decomposition: Label " + str(labels.name),
    )


def plotAllLabelsUMAP(X: anndata.AnnData, labelType: str, ax: Axes):
    """Scatterplot of UMAP visualization weighted by condition or cell type"""
    points(
        X.obsm["embedding"],
        labels=X.obs[labelType],
        ax=ax,
        color_key_cmap="tab20",
        show_legend=True,
    )
    ax.set(title="Pf2-Based Decomposition", ylabel="UMAP2", xlabel="UMAP1")


def plotCmpPerCellType(X: anndata.AnnData, cmp: int, ax: Axes, outliers=False):
    """Boxplot of weighted projections for one component across cell types"""
    XX = X.obsm["weighted_projections"]
    XX = XX[:, cmp - 1]
    cmpName = f"Cmp. {cmp}"
    cellTypes = X.obs["Cell Type"]
    df = pd.DataFrame(
        data=np.transpose(np.vstack((XX, cellTypes))), columns=[cmpName, "Cell Type"]
    )

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
