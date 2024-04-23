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
from scipy.sparse import spmatrix


def _get_canvas(points: np.ndarray) -> ds.Canvas:
    """Compute bounds on a space with appropriate padding"""
    min_xy = np.nanmin(points, axis=0)
    assert min_xy.size == 2
    max_xy = np.nanmax(points, axis=0)

    mins = np.round(min_xy - 0.05 * (max_xy - min_xy))
    maxs = np.round(max_xy + 0.05 * (max_xy - min_xy))

    canvas = ds.Canvas(
        plot_width=300,
        plot_height=300,
        x_range=(mins[0], maxs[0]),
        y_range=(mins[1], maxs[1]),
    )

    return canvas


def _to_hex(arr):
    return [matplotlib.colors.to_hex(c) for c in arr]


def ds_show(result, ax):
    result = tf.set_background(result, "white")
    img_rev = result.data[::-1]
    mpl_img = np.dstack(
        [img_rev & 0x0000FF, (img_rev & 0x00FF00) >> 8, (img_rev & 0xFF0000) >> 16]
    )

    ax.imshow(mpl_img)


def plot_gene_pacmap(gene: str, decompType: str, X: anndata.AnnData, ax: Axes):
    """Scatterplot of PaCMAP visualization weighted by gene"""
    geneList = X[:, gene].X
    if isinstance(geneList, spmatrix):
        geneList = geneList.toarray()

    geneList = np.clip(geneList, None, np.quantile(geneList, 0.99))
    cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)

    values = geneList

    points = np.array(X.obsm["X_pf2_PaCMAP"])

    canvas = _get_canvas(points)
    data = pd.DataFrame(points, columns=("x", "y"))

    # Color by values
    values -= np.min(values)
    values /= np.max(values)
    data["val_cat"] = values
    result = tf.shade(
        agg=canvas.points(data, "x", "y", agg=ds.mean("val_cat")),
        cmap=cmap,
        span=(0, 1),
        how="linear",
        min_alpha=255,
    )

    ds_show(result, ax)

    psm = plt.pcolormesh([[0, 1], [0, 1]], cmap=cmap)
    plt.colorbar(psm, ax=ax)

    ax = assign_labels(ax)
    ax.set(title=f"{gene}-{decompType}-Based Decomposition")


def plot_wp_pacmap(X: anndata.AnnData, cmp: int, ax: Axes, cbarMax: float = 1.0):
    """Scatterplot of UMAP visualization weighted by
    projections for a component and eigenstate"""
    values = X.obsm["weighted_projections"][:, cmp - 1]
    points = X.obsm["X_pf2_PaCMAP"]

    cmap = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)

    canvas = _get_canvas(points)
    data = pd.DataFrame(points, columns=("x", "y"))

    # Color by values
    values /= np.max(np.abs(values))

    data["val_cat"] = values
    result = tf.shade(
        agg=canvas.points(data, "x", "y", agg=ds.mean("val_cat")),
        cmap=cmap,
        span=(-cbarMax, cbarMax),
        how="linear",
        alpha=255,
        min_alpha=255,
    )

    ds_show(result, ax)

    psm = plt.pcolormesh([[-cbarMax, cbarMax], [-cbarMax, cbarMax]], cmap=cmap)
    plt.colorbar(psm, ax=ax)
    ax.set(title="Cmp. " + str(cmp))
    ax = assign_labels(ax)


def plot_labels_pacmap(
    X: anndata.AnnData,
    labelType: str,
    ax: Axes,
    condition=None,
    cmap="tab20",
    color_key=None,
):
    """Scatterplot of UMAP visualization weighted by condition or cell type"""
    labels = X.obs[labelType]

    if condition is not None:
        labels = pd.Series([c if c in condition else "Z Other" for c in labels])
    if labels.dtype == "category":
        labels = labels.cat.set_categories(
            np.sort(labels.cat.categories.values), ordered=True
        )
    indices = np.argsort(labels)

    points = X.obsm["X_pf2_PaCMAP"][indices, :]
    labels = labels.iloc[indices]

    canvas = _get_canvas(points)
    data = pd.DataFrame(points, columns=("x", "y"))

    data["label"] = pd.Categorical(labels)
    aggregation = canvas.points(data, "x", "y", agg=ds.count_cat("label"))

    unique_labels = np.unique(labels)
    num_labels = unique_labels.shape[0]
    if color_key is None:
        color_key = _to_hex(plt.get_cmap(cmap)(np.linspace(0, 1, num_labels)))
    legend_elements = [
        Patch(facecolor=color_key[i], label=k) for i, k in enumerate(unique_labels)
    ]
    result = tf.shade(
        aggregation,
        color_key=color_key,
        how="eq_hist",
        min_alpha=255,
    )

    ds_show(result, ax)
    ax.legend(handles=legend_elements)
    ax = assign_labels(ax)


def plot_wp_per_celltype(
    X: anndata.AnnData, cmp: int, ax: Axes, outliers: bool = False, cellType="Cell Type"
):
    """Boxplot of weighted projections for one component across cell types"""
    XX = X.obsm["weighted_projections"][:, cmp - 1]
    cmpName = f"Cmp. {cmp}"

    df = pd.DataFrame({cmpName: XX, "Cell Type": X.obs[cellType].to_numpy()})

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


def assign_labels(ax):
    ax.set(xlabel="PaCMAP1", ylabel="PaCMAP2", xticks=[], yticks=[])
    return ax
