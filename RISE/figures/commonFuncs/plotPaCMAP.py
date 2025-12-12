import anndata
import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.colors
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch


def _get_canvas(points: np.ndarray):
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


def ds_show(result, ax: Axes):
    result = tf.set_background(result, "white")
    img_rev = result.data[::-1]
    mpl_img = np.dstack(
        [img_rev & 0x0000FF, (img_rev & 0x00FF00) >> 8, (img_rev & 0xFF0000) >> 16]
    )

    ax.imshow(mpl_img)


def plot_gene_pacmap(gene: str, X: anndata.AnnData, ax: Axes, clip_outliers=0.9995):
    """Plot PaCMAP embedding colored by gene expression levels.
    
    This visualization overlays gene expression onto the PaCMAP embedding of cells,
    revealing which cell populations express specific genes. Useful for validating
    component interpretations by checking if marker genes align with component patterns.
    
    Parameters
    ----------
    gene : str
        Name of gene to visualize. Must be present in X.var_names.
    X : anndata.AnnData
        AnnData object with RISE decomposition results. Must contain:
        - X.obsm["X_pf2_PaCMAP"]: PaCMAP embedding coordinates (n_cells, 2)
        - X[:, gene]: Gene expression values
        - X.var["means"]: Pre-computed gene means for centering
    ax : matplotlib.axes.Axes
        Matplotlib axes object to plot on.
    clip_outliers : float, optional (default: 0.9995)
        Quantile threshold for clipping extreme expression values.
        Values above this quantile are clipped to improve visualization contrast.
    
    Examples
    --------
    >>> from RISE.figures.commonFuncs.plotPaCMAP import plot_gene_pacmap
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(figsize=(8, 8))
    >>> gene = "MS4A1"  # B cell marker
    >>> plot_gene_pacmap(gene, adata, ax=ax, clip_outliers=0.9995)
    >>> plt.tight_layout()
    >>> plt.show()
    
    See Also
    --------
    plot_labels_pacmap : Color cells by condition or cell type
    plot_wp_pacmap : Color cells by component associations
    """
    geneList = X[:, gene].to_df().values

    geneList = np.clip(geneList, None, np.quantile(geneList, clip_outliers))
    cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)

    values = geneList

    points = np.array(X.obsm["X_pf2_PaCMAP"])

    canvas = _get_canvas(points)
    data = pd.DataFrame(points, columns=("x", "y"))

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
    ax.set(title=f"{gene}")


def plot_wp_pacmap(X: anndata.AnnData, cmp: int, ax: Axes, cbarMax: float = 1.0):
    """Plot PaCMAP embedding colored by weighted projections for a component.
    
    This visualization shows which cells contribute most strongly to a specific
    component by coloring them according to their weighted projections. Cells with
    high weighted projections (bright colors) are most representative of that
    component's expression pattern.
    
    Parameters
    ----------
    X : anndata.AnnData
        AnnData object with RISE decomposition results. Must contain:
        - X.obsm["X_pf2_PaCMAP"]: PaCMAP embedding coordinates (n_cells, 2)
        - X.obsm["weighted_projections"]: Weighted cell projections (n_cells, rank)
    cmp : int
        Component number to visualize (1-indexed). For example, cmp=10 shows
        the cell associations for component 10.
    ax : matplotlib.axes.Axes
        Matplotlib axes object to plot on.
    cbarMax : float, optional (default: 1.0)
        Maximum value for the color scale. Values are normalized to [-cbarMax, cbarMax].
        Lower values increase contrast for components with weaker associations.
    
    Examples
    --------
    >>> from RISE.figures.commonFuncs.plotPaCMAP import plot_wp_pacmap
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(figsize=(8, 8))
    >>> plot_wp_pacmap(adata, cmp=10, ax=ax, cbarMax=0.9)
    >>> plt.tight_layout()
    >>> plt.show()
    
    See Also
    --------
    plot_labels_pacmap : Color cells by condition or cell type
    plot_gene_pacmap : Color cells by gene expression
    """
    values = X.obsm["weighted_projections"][:, cmp - 1]
    points = X.obsm["X_pf2_PaCMAP"]

    cmap = sns.diverging_palette(240, 10, as_cmap=True)

    canvas = _get_canvas(points)
    data = pd.DataFrame(points, columns=("x", "y"))

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
    cmap: str = "tab20",
    color_key=None,
):
    """Plot PaCMAP embedding colored by categorical labels (cell type or condition).
    
    This visualization shows the overall structure of the cell embedding, revealing
    how cells cluster by cell type, experimental condition, or other categorical
    metadata. Useful for understanding the biological organization captured by RISE.
    
    Parameters
    ----------
    X : anndata.AnnData
        AnnData object with RISE decomposition results. Must contain:
        - X.obsm["X_pf2_PaCMAP"]: PaCMAP embedding coordinates (n_cells, 2)
        - X.obs[labelType]: Categorical labels for coloring cells
    labelType : str
        Name of column in X.obs containing categorical labels to color by.
        Common values: "Cell Type", "Condition", "Sample", etc.
    ax : matplotlib.axes.Axes
        Matplotlib axes object to plot on.
    condition : list of str, optional (default: None)
        If provided, only highlights cells from these specific conditions/labels.
        All other cells are labeled as "Other".
    cmap : str, optional (default: "tab20")
        Matplotlib colormap name for coloring categories.
    color_key : list, optional (default: None)
        Custom list of colors for categories. If None, uses cmap.
    
    Examples
    --------
    >>> from RISE.figures.commonFuncs.plotPaCMAP import plot_labels_pacmap
    >>> import matplotlib.pyplot as plt
    >>> # Color by cell type
    >>> fig, ax = plt.subplots(figsize=(8, 8))
    >>> plot_labels_pacmap(adata, labelType="Cell Type", ax=ax)
    >>> plt.tight_layout()
    >>> plt.show()
    
    See Also
    --------
    plot_gene_pacmap : Color cells by gene expression
    plot_wp_pacmap : Color cells by component associations
    """
    labels = X.obs[labelType]

    if condition is not None:
        labels = pd.Series([c if c in condition else "Other" for c in labels])
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


def assign_labels(ax):
    """Assign labels to plot"""
    ax.set(xlabel="PaCMAP1", ylabel="PaCMAP2", xticks=[], yticks=[])
    return ax
