"""
Plotting functions for cell-type alignment scoring.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from ..annotation_alignment import CellTypeAlignmentResults, score_cell_type_alignment

cmap_enrichment = sns.diverging_palette(240, 10, as_cmap=True)


def plot_cell_type_alignment(
    data: anndata.AnnData | CellTypeAlignmentResults | pd.DataFrame,
    ax: Axes | Sequence[Axes] | None = None,
    cell_type_col: str = "cell_type",
    projection_key: str = "weighted_projections",
    signed: bool = False,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    annotate_significance: bool = True,
    show_metrics: bool = True,
    reorder: bool = True,
    cmap=None,
    metrics_cmap="Blues",
    random_state=None,
) -> Axes | tuple[Axes, ...]:
    """Plot cell-type alignment heatmap for RISE components.

    Visualizes per-cell-type AUROC enrichment for each component with optional
    significance stars (* for q <= alpha) and row annotations for uniqueness (tau)
    and variance explained (eta^2).

    Parameters
    ----------
    data : anndata.AnnData | CellTypeAlignmentResults | pd.DataFrame
        Alignment results or AnnData object containing RISE results.
    ax : Axes | Sequence[Axes] | None, optional
        Matplotlib Axes to plot on. Can be:
        - None: creates a new figure with appropriate subplots.
        - Single Axes: plots the main AUROC heatmap on the provided axes.
        - Pair of Axes (ax_main, ax_metrics): plots heatmap on ax_main and metrics on ax_metrics.
    cell_type_col : str, optional (default: "cell_type")
        Column name in data.obs containing cell type labels (if data is AnnData).
    projection_key : str, optional (default: "weighted_projections")
        Key in data.obsm containing projections (if data is AnnData).
    signed : bool, optional (default: False)
        If True, takes the absolute value of loadings.
    n_permutations : int, optional (default: 1000)
        Number of permutations to run (if scoring an AnnData).
    alpha : float, optional (default: 0.05)
        FDR significance threshold.
    annotate_significance : bool, optional (default: True)
        If True, marks significant cell types (q <= alpha, AUROC > 0.5) with an asterisk (*).
    show_metrics : bool, optional (default: True)
        If True and subplots are available (or ax is None), shows tau and eta^2 row annotations.
    reorder : bool, optional (default: True)
        If True, reorders components by dominant cell type.
    cmap : Colormap or str, optional
        Colormap for AUROC enrichment heatmap (defaults to blue-red diverging).
    metrics_cmap : Colormap or str, optional (default: "Blues")
        Colormap for tau and eta^2 annotations.
    random_state : int | None, optional
        Random seed for permutations.

    Returns
    -------
    Axes | tuple[Axes, ...]
        The plotted Matplotlib Axes.
    """
    if isinstance(data, CellTypeAlignmentResults):
        results = data
    elif isinstance(data, anndata.AnnData):
        results = score_cell_type_alignment(
            data=data,
            cell_types=cell_type_col,
            projection_key=projection_key,
            signed=signed,
            n_permutations=n_permutations,
            alpha=alpha,
            random_state=random_state,
        )
    elif isinstance(data, pd.DataFrame):
        # Raw enrichment dataframe
        enrichment = data
        q_values = pd.DataFrame(1.0, index=enrichment.index, columns=enrichment.columns)
        tau = pd.Series(
            [
                float(np.sum(1 - row / np.max(row)) / max(1, len(row) - 1))
                for _, row in enrichment.iterrows()
            ],
            index=enrichment.index,
        )
        eta2 = pd.Series(0.0, index=enrichment.index)
        results = CellTypeAlignmentResults(
            results=[],
            enrichment=enrichment,
            p_values=q_values,
            q_values=q_values,
            tau=tau,
            eta_squared=eta2,
            kruskal_epsilon_squared=eta2,
            significant_cell_types={},
            alpha=alpha,
        )
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    enrichment_df = results.enrichment.copy()
    q_values_df = results.q_values.copy()
    tau_series = results.tau.copy()
    eta2_series = results.eta_squared.copy()

    # Reorder components by dominant cell type if requested
    if reorder and len(enrichment_df) > 1:
        vals = enrichment_df.to_numpy()
        max_idx = np.argmax(vals, axis=1)
        max_val = vals[np.arange(vals.shape[0]), max_idx]
        order = np.lexsort((-max_val, max_idx))
        enrichment_df = enrichment_df.iloc[order]
        q_values_df = q_values_df.iloc[order]
        tau_series = tau_series.iloc[order]
        eta2_series = eta2_series.iloc[order]

    # Annotations for significance
    annot_df = None
    if annotate_significance:
        annot_mat = np.full(enrichment_df.shape, "", dtype=object)
        for i, comp in enumerate(enrichment_df.index):
            for j, ctype in enumerate(enrichment_df.columns):
                q = q_values_df.loc[comp, ctype]
                auc = enrichment_df.loc[comp, ctype]
                if q <= results.alpha and auc > 0.5:
                    annot_mat[i, j] = "*"
        annot_df = pd.DataFrame(
            annot_mat, index=enrichment_df.index, columns=enrichment_df.columns
        )

    # Set up axes
    heatmap_cmap = cmap if cmap is not None else cmap_enrichment

    ax_main: Axes
    ax_metrics: Axes | None

    if ax is None:
        if show_metrics:
            _, axes = plt.subplots(
                1,
                2,
                figsize=(
                    max(6, len(enrichment_df.columns) * 0.8 + 2),
                    max(4, len(enrichment_df) * 0.4 + 1),
                ),
                gridspec_kw={
                    "width_ratios": [len(enrichment_df.columns), 2],
                    "wspace": 0.08,
                },
            )
            ax_main = axes[0]
            ax_metrics = axes[1]
        else:
            _, ax_main = plt.subplots(
                figsize=(
                    max(5, len(enrichment_df.columns) * 0.8),
                    max(4, len(enrichment_df) * 0.4 + 1),
                )
            )
            ax_metrics = None

    elif isinstance(ax, (list, tuple, np.ndarray)) and len(ax) >= 2:
        ax_seq = cast(Sequence[Axes], ax)
        ax_main = ax_seq[0]
        ax_metrics = ax_seq[1]
    else:
        ax_main = cast(Axes, ax)
        ax_metrics = None

    # Main AUROC heatmap
    sns.heatmap(
        data=enrichment_df,
        ax=ax_main,
        cmap=heatmap_cmap,
        center=0.5,
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "AUROC (Enrichment)"},
        annot=annot_df if annotate_significance else False,
        fmt="",
        annot_kws={"size": 14, "va": "center", "ha": "center", "weight": "bold"},
    )
    ax_main.set_ylabel("Component")
    ax_main.set_xlabel("Cell Type")
    ax_main.tick_params(axis="y", rotation=0)
    ax_main.tick_params(axis="x", rotation=45)

    # Side metrics heatmap
    if ax_metrics is not None and show_metrics:
        metrics_df = pd.DataFrame(
            {
                r"$\tau$": tau_series,
                r"$\eta^2$": eta2_series,
            },
            index=enrichment_df.index,
        )
        sns.heatmap(
            data=metrics_df,
            ax=ax_metrics,
            cmap=metrics_cmap,
            vmin=0.0,
            vmax=1.0,
            cbar_kws={"label": "Score"},
            annot=True,
            fmt=".2f",
            annot_kws={"size": 9},
            yticklabels=False,
        )
        ax_metrics.set_ylabel("")
        ax_metrics.tick_params(axis="x", rotation=0)

    if ax_metrics is not None:
        return (ax_main, ax_metrics)
    return ax_main
