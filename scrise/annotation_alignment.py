"""
Cell-type alignment scoring for RISE component projections.

Quantifies how well a RISE component's cell loadings / projections align
with annotated cell types, answering:
1. Uniqueness: Does the component concentrate on a single annotated cell type (tau)?
2. Combination alignment: Does it align with a specific subset of cell types (AUROC + FDR, eta^2)?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import anndata
import numpy as np
import pandas as pd
import scipy.stats as sp

from .alignment_stats import (
    _as_generator,
    _component_p_values,
    _validate_and_encode_cell_types,
    compute_auroc_per_cell_type,
    compute_eta_squared,
    compute_kruskal_epsilon_squared,
    compute_tau,
)


@dataclass
class ComponentAlignmentResult:
    """Alignment results for a single RISE component.

    Parameters
    ----------
    component : int | str
        Component index or label.
    enrichment : pd.Series
        AUROC per cell type distinguishing that type from all others.
    p_values : pd.Series
        Empirical permutation p-values for enrichment (AUROC > null).
    q_values : pd.Series
        Benjamini-Hochberg FDR-adjusted p-values.
    tau : float
        Tissue specificity index tau (uniqueness score in [0, 1]).
    eta_squared : float
        Proportion of loading variance explained by cell type (in [0, 1]).
    kruskal_epsilon_squared : float
        Non-parametric effect size epsilon-squared from Kruskal-Wallis.
    significant_cell_types : list[str]
        List of cell types with significant enrichment (q <= alpha and AUROC > 0.5).
    alpha : float
        Significance threshold used for q-values.
    """

    component: int | str
    enrichment: pd.Series
    p_values: pd.Series
    q_values: pd.Series
    tau: float
    eta_squared: float
    kruskal_epsilon_squared: float
    significant_cell_types: list[str] = field(default_factory=list)
    alpha: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a dictionary."""
        return {
            "component": self.component,
            "enrichment": self.enrichment.to_dict(),
            "p_values": self.p_values.to_dict(),
            "q_values": self.q_values.to_dict(),
            "tau": self.tau,
            "eta_squared": self.eta_squared,
            "kruskal_epsilon_squared": self.kruskal_epsilon_squared,
            "significant_cell_types": list(self.significant_cell_types),
            "alpha": self.alpha,
        }


@dataclass
class CellTypeAlignmentResults:
    """Alignment results across multiple RISE components.

    Parameters
    ----------
    results : list[ComponentAlignmentResult]
        List of alignment results for individual components.
    enrichment : pd.DataFrame
        Components x cell types matrix of AUROC values.
    p_values : pd.DataFrame
        Components x cell types matrix of permutation p-values.
    q_values : pd.DataFrame
        Components x cell types matrix of joint BH FDR-adjusted p-values.
    tau : pd.Series
        Tissue specificity index tau for each component.
    eta_squared : pd.Series
        Eta-squared (variance explained) for each component.
    kruskal_epsilon_squared : pd.Series
        Kruskal-Wallis epsilon-squared for each component.
    significant_cell_types : dict[int | str, list[str]]
        Mapping of component to significant cell types.
    alpha : float
        Significance threshold used for q-values.
    """

    results: list[ComponentAlignmentResult]
    enrichment: pd.DataFrame
    p_values: pd.DataFrame
    q_values: pd.DataFrame
    tau: pd.Series
    eta_squared: pd.Series
    kruskal_epsilon_squared: pd.Series
    significant_cell_types: dict[int | str, list[str]]
    alpha: float = 0.05

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame across all components."""
        top_types = []
        for comp in self.enrichment.index:
            row = self.enrichment.loc[comp]
            top_types.append(row.idxmax())

        sig_str = [
            ", ".join(self.significant_cell_types.get(comp, []))
            for comp in self.enrichment.index
        ]

        df = pd.DataFrame(
            {
                "tau": self.tau,
                "eta_squared": self.eta_squared,
                "kruskal_epsilon_squared": self.kruskal_epsilon_squared,
                "top_cell_type": top_types,
                "significant_cell_types": sig_str,
            },
            index=self.enrichment.index,
        )
        return df


_CELL_TYPE_COLUMN_CANDIDATES = (
    "cell_type",
    "CellType",
    "cell_types",
    "Cell_Type",
    "celltype",
)


def _loadings_from_anndata(data: anndata.AnnData, projection_key: str) -> np.ndarray:
    """Pull the cell-loading matrix out of ``obsm``, falling back to projections."""
    if projection_key in data.obsm:
        return np.asarray(data.obsm[projection_key])
    if projection_key == "weighted_projections" and "projections" in data.obsm:
        return np.asarray(data.obsm["projections"])
    raise KeyError(f"Could not find '{projection_key}' in data.obsm.")


def _cell_types_from_anndata(
    data: anndata.AnnData, cell_types: pd.Series | np.ndarray | str | None
) -> pd.Series:
    """Resolve cell-type labels for an AnnData: named column, guess, or literal."""
    if cell_types is None:
        for candidate in _CELL_TYPE_COLUMN_CANDIDATES:
            if candidate in data.obs:
                return pd.Series(data.obs[candidate])
        raise KeyError(
            "Cell-type column not specified and none of ['cell_type', 'CellType', 'cell_types'] found in data.obs."
        )
    if isinstance(cell_types, str):
        if cell_types not in data.obs:
            raise KeyError(f"Column '{cell_types}' not found in data.obs.")
        return pd.Series(data.obs[cell_types])
    return pd.Series(cell_types)


def _resolve_loadings_and_cell_types(
    data: anndata.AnnData | np.ndarray | pd.DataFrame,
    cell_types: pd.Series | np.ndarray | str | None,
    projection_key: str,
) -> tuple[np.ndarray, pd.Series]:
    """Normalise the three accepted input shapes to (matrix, label series)."""
    if isinstance(data, anndata.AnnData):
        return (
            _loadings_from_anndata(data, projection_key),
            _cell_types_from_anndata(data, cell_types),
        )

    loadings_matrix = (
        data.to_numpy() if isinstance(data, pd.DataFrame) else np.asarray(data)
    )
    if cell_types is None or isinstance(cell_types, str):
        raise ValueError(
            "cell_types must be provided when data is a DataFrame or array."
        )
    return loadings_matrix, pd.Series(cell_types)


def cell_type_alignment(
    loadings: np.ndarray | pd.Series,
    cell_types: pd.Series | np.ndarray,
    signed: bool = False,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    random_state: int | np.random.Generator | None = None,
    component_label: int | str = 1,
) -> ComponentAlignmentResult:
    """Score alignment of a single component's cell loadings with annotated cell types.

    Parameters
    ----------
    loadings : np.ndarray | pd.Series
        Cell loading vector for one component (shape: n_cells,).
    cell_types : pd.Series | np.ndarray
        Cell type annotations (length: n_cells).
    signed : bool, optional (default: False)
        If True, uses the absolute value of loadings (|loading|), which is appropriate
        for signed eigen-state projections (e.g. P_i @ B[:, r]).
    n_permutations : int, optional (default: 1000)
        Number of label permutations to compute empirical p-values. If 0, p-values
        are computed using the asymptotic one-sided Mann-Whitney U test.
    alpha : float, optional (default: 0.05)
        FDR significance threshold for identifying enriched cell types.
    random_state : int | np.random.Generator | None, optional (default: None)
        Random seed or Generator for permutation reproducibility.
    component_label : int | str, optional (default: 1)
        Identifier for the component.

    Returns
    -------
    ComponentAlignmentResult
        Dataclass containing AUROC enrichment, p-values, q-values, tau, eta^2,
        and significant cell types.
    """
    y = np.asarray(loadings, dtype=float)
    if signed:
        y = np.abs(y)

    codes, categories = _validate_and_encode_cell_types(cell_types)
    n_types = len(categories)
    n_cells = y.size

    if codes.size != n_cells:
        raise ValueError(
            f"Length mismatch: loadings has {n_cells} cells but cell_types has {codes.size}."
        )

    # Compute observed AUROC per cell type
    aurocs = compute_auroc_per_cell_type(y, codes, n_types)

    p_values = _component_p_values(
        y,
        codes,
        n_types,
        n_cells,
        aurocs,
        n_permutations,
        _as_generator(random_state),
    )

    # BH FDR correction across cell types for this component
    if n_types > 1:
        q_values = sp.false_discovery_control(p_values, method="bh")
    else:
        q_values = p_values.copy()

    # Scores
    tau = compute_tau(aurocs, baseline=0.0)
    eta2 = compute_eta_squared(y, codes, n_types)
    eps2 = compute_kruskal_epsilon_squared(y, codes, n_types)

    enrichment_series = pd.Series(aurocs, index=categories, name="AUROC")
    p_series = pd.Series(p_values, index=categories, name="p_value")
    q_series = pd.Series(q_values, index=categories, name="q_value")

    significant = [
        categories[k]
        for k in range(n_types)
        if q_values[k] <= alpha and aurocs[k] > 0.5
    ]

    return ComponentAlignmentResult(
        component=component_label,
        enrichment=enrichment_series,
        p_values=p_series,
        q_values=q_series,
        tau=tau,
        eta_squared=eta2,
        kruskal_epsilon_squared=eps2,
        significant_cell_types=significant,
        alpha=alpha,
    )


def _component_metrics(
    loadings_matrix: np.ndarray,
    codes: np.ndarray,
    n_types: int,
    signed: bool,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-component AUROCs, permutation p-values, tau, eta^2 and epsilon^2."""
    n_cells, n_comps = loadings_matrix.shape
    aurocs_mat = np.zeros((n_comps, n_types), dtype=float)
    p_vals_mat = np.ones((n_comps, n_types), dtype=float)
    tau_vec = np.zeros(n_comps, dtype=float)
    eta2_vec = np.zeros(n_comps, dtype=float)
    eps2_vec = np.zeros(n_comps, dtype=float)

    for comp_idx in range(n_comps):
        y = loadings_matrix[:, comp_idx].astype(float)
        if signed:
            y = np.abs(y)

        auc = compute_auroc_per_cell_type(y, codes, n_types)
        aurocs_mat[comp_idx] = auc
        tau_vec[comp_idx] = compute_tau(auc, baseline=0.0)
        eta2_vec[comp_idx] = compute_eta_squared(y, codes, n_types)
        eps2_vec[comp_idx] = compute_kruskal_epsilon_squared(y, codes, n_types)

        p_vals_mat[comp_idx] = _component_p_values(
            y, codes, n_types, n_cells, auc, n_permutations, rng
        )

    return aurocs_mat, p_vals_mat, tau_vec, eta2_vec, eps2_vec


def score_cell_type_alignment(
    data: anndata.AnnData | np.ndarray | pd.DataFrame,
    cell_types: pd.Series | np.ndarray | str | None = None,
    signed: bool = False,
    projection_key: str = "weighted_projections",
    n_permutations: int = 1000,
    alpha: float = 0.05,
    random_state: int | np.random.Generator | None = None,
) -> CellTypeAlignmentResults:
    """Score cell-type alignment across all RISE components.

    Jointly calculates per-cell-type AUROC enrichment, empirical significance with
    Benjamini-Hochberg FDR correction across all (component x cell type) tests,
    uniqueness (tau), and combination alignment (eta^2).

    Parameters
    ----------
    data : anndata.AnnData | np.ndarray | pd.DataFrame
        AnnData containing fitted RISE results, or a matrix of cell loadings
        (shape: n_cells, n_components).
    cell_types : pd.Series | np.ndarray | str | None, optional
        Cell type annotations. If data is AnnData and cell_types is a string (or None),
        looks up data.obs[cell_types] (defaults to 'cell_type' or 'CellType').
    signed : bool, optional (default: False)
        If True, takes the absolute value of loadings (|loading|).
    projection_key : str, optional (default: "weighted_projections")
        Key in data.obsm to extract loadings from when data is an AnnData.
        Defaults to 'weighted_projections', or falls back to 'projections'.
    n_permutations : int, optional (default: 1000)
        Number of permutations for null AUROC distribution.
    alpha : float, optional (default: 0.05)
        Significance threshold for FDR q-values.
    random_state : int | np.random.Generator | None, optional (default: None)
        Random seed or Generator for permutations.

    Returns
    -------
    CellTypeAlignmentResults
        Container with full results across all components.
    """
    loadings_matrix, cell_type_series = _resolve_loadings_and_cell_types(
        data, cell_types, projection_key
    )

    if loadings_matrix.ndim == 1:
        loadings_matrix = loadings_matrix[:, np.newaxis]

    n_cells, n_comps = loadings_matrix.shape
    codes, categories = _validate_and_encode_cell_types(cell_type_series)
    n_types = len(categories)

    if codes.size != n_cells:
        raise ValueError(
            f"Length mismatch: data has {n_cells} cells but cell_types has {codes.size}."
        )

    rng = _as_generator(random_state)

    component_labels = [i + 1 for i in range(n_comps)]
    aurocs_mat, p_vals_mat, tau_vec, eta2_vec, eps2_vec = _component_metrics(
        loadings_matrix, codes, n_types, signed, n_permutations, rng
    )

    # Joint BH FDR correction across all (component x cell_type) tests
    if aurocs_mat.size > 1:
        q_vals_mat = sp.false_discovery_control(
            p_vals_mat.ravel(), method="bh"
        ).reshape(p_vals_mat.shape)
    else:
        q_vals_mat = p_vals_mat.copy()

    enrichment_df = pd.DataFrame(aurocs_mat, index=component_labels, columns=categories)
    p_values_df = pd.DataFrame(p_vals_mat, index=component_labels, columns=categories)
    q_values_df = pd.DataFrame(q_vals_mat, index=component_labels, columns=categories)
    tau_series = pd.Series(tau_vec, index=component_labels, name="tau")
    eta2_series = pd.Series(eta2_vec, index=component_labels, name="eta_squared")
    eps2_series = pd.Series(
        eps2_vec, index=component_labels, name="kruskal_epsilon_squared"
    )

    results_list: list[ComponentAlignmentResult] = []
    sig_dict: dict[int | str, list[str]] = {}

    for comp_idx, comp_lbl in enumerate(component_labels):
        sig_types = [
            categories[k]
            for k in range(n_types)
            if q_vals_mat[comp_idx, k] <= alpha and aurocs_mat[comp_idx, k] > 0.5
        ]
        sig_dict[comp_lbl] = sig_types
        res = ComponentAlignmentResult(
            component=comp_lbl,
            enrichment=enrichment_df.loc[comp_lbl],
            p_values=p_values_df.loc[comp_lbl],
            q_values=q_values_df.loc[comp_lbl],
            tau=tau_vec[comp_idx],
            eta_squared=eta2_vec[comp_idx],
            kruskal_epsilon_squared=eps2_vec[comp_idx],
            significant_cell_types=sig_types,
            alpha=alpha,
        )
        results_list.append(res)

    return CellTypeAlignmentResults(
        results=results_list,
        enrichment=enrichment_df,
        p_values=p_values_df,
        q_values=q_values_df,
        tau=tau_series,
        eta_squared=eta2_series,
        kruskal_epsilon_squared=eps2_series,
        significant_cell_types=sig_dict,
        alpha=alpha,
    )


__all__ = [
    "CellTypeAlignmentResults",
    "ComponentAlignmentResult",
    "cell_type_alignment",
    "score_cell_type_alignment",
]
