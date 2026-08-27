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


def compute_auroc_per_cell_type(
    loadings: np.ndarray,
    cell_type_codes: np.ndarray,
    n_types: int,
) -> np.ndarray:
    """Compute AUROC for each cell type vs all other cells.

    Parameters
    ----------
    loadings : np.ndarray
        1D array of cell loadings of shape (n_cells,).
    cell_type_codes : np.ndarray
        1D array of integer cell type assignments in [0, n_types - 1].
    n_types : int
        Total number of unique cell types.

    Returns
    -------
    np.ndarray
        1D array of AUROC values for each cell type of shape (n_types,).
    """
    n_cells = loadings.size
    if n_cells == 0 or n_types <= 1:
        return np.full(n_types, 0.5, dtype=float)

    ranks = sp.rankdata(loadings, method="average")
    counts = np.bincount(cell_type_codes, minlength=n_types).astype(float)
    rank_sums = np.bincount(cell_type_codes, weights=ranks, minlength=n_types).astype(
        float
    )

    aurocs = np.full(n_types, 0.5, dtype=float)
    for k in range(n_types):
        n1 = counts[k]
        n0 = n_cells - n1
        if n1 > 0 and n0 > 0:
            u1 = rank_sums[k] - n1 * (n1 + 1.0) / 2.0
            aurocs[k] = u1 / (n1 * n0)

    return aurocs


def compute_tau(
    enrichment_scores: np.ndarray | pd.Series,
    baseline: float = 0.0,
) -> float:
    """Compute tissue specificity index tau (Yanai et al., 2005).

    tau = sum(1 - x_hat) / (n_types - 1), where x_hat = x / max(x).

    When computed over AUROC enrichment values:
    - tau -> 1 indicates the component is specific/private to a single cell type.
    - tau -> 0 indicates the component is evenly distributed across cell types.

    Parameters
    ----------
    enrichment_scores : np.ndarray | pd.Series
        1D array of enrichment values (e.g. AUROC) across cell types.
    baseline : float, optional (default: 0.0)
        Baseline value subtracted before calculating tau. Values below baseline are clipped to 0.

    Returns
    -------
    float
        Tau index in [0.0, 1.0].
    """
    x = np.asarray(enrichment_scores, dtype=float)
    n = x.size
    if n <= 1:
        return 0.0

    if baseline > 0.0:
        x = np.maximum(0.0, x - baseline)

    max_val = np.nanmax(x)
    if max_val <= 0.0 or not np.isfinite(max_val):
        return 0.0

    x_hat = x / max_val
    tau = np.sum(1.0 - x_hat) / (n - 1.0)
    return float(np.clip(tau, 0.0, 1.0))


def compute_eta_squared(
    loadings: np.ndarray,
    cell_type_codes: np.ndarray,
    n_types: int,
) -> float:
    """Compute omnibus eta-squared (loading ~ cell_type).

    Proportion of loading variance explained by cell-type identity.

    Parameters
    ----------
    loadings : np.ndarray
        1D array of cell loadings (shape: n_cells,).
    cell_type_codes : np.ndarray
        1D array of integer cell type assignments (shape: n_cells,).
    n_types : int
        Number of unique cell types.

    Returns
    -------
    float
        Eta-squared value in [0.0, 1.0].
    """
    y = np.asarray(loadings, dtype=float)
    n_cells = y.size
    if n_cells <= 1 or n_types <= 1:
        return 0.0

    y_mean = np.mean(y)
    ss_total = np.sum((y - y_mean) ** 2)
    if ss_total <= 0.0:
        return 0.0

    counts = np.bincount(cell_type_codes, minlength=n_types).astype(float)
    valid = counts > 0
    sums = np.bincount(cell_type_codes, weights=y, minlength=n_types)

    means = np.zeros(n_types, dtype=float)
    means[valid] = sums[valid] / counts[valid]

    ss_between = np.sum(counts[valid] * (means[valid] - y_mean) ** 2)
    eta2 = ss_between / ss_total
    return float(np.clip(eta2, 0.0, 1.0))


def compute_kruskal_epsilon_squared(
    loadings: np.ndarray,
    cell_type_codes: np.ndarray,
    n_types: int,
) -> float:
    """Compute Kruskal-Wallis epsilon-squared effect size.

    Non-parametric measure of association between loading and cell type.

    Parameters
    ----------
    loadings : np.ndarray
        1D array of cell loadings (shape: n_cells,).
    cell_type_codes : np.ndarray
        1D array of integer cell type assignments (shape: n_cells,).
    n_types : int
        Number of unique cell types.

    Returns
    -------
    float
        Epsilon-squared value in [0.0, 1.0].
    """
    y = np.asarray(loadings, dtype=float)
    n_cells = y.size
    if n_cells <= 1 or n_types <= 1:
        return 0.0

    groups = [
        y[cell_type_codes == k]
        for k in range(n_types)
        if np.sum(cell_type_codes == k) > 0
    ]
    if len(groups) <= 1:
        return 0.0

    try:
        stat, _ = sp.kruskal(*groups)
        if not np.isfinite(stat) or stat < 0:
            return 0.0
        eps2 = stat / (n_cells - 1.0)
        return float(np.clip(eps2, 0.0, 1.0))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _validate_and_encode_cell_types(
    cell_types: pd.Series | np.ndarray | list,
) -> tuple[np.ndarray, list[str]]:
    """Encode cell types to integer codes and return category names."""
    if isinstance(cell_types, pd.Series) and isinstance(
        cell_types.dtype, pd.CategoricalDtype
    ):
        categories = list(cell_types.cat.categories)
        codes = cell_types.cat.codes.to_numpy()
        # If there are unused categories or NaN, retain only observed categories
        observed = np.unique(codes[codes >= 0])
        if len(observed) < len(categories):
            # Remap to dense 0..K-1
            remap = {old: new for new, old in enumerate(observed)}
            categories = [categories[old] for old in observed]
            codes = np.array([remap.get(c, -1) for c in codes], dtype=int)
        return codes, categories

    s = pd.Series(cell_types)
    cat = pd.Categorical(s)
    categories = [str(c) for c in cat.categories]
    codes = cat.codes
    return codes, categories


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

    # Compute permutation p-values
    p_values = np.ones(n_types, dtype=float)
    if n_permutations > 0 and n_types > 1 and n_cells > 0:
        rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        ranks = sp.rankdata(y, method="average")
        counts = np.bincount(codes, minlength=n_types).astype(float)
        null_counts = np.zeros(n_types, dtype=int)

        for _ in range(n_permutations):
            perm_ranks = rng.permutation(ranks)
            perm_sums = np.bincount(
                codes, weights=perm_ranks, minlength=n_types
            ).astype(float)
            for k in range(n_types):
                n1 = counts[k]
                n0 = n_cells - n1
                if n1 > 0 and n0 > 0:
                    u_null = perm_sums[k] - n1 * (n1 + 1.0) / 2.0
                    auc_null = u_null / (n1 * n0)
                    if auc_null >= aurocs[k]:
                        null_counts[k] += 1

        p_values = (1.0 + null_counts) / (1.0 + n_permutations)
    elif n_permutations == 0 and n_types > 1 and n_cells > 0:
        # Asymptotic one-sided Mann-Whitney test
        for k in range(n_types):
            pos = y[codes == k]
            neg = y[codes != k]
            if pos.size > 0 and neg.size > 0:
                res = sp.mannwhitneyu(pos, neg, alternative="greater")
                p_values[k] = float(res.pvalue)

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
    # Extract loadings and cell_types
    if isinstance(data, anndata.AnnData):
        if projection_key in data.obsm:
            loadings_matrix = np.asarray(data.obsm[projection_key])
        elif projection_key == "weighted_projections" and "projections" in data.obsm:
            loadings_matrix = np.asarray(data.obsm["projections"])
        else:
            raise KeyError(f"Could not find '{projection_key}' in data.obsm.")

        if cell_types is None:
            for candidate in [
                "cell_type",
                "CellType",
                "cell_types",
                "Cell_Type",
                "celltype",
            ]:
                if candidate in data.obs:
                    cell_type_series = data.obs[candidate]
                    break
            else:
                raise KeyError(
                    "Cell-type column not specified and none of ['cell_type', 'CellType', 'cell_types'] found in data.obs."
                )
        elif isinstance(cell_types, str):
            if cell_types not in data.obs:
                raise KeyError(f"Column '{cell_types}' not found in data.obs.")
            cell_type_series = data.obs[cell_types]
        else:
            cell_type_series = pd.Series(cell_types)
    elif isinstance(data, pd.DataFrame):
        loadings_matrix = data.to_numpy()
        if cell_types is None or isinstance(cell_types, str):
            raise ValueError(
                "cell_types must be provided when data is a DataFrame or array."
            )
        cell_type_series = pd.Series(cell_types)
    else:
        loadings_matrix = np.asarray(data)
        if cell_types is None or isinstance(cell_types, str):
            raise ValueError(
                "cell_types must be provided when data is a DataFrame or array."
            )
        cell_type_series = pd.Series(cell_types)

    if loadings_matrix.ndim == 1:
        loadings_matrix = loadings_matrix[:, np.newaxis]

    n_cells, n_comps = loadings_matrix.shape
    codes, categories = _validate_and_encode_cell_types(cell_type_series)
    n_types = len(categories)

    if codes.size != n_cells:
        raise ValueError(
            f"Length mismatch: data has {n_cells} cells but cell_types has {codes.size}."
        )

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    component_labels = [i + 1 for i in range(n_comps)]
    aurocs_mat = np.zeros((n_comps, n_types), dtype=float)
    p_vals_mat = np.ones((n_comps, n_types), dtype=float)
    tau_vec = np.zeros(n_comps, dtype=float)
    eta2_vec = np.zeros(n_comps, dtype=float)
    eps2_vec = np.zeros(n_comps, dtype=float)

    counts = np.bincount(codes, minlength=n_types).astype(float)

    for comp_idx in range(n_comps):
        y = loadings_matrix[:, comp_idx].astype(float)
        if signed:
            y = np.abs(y)

        auc = compute_auroc_per_cell_type(y, codes, n_types)
        aurocs_mat[comp_idx] = auc
        tau_vec[comp_idx] = compute_tau(auc, baseline=0.0)
        eta2_vec[comp_idx] = compute_eta_squared(y, codes, n_types)
        eps2_vec[comp_idx] = compute_kruskal_epsilon_squared(y, codes, n_types)

        if n_permutations > 0 and n_types > 1 and n_cells > 0:
            ranks = sp.rankdata(y, method="average")
            null_counts = np.zeros(n_types, dtype=int)
            for _ in range(n_permutations):
                perm_ranks = rng.permutation(ranks)
                perm_sums = np.bincount(
                    codes, weights=perm_ranks, minlength=n_types
                ).astype(float)
                for k in range(n_types):
                    n1 = counts[k]
                    n0 = n_cells - n1
                    if n1 > 0 and n0 > 0:
                        u_null = perm_sums[k] - n1 * (n1 + 1.0) / 2.0
                        auc_null = u_null / (n1 * n0)
                        if auc_null >= auc[k]:
                            null_counts[k] += 1
            p_vals_mat[comp_idx] = (1.0 + null_counts) / (1.0 + n_permutations)
        elif n_permutations == 0 and n_types > 1 and n_cells > 0:
            for k in range(n_types):
                pos = y[codes == k]
                neg = y[codes != k]
                if pos.size > 0 and neg.size > 0:
                    res = sp.mannwhitneyu(pos, neg, alternative="greater")
                    p_vals_mat[comp_idx, k] = float(res.pvalue)

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
