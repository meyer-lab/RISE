"""Statistics behind cell-type alignment scoring.

Pure numeric routines -- AUROC enrichment, uniqueness (tau), effect sizes,
and the two p-value nulls -- with no AnnData dependency, so they can be
tested and reused on plain arrays. :mod:`scrise.annotation_alignment` adds
the AnnData plumbing and assembles the results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as sp


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


def _permutation_p_values(
    y: np.ndarray,
    codes: np.ndarray,
    n_types: int,
    n_cells: int,
    aurocs: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Empirical one-sided p-values against a rank-permutation null."""
    ranks = sp.rankdata(y, method="average")
    counts = np.bincount(codes, minlength=n_types).astype(float)
    null_counts = np.zeros(n_types, dtype=int)

    for _ in range(n_permutations):
        perm_ranks = rng.permutation(ranks)
        perm_sums = np.bincount(codes, weights=perm_ranks, minlength=n_types).astype(
            float
        )
        for k in range(n_types):
            n1 = counts[k]
            n0 = n_cells - n1
            if n1 > 0 and n0 > 0:
                u_null = perm_sums[k] - n1 * (n1 + 1.0) / 2.0
                auc_null = u_null / (n1 * n0)
                if auc_null >= aurocs[k]:
                    null_counts[k] += 1

    return (1.0 + null_counts) / (1.0 + n_permutations)


def _asymptotic_p_values(y: np.ndarray, codes: np.ndarray, n_types: int) -> np.ndarray:
    """One-sided Mann-Whitney U p-values, used when ``n_permutations == 0``."""
    p_values = np.ones(n_types, dtype=float)
    for k in range(n_types):
        pos = y[codes == k]
        neg = y[codes != k]
        if pos.size > 0 and neg.size > 0:
            res = sp.mannwhitneyu(pos, neg, alternative="greater")
            p_values[k] = float(res.pvalue)
    return p_values


def _component_p_values(
    y: np.ndarray,
    codes: np.ndarray,
    n_types: int,
    n_cells: int,
    aurocs: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Per-cell-type p-values for one component's loadings.

    Shared by :func:`cell_type_alignment` and
    :func:`score_cell_type_alignment`, which compute this identically. A
    single cell type (or no cells) leaves every p-value at 1.0, since there
    is nothing to be enriched against.
    """
    if n_types <= 1 or n_cells == 0:
        return np.ones(n_types, dtype=float)
    if n_permutations > 0:
        return _permutation_p_values(
            y, codes, n_types, n_cells, aurocs, n_permutations, rng
        )
    if n_permutations == 0:
        return _asymptotic_p_values(y, codes, n_types)
    return np.ones(n_types, dtype=float)


def _as_generator(
    random_state: int | np.random.Generator | None,
) -> np.random.Generator:
    """Accept a seed, an existing Generator, or None."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)
