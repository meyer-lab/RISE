"""
Test energy-based component ordering and sign canonicalization.
"""

import anndata
import numpy as np

from ..factorization import (
    canonical_component_signs,
    match_components_across_ranks,
    order_components_by_energy,
)


def test_canonical_component_signs_flips_negative_dominant_columns():
    """The largest-magnitude entry of each column should end up positive."""
    C = np.array(
        [
            [1.0, -5.0],
            [2.0, 3.0],
            [-4.0, 1.0],
        ]
    )
    signs = canonical_component_signs(C)

    C_signed = C * signs
    max_idx = np.argmax(np.abs(C_signed), axis=0)
    assert np.all(C_signed[max_idx, np.arange(C.shape[1])] > 0)


def test_canonical_component_signs_leaves_positive_dominant_columns_alone():
    C = np.array([[3.0], [1.0], [-2.0]])
    signs = canonical_component_signs(C)
    assert signs[0] == 1.0


def _mock_adata(A, B, C, weights, projections):
    n_cells = projections.shape[0]
    n_genes = C.shape[0]
    n_conditions = A.shape[0]

    obs = {"Condition": [f"cond_{i % n_conditions}" for i in range(n_cells)]}
    var = {"gene_name": [f"gene_{j}" for j in range(n_genes)]}

    return anndata.AnnData(
        obs=obs,
        var=var,
        uns={"Pf2_A": A, "Pf2_B": B, "Pf2_weights": weights},
        varm={"Pf2_C": C},
        obsm={"projections": projections},
    )


def test_order_components_by_energy_descending():
    """Components should be reordered from highest to lowest energy."""
    rng = np.random.default_rng(0)
    n_cells, n_genes, n_conditions, rank = 50, 30, 8, 4

    A = rng.normal(size=(n_conditions, rank))
    C = rng.normal(size=(n_genes, rank))
    B = rng.normal(size=(rank, rank))
    weights = rng.random(rank)
    projections, _ = np.linalg.qr(rng.normal(size=(n_cells, rank)))

    # Make the energy ordering unambiguous by rescaling columns.
    scales = np.array([0.01, 10.0, 1.0, 5.0])
    A = A * scales
    expected_energy_order = np.argsort(
        np.linalg.norm(A, axis=0) * np.linalg.norm(C, axis=0)
    )[::-1]

    adata = _mock_adata(A.copy(), B.copy(), C.copy(), weights.copy(), projections)
    ordered = order_components_by_energy(adata)

    new_energy = np.linalg.norm(ordered.uns["Pf2_A"], axis=0) * np.linalg.norm(
        ordered.varm["Pf2_C"], axis=0
    )
    assert np.all(np.diff(new_energy) <= 1e-8)

    # Check that the columns were permuted as expected (up to sign).
    for new_idx, old_idx in enumerate(expected_energy_order):
        col_C_new = ordered.varm["Pf2_C"][:, new_idx]
        col_C_old = C[:, old_idx]
        cos = np.dot(col_C_new, col_C_old) / (
            np.linalg.norm(col_C_new) * np.linalg.norm(col_C_old)
        )
        assert abs(abs(cos) - 1.0) < 1e-8


def test_order_components_by_energy_preserves_reconstruction():
    """Reordering (and sign-flipping) must not change weighted_projections
    up to the corresponding permutation and consistent sign."""
    rng = np.random.default_rng(1)
    n_cells, n_genes, n_conditions, rank = 40, 25, 6, 3

    A = rng.normal(size=(n_conditions, rank))
    C = rng.normal(size=(n_genes, rank))
    B = rng.normal(size=(rank, rank))
    weights = rng.random(rank)
    projections, _ = np.linalg.qr(rng.normal(size=(n_cells, rank)))

    adata = _mock_adata(A.copy(), B.copy(), C.copy(), weights.copy(), projections)

    before_wp = projections @ B

    ordered = order_components_by_energy(adata)

    after_wp = ordered.obsm["weighted_projections"]

    # Column r of after_wp, weighted by A/C for that component, should match
    # some permuted (and consistently signed) column of before_wp.
    energy = np.linalg.norm(A, axis=0) * np.linalg.norm(C, axis=0)
    order = np.argsort(energy)[::-1]

    # B itself is left as the unflipped sign reference, so weighted_projections
    # (= projections @ B) is simply permuted by the energy ordering.
    expected_after_wp = before_wp[:, order]
    np.testing.assert_allclose(after_wp, expected_after_wp, atol=1e-6)


def test_order_components_by_energy_reorders_weights_and_B():
    rng = np.random.default_rng(2)
    n_cells, n_genes, n_conditions, rank = 20, 15, 5, 3

    A = rng.normal(size=(n_conditions, rank))
    C = rng.normal(size=(n_genes, rank))
    B = rng.normal(size=(rank, rank))
    weights = np.arange(rank, dtype=float)
    projections, _ = np.linalg.qr(rng.normal(size=(n_cells, rank)))

    energy = np.linalg.norm(A, axis=0) * np.linalg.norm(C, axis=0)
    order = np.argsort(energy)[::-1]

    adata = _mock_adata(A.copy(), B.copy(), C.copy(), weights.copy(), projections)
    ordered = order_components_by_energy(adata)

    np.testing.assert_allclose(ordered.uns["Pf2_weights"], weights[order])
    np.testing.assert_allclose(ordered.uns["Pf2_B"], B[:, order])


def test_match_components_across_ranks_identifies_new_component():
    """A rank-(N+1) fit that adds one unrelated component to an otherwise
    identical rank-N fit should report that component as unmatched."""
    rng = np.random.default_rng(3)
    n_genes, rank_low = 40, 4

    C_low = rng.normal(size=(n_genes, rank_low))
    # Canonicalize sign for reproducibility.
    C_low = C_low * canonical_component_signs(C_low)

    new_component = rng.normal(size=(n_genes, 1))
    new_component = new_component * canonical_component_signs(new_component)
    C_high = np.concatenate([C_low, new_component], axis=1)

    matched_pairs, unmatched_high = match_components_across_ranks(C_low, C_high)

    assert matched_pairs.shape[0] == rank_low
    # Every low-rank component should match its identical counterpart.
    assert set(matched_pairs[:, 0].tolist()) == set(range(rank_low))
    assert set(matched_pairs[:, 1].tolist()) == set(range(rank_low))
    assert unmatched_high.tolist() == [rank_low]
