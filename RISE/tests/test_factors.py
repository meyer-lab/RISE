"""
Test OPQ compression, export_factors, and load_factors.
"""

import tempfile

import anndata
import numpy as np
import pytest

from RISE import OPQQuantizer, export_factors, find_optimal_opq, load_factors


def test_opq_quantizer_synthetic():
    """Test OPQQuantizer encode/decode and reconstruction fidelity."""
    np.random.seed(42)
    N, D = 1000, 20
    P = np.random.randn(N, D).astype(np.float32)
    P, _ = np.linalg.qr(P)

    quantizer = OPQQuantizer(M=10, n_bits=8, random_state=42)
    codes, recon, r2 = quantizer.fit_transform(P)

    assert codes.shape == (N, 10)
    assert codes.dtype == np.uint8
    assert recon.shape == (N, D)
    assert recon.dtype == np.float32
    assert r2 > 0.98

    # Test from_saved
    loaded_quantizer = OPQQuantizer.from_saved(
        R=quantizer.R,
        centroids_cat=quantizer.centroids_cat,
        sub_dims=quantizer.sub_dims,
    )
    recon_loaded = loaded_quantizer.decode(codes)
    np.testing.assert_allclose(recon, recon_loaded, atol=1e-6)


def test_find_optimal_opq_fidelity_threshold():
    """Test that find_optimal_opq selects an M meeting the fidelity threshold."""
    np.random.seed(42)
    N, D = 2000, 20
    P = np.random.randn(N, D).astype(np.float32)
    P, _ = np.linalg.qr(P)

    _, codes, r2 = find_optimal_opq(P, fidelity_threshold=0.95, random_state=42)
    assert r2 >= 0.95
    assert codes.shape[0] == N


def test_export_and_load_factors_roundtrip():
    """Test export_factors and load_factors round-trip functionality."""
    np.random.seed(42)
    n_cells = 500
    n_genes = 200
    n_conditions = 10
    rank = 15

    P = np.random.randn(n_cells, rank).astype(np.float32)
    P, _ = np.linalg.qr(P)
    A = np.random.randn(n_conditions, rank).astype(np.float32)
    B = np.random.randn(rank, rank).astype(np.float32)
    weights = np.random.rand(rank).astype(np.float32)
    C = np.random.randn(n_genes, rank).astype(np.float32)
    embedding = np.random.randn(n_cells, 2).astype(np.float32)

    obs = {"Condition": [f"cond_{i % n_conditions}" for i in range(n_cells)]}
    var = {"gene_name": [f"gene_{j}" for j in range(n_genes)]}

    mock_adata = anndata.AnnData(
        obs=obs,
        var=var,
        uns={"Pf2_A": A, "Pf2_B": B, "Pf2_weights": weights},
        varm={"Pf2_C": C},
        obsm={"projections": P, "X_pf2_PaCMAP": embedding},
    )

    with tempfile.NamedTemporaryFile(suffix=".h5ad") as tmp:
        # Export with OPQ
        export_factors(mock_adata, tmp.name, fidelity_threshold=0.99, random_state=42)

        # Inspect on-disk AnnData directly (verify weighted_projections and uncompressed projections are omitted)
        on_disk = anndata.read_h5ad(tmp.name)
        assert "weighted_projections" not in on_disk.obsm
        assert "projections_opq_codes" in on_disk.obsm
        assert on_disk.X is None
        assert on_disk.uns["Pf2_A"].dtype == np.float32
        assert on_disk.uns["Pf2_B"].dtype == np.float32
        assert on_disk.uns["Pf2_weights"].dtype == np.float32
        assert on_disk.varm["Pf2_C"].dtype == np.float32
        assert "embedding" in on_disk.obsm

        # Load back with load_factors
        loaded = load_factors(tmp.name)
        assert loaded.obsm["projections"].shape == (n_cells, rank)
        assert loaded.obsm["projections"].dtype == np.float32
        assert loaded.obsm["weighted_projections"].shape == (n_cells, rank)
        assert loaded.obsm["weighted_projections"].dtype == np.float32
        assert loaded.obsm["X_pf2_PaCMAP"].shape == (n_cells, 2)
        np.testing.assert_allclose(
            loaded.obsm["weighted_projections"],
            loaded.obsm["projections"] @ loaded.uns["Pf2_B"],
            atol=1e-5,
        )


def test_export_factors_missing_keys():
    """Test that export_factors raises KeyError when required factors are missing."""
    adata = anndata.AnnData(np.zeros((10, 10)))
    with tempfile.NamedTemporaryFile(suffix=".h5ad") as tmp, pytest.raises(KeyError):
        export_factors(adata, tmp.name)


def test_load_thomson_cached_factors():
    """Test loading the converted Thomson OPQ factors."""
    factors = load_factors("analysis/data/Thomson_cached_factors.h5ad")
    assert factors.shape == (29433, 12164)
    assert factors.X is None
    assert "projections" in factors.obsm
    assert "weighted_projections" in factors.obsm
    assert "X_pf2_PaCMAP" in factors.obsm
    assert factors.obsm["projections"].shape == (29433, 20)
    assert factors.obsm["weighted_projections"].shape == (29433, 20)
    assert factors.uns["opq_fidelity"] >= 0.99
