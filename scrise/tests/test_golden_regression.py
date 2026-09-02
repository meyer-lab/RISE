"""
Golden regression test for the end-to-end pf2() fit.

The invariant/contract tests elsewhere check *properties* that must hold
for any input; this test instead pins the actual numbers produced by a
fixed synthetic dataset + fixed random_state, so a silent change in
behavior -- e.g. from a `parafac2` dependency bump changing its solver's
convergence path, or an accidental change to `pf2`'s defaults -- shows up
as a diff here even if it doesn't happen to violate any general property.

The expected values below were captured from the current `pf2()` output on
the fixture defined in this file; if they need to be regenerated (a
deliberate algorithmic change, a new `parafac2` release that legitimately
changes results), rerun this file's `if __name__ == "__main__"` block and
paste the printed values back in.
"""

import numpy as np
import pandas as pd
import pytest
from parafac2.parafac2 import parafac2_nd

from .. import pf2
from .conftest import make_synthetic_pf2_data


def _fixture():
    X = make_synthetic_pf2_data(
        n_cond=4, n_genes=25, rank=3, seed=7, cells_per_cond=(20, 30)
    )
    X.obs["condition_unique_idxs"] = pd.Categorical(
        X.obs["condition_unique_idxs"].astype(int)
    )
    return X


def test_pf2_golden_values_on_fixed_synthetic_fixture():
    X = _fixture()

    # Reconstruction quality (R2X) is deterministic for a fixed seed/data and
    # is the single most sensitive summary of "did the fit change at all".
    # parafac2_nd returns it directly, so use that rather than recomputing
    # by hand (pf2() reorders/signs the factors before returning them, so a
    # manual reconstruction from its output must exactly replicate that
    # bookkeeping to agree -- fragile for what this test wants to check).
    _, r2x = parafac2_nd(
        X, rank=3, random_state=1, tol=1e-6, n_iter_max=100, compress=None
    )
    assert r2x == pytest.approx(0.9814689334613979, abs=1e-3)

    result = pf2(
        X, rank=3, random_state=1, doEmbedding=False, compress=None, max_iter=100
    )
    C = np.array(result.varm["Pf2_C"])
    weights = np.array(result.uns["Pf2_weights"])

    # Component weights, order, and which gene dominates each component are
    # a much coarser (and thus more robust-to-numerical-noise) fingerprint
    # of the fit than raw factor values.
    np.testing.assert_allclose(weights, [22.2080, 56.6194, 24.3079], rtol=1e-2)

    top_gene_per_component = np.argmax(np.abs(C), axis=0)
    np.testing.assert_array_equal(top_gene_per_component, [12, 13, 5])


if __name__ == "__main__":
    X = _fixture()
    result = pf2(
        X, rank=3, random_state=1, doEmbedding=False, compress=None, max_iter=100
    )
    A = np.array(result.uns["Pf2_A"])
    C = np.array(result.varm["Pf2_C"])
    weights = np.array(result.uns["Pf2_weights"])
    print("weights:", weights.tolist())
    print("top_gene_per_component:", np.argmax(np.abs(C), axis=0).tolist())
