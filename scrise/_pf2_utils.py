"""Shared helper for invoking ``parafac2_nd`` with pass-through kwargs.

Used by both :func:`scrise.factorization.pf2`/``rise_pca_r2x`` and
:func:`scrise.rank_selection.bicv` so that additional ``parafac2_nd`` and
``compress_dataset`` options can be forwarded without every wrapper function
needing to hardcode each one individually (see issue #544).
"""

from typing import Any

import anndata
from parafac2.compress import CompressedData, compress_dataset
from parafac2.parafac2 import parafac2_nd


def run_parafac2(
    X: anndata.AnnData,
    rank: int,
    random_state: int | None,
    tol: float,
    n_iter_max: int,
    normalize_slices: bool = False,
    backend: str | None = None,
    compress: int | tuple[int, int | None] | str | bool | None = None,
    compression_kwarg: dict[str, Any] | None = None,
    parafac2_kwarg: dict[str, Any] | None = None,
) -> tuple[tuple, float]:
    """Fit PARAFAC2, forwarding extra options to the underlying methods.

    ``parafac2_kwarg`` is passed through as additional keyword arguments to
    ``parafac2_nd`` (e.g. ``n_inner``, ``callback``). ``compression_kwarg`` is
    passed through to ``compress_dataset`` (e.g. ``n_power_iter``); it
    requires ``compress`` to be set, since otherwise no compression step
    runs. When ``compression_kwarg`` is given, compression is performed
    directly via ``compress_dataset`` before calling ``parafac2_nd``, rather
    than relying on ``parafac2_nd``'s own ``compress`` shortcut, so that the
    extra keyword arguments can reach it.
    """
    compression_kwarg = dict(compression_kwarg) if compression_kwarg else {}
    parafac2_kwarg = dict(parafac2_kwarg) if parafac2_kwarg else {}

    # parafac2_kwarg may override any of the named options below (e.g. to
    # set normalize_slices/backend without a dedicated top-level argument).
    normalize_slices = parafac2_kwarg.pop("normalize_slices", normalize_slices)
    backend = parafac2_kwarg.pop("backend", backend)

    if compression_kwarg and (compress is None or compress is False):
        raise ValueError("compression_kwarg requires compress to be set.")

    X_in: anndata.AnnData | CompressedData = X
    if compression_kwarg:
        X_in = compress_dataset(
            X,
            L=compress,
            rank=rank,
            random_state=random_state,
            normalize_slices=normalize_slices,
            backend=backend,
            **compression_kwarg,
        )
        compress = None

    return parafac2_nd(
        X_in,
        rank=rank,
        random_state=random_state,
        tol=tol,
        n_iter_max=n_iter_max,
        normalize_slices=normalize_slices,
        backend=backend,
        compress=compress,
        **parafac2_kwarg,
    )
