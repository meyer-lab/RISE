import numpy.typing as npt
import scanpy


def gateThomsonCells(X) -> npt.ArrayLike:
    """Manually gates cell types for Thomson UMAP"""
    X = scanpy.pp.neighbors(X, n_neighbors=15, copy=True, use_rep="embedding")
    scanpy.tl.leiden(X, resolution=0.1, random_state=1)

    return X.obs["leiden"]
