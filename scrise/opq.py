"""
Optimized Product Quantization (OPQ) for projecting and compressing factor matrices.
"""

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import r2_score


class OPQQuantizer:
    """Optimized Product Quantizer for compressing projection matrices.

    Parameters
    ----------
    M : int
        Number of sub-quantizers (sub-vector partitions).
    n_bits : int, optional (default: 8)
        Number of bits per sub-quantizer. 8 bits corresponds to 256 centroids per sub-space.
    n_iter : int, optional (default: 5)
        Number of alternating OPQ optimization iterations for the rotation matrix.
    random_state : int, optional (default: 42)
        Random seed for reproducibility.
    """

    def __init__(
        self,
        M: int,
        n_bits: int = 8,
        n_iter: int = 5,
        random_state: int = 42,
    ):
        self.M = M
        self.n_bits = n_bits
        self.n_iter = n_iter
        self.random_state = random_state
        self.R: np.ndarray | None = None
        self.centroids_cat: np.ndarray | None = None
        self.sub_dims: np.ndarray | None = None

    def fit(self, X: np.ndarray, sample_size: int = 50000) -> "OPQQuantizer":
        """Fit rotation matrix and sub-quantizer codebooks on data matrix X."""
        X_arr = np.asarray(X, dtype=np.float32)
        N, D = X_arr.shape
        M = min(self.M, D)
        self.M = M

        # Partition dimensions across M sub-vectors
        sub_dims = [D // M + (1 if i < D % M else 0) for i in range(M)]
        self.sub_dims = np.array(sub_dims, dtype=np.int32)
        dim_offsets = np.cumsum([0] + sub_dims)
        K_centroids = min(2**self.n_bits, N)

        # Subsample for training if N > sample_size
        if N > sample_size:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(N, size=sample_size, replace=False)
            train_X = X_arr[idx]
        else:
            train_X = X_arr

        # Initialize rotation matrix
        R = np.eye(D, dtype=np.float32)

        # Alternating OPQ iterations
        for _ in range(self.n_iter):
            X_rot = train_X @ R
            X_hat_rot = np.zeros_like(X_rot)
            for m in range(M):
                ds, de = dim_offsets[m], dim_offsets[m + 1]
                sub = X_rot[:, ds:de]
                if sub.shape[1] == 1:
                    km = KMeans(
                        n_clusters=K_centroids,
                        random_state=self.random_state,
                        n_init=1,
                        max_iter=20,
                    ).fit(sub)
                else:
                    km = MiniBatchKMeans(
                        n_clusters=K_centroids,
                        random_state=self.random_state,
                        batch_size=min(2048, len(train_X)),
                        n_init=1,
                        max_iter=20,
                    ).fit(sub)
                X_hat_rot[:, ds:de] = km.cluster_centers_[km.labels_]

            # Update orthogonal rotation matrix via Procrustes SVD
            U, _, Vt = np.linalg.svd(train_X.T @ X_hat_rot)
            R = (U @ Vt).astype(np.float32)

        # Final codebook fitting on rotated training data
        X_rot_train = train_X @ R
        centroids_list = []
        for m in range(M):
            ds, de = dim_offsets[m], dim_offsets[m + 1]
            sub = X_rot_train[:, ds:de]
            if sub.shape[1] == 1:
                km = KMeans(
                    n_clusters=K_centroids,
                    random_state=self.random_state,
                    n_init=2,
                    max_iter=50,
                ).fit(sub)
            else:
                km = MiniBatchKMeans(
                    n_clusters=K_centroids,
                    random_state=self.random_state,
                    batch_size=min(4096, len(train_X)),
                    n_init=3,
                    max_iter=50,
                ).fit(sub)
            centroids_list.append(km.cluster_centers_.astype(np.float32))

        self.R = R
        self.centroids_cat = np.concatenate(centroids_list, axis=1).astype(np.float32)
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode data matrix X into uint8 sub-quantizer centroid indices."""
        if self.R is None or self.centroids_cat is None or self.sub_dims is None:
            raise ValueError("OPQQuantizer has not been fitted yet.")

        X_arr = np.asarray(X, dtype=np.float32)
        N, _ = X_arr.shape
        M = self.M
        dim_offsets = np.cumsum([0] + list(self.sub_dims))
        X_rot = X_arr @ self.R
        codes = np.zeros((N, M), dtype=np.uint8)

        for m in range(M):
            ds, de = dim_offsets[m], dim_offsets[m + 1]
            sub = X_rot[:, ds:de]
            c = self.centroids_cat[:, ds:de]
            # Vectorized nearest centroid computation: dist^2 = ||sub||^2 - 2 sub @ c.T + ||c||^2
            sub_sq = np.sum(sub**2, axis=1, keepdims=True)
            c_sq = np.sum(c**2, axis=1, keepdims=True).T
            dists = sub_sq - 2.0 * (sub @ c.T) + c_sq
            codes[:, m] = np.argmin(dists, axis=1).astype(np.uint8)

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode uint8 sub-quantizer codes back to the continuous feature space."""
        if self.R is None or self.centroids_cat is None or self.sub_dims is None:
            raise ValueError("OPQQuantizer has not been fitted yet.")

        codes_arr = np.asarray(codes, dtype=np.uint8)
        N, M = codes_arr.shape
        D = self.R.shape[0]
        dim_offsets = np.cumsum([0] + list(self.sub_dims))
        X_recon_rot = np.zeros((N, D), dtype=np.float32)

        for m in range(M):
            ds, de = dim_offsets[m], dim_offsets[m + 1]
            c = self.centroids_cat[:, ds:de]
            X_recon_rot[:, ds:de] = c[codes_arr[:, m]]

        return (X_recon_rot @ self.R.T).astype(np.float32)

    def fit_transform(
        self, X: np.ndarray, sample_size: int = 50000
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Fit OPQ, encode X to codes, reconstruct, and compute R^2."""
        self.fit(X, sample_size=sample_size)
        codes = self.encode(X)
        recon = self.decode(codes)
        r2 = float(r2_score(X, recon))
        return codes, recon, r2

    @classmethod
    def from_saved(
        cls,
        R: np.ndarray,
        centroids_cat: np.ndarray,
        sub_dims: np.ndarray,
        n_bits: int = 8,
    ) -> "OPQQuantizer":
        """Instantiate a fitted OPQQuantizer from saved parameters."""
        quantizer = cls(M=len(sub_dims), n_bits=n_bits)
        quantizer.R = np.asarray(R, dtype=np.float32)
        quantizer.centroids_cat = np.asarray(centroids_cat, dtype=np.float32)
        quantizer.sub_dims = np.asarray(sub_dims, dtype=np.int32)
        return quantizer


def find_optimal_opq(
    P: np.ndarray,
    fidelity_threshold: float = 0.99,
    random_state: int = 42,
) -> tuple[OPQQuantizer, np.ndarray, float]:
    """Find the smallest number of sub-quantizers M achieving R^2 >= fidelity_threshold.

    Parameters
    ----------
    P : np.ndarray
        Projection matrix of shape (N, D).
    fidelity_threshold : float, optional (default: 0.99)
        Target R^2 reconstruction accuracy.
    random_state : int, optional (default: 42)
        Random seed.

    Returns
    -------
    tuple of (OPQQuantizer, np.ndarray, float)
        (quantizer, codes, r2)
    """
    P_arr = np.asarray(P, dtype=np.float32)
    _, D = P_arr.shape

    # Candidate M values to evaluate (increasing order for compression)
    candidate_Ms = sorted({1, 2, 4, 5, 8, 10, 12, 15, 20, 25, 30, D})
    candidate_Ms = [m for m in candidate_Ms if m <= D]
    if D not in candidate_Ms:
        candidate_Ms.append(D)

    best_quantizer = None
    best_codes = None
    best_r2 = -1.0

    for M in candidate_Ms:
        quantizer = OPQQuantizer(M=M, random_state=random_state)
        codes, _, r2 = quantizer.fit_transform(P_arr)

        if r2 > best_r2:
            best_quantizer = quantizer
            best_codes = codes
            best_r2 = r2

        if r2 >= fidelity_threshold:
            return quantizer, codes, r2

    # Fallback to highest fidelity achieved: the loop always runs at least
    # once (candidate_Ms is never empty), so both are guaranteed to be set.
    assert best_quantizer is not None
    assert best_codes is not None
    return best_quantizer, best_codes, best_r2
