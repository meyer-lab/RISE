# RISE - Reduction and Insight in Single-cell Exploration

RISE (Reduction and Insight in Single-cell Exploration) is an unsupervised, tensor-based computational method designed for the integrative analysis of single-cell RNA sequencing (scRNA-seq) data across multiple experimental conditions, such as drug treatments, patient cohorts, or time points. Built upon the PARAFAC2 tensor decomposition framework, RISE preserves the inherent three-dimensional structure of multi-condition single-cell data—conditions × cells × genes—instead of flattening it into a conventional two-dimensional matrix. This allows RISE to decompose variation into distinct, interpretable patterns associated with experimental conditions, individual cells, and genes, providing a more nuanced and biologically meaningful analysis.

RISE does not require prior cell-type labels or clustering, reducing bias and enabling discovery of novel cell states, while also separating technical, biological, and condition-driven variation without batch correction that may erase meaningful signals. Its high resolution enables the identification of cell populations and condition-specific subpopulations missed by pseudobulk or clustering-based approaches, and each resulting component is directly linked to specific conditions, genes, and cells, making the results biologically tractable.

- **Read the documentation** at [RISE Documentation](https://meyer-lab.github.io/RISE/).
- RISE uses the [AnnData](https://anndata.readthedocs.io/) format for handling single-cell data matrices.

## Installation

> **Note:** The `RISE` package was renamed to `scrise` on PyPI (the import name changed from `RISE` to `scrise`). The GitHub repository name is unchanged. If you have `RISE` pinned in a `requirements.txt` or install script, update it to `scrise` as shown below.

To add `scrise` to your Python environment, you can install it directly from GitHub:

```bash
pip install git+https://github.com/meyer-lab/RISE.git@main
```

For GPU acceleration support (propagated to `parafac2[gpu]`):

```bash
pip install "scrise[gpu] @ git+https://github.com/meyer-lab/RISE.git@main"
```

Or add the following line to your `requirements.txt`:

```
git+https://github.com/meyer-lab/RISE.git@main
```

or with GPU support:

```
scrise[gpu] @ git+https://github.com/meyer-lab/RISE.git@main
```


## Quick Start

RISE works with preprocessed AnnData objects containing single-cell RNA-seq data:

```python
from scrise.factorization import pf2

# Perform PARAFAC2 tensor decomposition
X = pf2(X=adata, rank=20, doEmbedding=True, random_state=42)

# Results are stored in the AnnData object:
# - X.uns["Pf2_weights"]: Component weights
# - X.uns["Pf2_A"]: Condition factors
# - X.uns["Pf2_B"]: Eigen-state factors
# - X.varm["Pf2_C"]: Gene factors
# - X.obsm["projections"]: Cell projections
# - X.obsm["weighted_projections"]: Weighted cell projections
```

See the [tutorial](https://meyer-lab.github.io/RISE/tutorial.html) for a complete workflow including preprocessing, rank selection, visualization, and interpretation.

## Key Features

- **Tensor-based decomposition**: Preserves the 3D structure of multi-condition scRNA-seq data
- **Unsupervised analysis**: No prior cell-type labels or clustering required
- **High resolution**: Identifies cell populations and condition-specific subpopulations
- **Interpretable results**: Components directly linked to conditions, cells, and genes
- **Integrated workflow**: Built-in preprocessing, visualization, and interpretation tools
- **Principled rank selection**: Bi-cross-validation (`scrise.rank_selection`) for choosing the number of components, either by evaluating a set of candidate ranks or automatically via Bayesian optimization

## Citation

If you use RISE in your work, please cite the RISE publication as follows:

**Integrative, high-resolution analysis of single-cell gene expression across experimental conditions with PARAFAC2-RISE**

Andrew Ramirez, [...], Aaron Meyer

*Cell Systems*, 2025. DOI: [10.1016/j.cels.2025.101294](https://doi.org/10.1016/j.cels.2025.101294)
