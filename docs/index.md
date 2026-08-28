# Welcome to RISE's documentation!

## Overview

**RISE** (Reduction and Insight in Single-cell Exploration) is an unsupervised, tensor-based computational method designed for the integrative analysis of single-cell RNA sequencing (scRNA-seq) data across multiple experimental conditions, such as drug treatments, patient cohorts, or time points. Built upon the PARAFAC2 tensor decomposition framework, RISE preserves the inherent three-dimensional structure of multi-condition single-cell data—conditions × cells × genes—instead of flattening it into a conventional two-dimensional matrix. This allows RISE to decompose variation into distinct, interpretable patterns associated with experimental conditions, individual cells, and genes, providing a more nuanced and biologically meaningful analysis.

RISE does not require prior cell-type labels or clustering, reducing bias and enabling discovery of novel cell states, while also separating technical, biological, and condition-driven variation without batch correction that may erase meaningful signals. Its high resolution enables the identification of cell populations and condition-specific subpopulations missed by pseudobulk or clustering-based approaches, and each resulting component is directly linked to specific conditions, genes, and cells, making the results biologically tractable.

## Documentation Contents

- [Tutorial](tutorial.md): Step-by-step walkthrough covering data preprocessing, rank selection, tensor decomposition, compression/export, and factor visualization.
- [API Reference](api.md): Detailed specifications for factorization routines, rank selection, preprocessing, and plotting utilities.
- [References](references.md): Key citations and methodology papers.
