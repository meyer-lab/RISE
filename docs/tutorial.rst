Tutorial for PARAFAC2-RISE on scRNA-seq Data
=============================================

This tutorial demonstrates the complete RISE workflow for analyzing single-cell RNA-seq data across experimental conditions.

Installation
------------

To add RISE to your Python package, add the following line to your ``requirements.txt`` and remake your virtual environment::

    git+https://github.com/meyer-lab/RISE.git@main

For GPU acceleration support (which propagates to ``parafac2[gpu]``)::

    RISE[gpu] @ git+https://github.com/meyer-lab/RISE.git@main


Preprocessing the Dataset
--------------------------

**Input Requirements**

Your AnnData object must meet the following requirements:

1. **Condition Index**: Include an observations column ``condition_unique_idxs`` that is a 0-indexed array indicating which condition each cell is derived from, along with the cell barcode. Condition 1 cells are indexed as 0, Condition 2 as 1, and so on.

2. **Preprocessing**: Your AnnData object must be preprocessed (doublets removed, genes filtered, normalized, and log-transformed) before running the algorithm. The ``prepare_dataset`` function can assist with preprocessing for gene filtering, normalization, and assigning ``condition_unique_idxs``.

**Using prepare_dataset**

The ``prepare_dataset`` function assists with preprocessing your data. Parameters:

- ``X``: AnnData object containing raw count data in sparse matrix format
- ``condition_name``: Name of the column in ``X.obs`` that specifies experimental conditions for each cell
- ``geneThreshold``: Minimum mean expression threshold for gene filtering (genes with mean expression below this value are removed)
- ``deviance``: If True, applies deviance transformation instead of log normalization (default: False)

The function performs the following steps:

- Filters cells with fewer than 10 total counts
- Filters genes based on the ``geneThreshold`` parameter
- Normalizes total counts per cell to the median
- Scales gene expression values
- Applies log₁₀((1000 × normalized_value) + 1) transformation (or deviance transformation if specified)
- Creates ``condition_unique_idxs`` column in ``X.obs`` with 0-indexed condition assignments
- Pre-calculates gene means and stores in ``X.var["means"]``

**Import and Prepare the Dataset**

Import your dataset as an AnnData object with preprocessed data.

Choosing the Rank
-----------------

**Assess Variance Explained by RISE and PCA**

Determine the optimal component/rank by plotting the variance explained (R²X) across different ranks for both RISE and PCA. This helps balance model complexity with explanatory power.

.. code-block:: python

    from RISE.plotting import plot_r2x
    import matplotlib.pyplot as plt

    ranks = [1, 5, 10, 15, 20, 25, 30]
    fig, ax = plt.subplots(figsize=(5, 5))

    plot_r2x(X, ranks, ax)
    plt.tight_layout()
    plt.show()

.. figure:: _static/tutorial_images/step2_r2x.png
   :align: center
   :width: 500px
   
**Evaluate Factor Stability with Factor Match Score (FMS)**

Measure the reproducibility of the RISE factorization across different ranks. An FMS above ~0.6 indicates stable components.

.. code-block:: python

    from RISE.plotting import plot_fms_diff_ranks

    fig, ax = plt.subplots(figsize=(5, 5))
    rank_list = list(ranks)

    plot_fms_diff_ranks(X, ax, ranksList=rank_list, runs=3)
    plt.tight_layout()
    plt.show()

.. figure:: _static/tutorial_images/step3_fms.png
   :align: center
   :width: 500px

**Select a Rank with Bi-Cross-Validation (BiCV)**

R2X (in-sample fit) always increases with rank, so it cannot by itself
identify an optimal rank — it can only reveal an elbow. Bi-cross-validation
(BiCV) addresses this by holding out a random subset of cells *and* genes,
fitting RISE on the remaining data, and scoring how well the fit predicts
the held-out block. Because it is evaluated on unseen data, BiCV R2X
penalizes overfitting and typically peaks (or plateaus) near the rank that
generalizes best, unlike the fit R2X.

Two functions are available in ``RISE.rank_selection``:

- ``bicv``: exhaustively evaluates a list of candidate ranks, each with
  several repeated random train/test splits, and returns both the fit R2X
  and the BiCV R2X for every rank so you can inspect the full curve.
- ``optimize_rank``: for a large search range, sequentially evaluates a
  small number of ranks, exploiting the fact that BiCV R2X vs. rank is
  expected to be an approximately quadratic, concave curve: each
  subsequent candidate rank is chosen near the vertex of a least-squares
  quadratic fit through the ranks evaluated so far, and the search stops
  early once that vertex estimate stabilizes.

.. code-block:: python

    from RISE.rank_selection import bicv
    from RISE.plotting import plot_bicv_r2x

    ranks = [5, 10, 15, 20, 25, 30]
    results = bicv(X, ranks, n_repeats=3, random_state=0)

    fig, ax = plt.subplots(figsize=(5, 5))
    plot_bicv_r2x(results, ax)
    plt.tight_layout()
    plt.show()

``bicv`` returns a long-form DataFrame (columns ``Rank``, ``Repeat``,
``Metric``, ``R2X``) suitable for further analysis as well as plotting.
Each BiCV trial holds out ``held_out_cell_frac`` of the cells within each
condition and ``held_out_gene_frac`` of the genes (both default to 0.2);
increase ``n_repeats`` for a smoother, less noisy BiCV curve at the cost of
more compute.

To search a wide rank range without evaluating every candidate rank, use
``optimize_rank`` instead:

.. code-block:: python

    from RISE.rank_selection import optimize_rank
    from RISE.plotting import plot_rank_optimization

    result = optimize_rank(X, rank_bounds=(5, 50), n_repeats=3, random_state=0)
    print(f"Best rank: {result.best_rank} (BiCV R2X = {result.best_r2x:.3f})")

    fig, ax = plt.subplots(figsize=(5, 5))
    plot_rank_optimization(result, ax)
    plt.tight_layout()
    plt.show()

``result.history`` contains every rank actually evaluated (with mean and
standard deviation of BiCV R2X across repeats) — typically well under
``n_calls``, since the search stops once it converges — and
``plot_rank_optimization`` visualizes the quadratic fit across the search
range alongside the best rank found. If ``result.quadratic_r2`` is low
(well under 0.5), the curve was not well described by a quadratic over the
searched range; treat ``result.best_rank`` as the answer in that case
rather than the fit's vertex, and consider running ``bicv`` over the full
range to inspect the curve directly.

Running the Factorization
--------------------------

**Perform RISE Factorization**

Based on the variance explained and FMS, select a rank and perform the RISE factorization. This decomposes the data into condition, eigen-state, and gene factors.

.. code-block:: python

    from RISE.factorization import pf2

    rank = 20
    X = pf2(X=X, rank=rank, doEmbedding=True, tolerance=1e-6, 
            max_iter=100, random_state=42)

The function ``pf2`` performs PARAFAC2 tensor decomposition on the AnnData object. Parameters:

- ``X``: AnnData object with preprocessed scRNA-seq data
- ``rank``: Number of components to extract
- ``random_state``: Random seed for reproducibility (default: 1)
- ``doEmbedding``: If True, automatically computes PaCMAP embeddings of cell projections and stores in ``X.obsm["embedding"]`` (default: True)
- ``tolerance``: Convergence threshold for the optimization algorithm (default: 1e-6)
- ``max_iter``: Maximum number of iterations (default: 100)

The output of ``pf2`` includes the original AnnData object with added results and the reconstruction error (R²X). The following are added to the AnnData object:

- **Weights**: ``X.uns["Pf2_weights"]`` - The weights for each component
- **Condition Factors**: ``X.uns["Pf2_A"]`` - Factors with respect to conditions
- **Eigen-state Factors**: ``X.uns["Pf2_B"]`` - Eigen-state factors
- **Gene Factors**: ``X.varm["Pf2_C"]`` - Gene factors (matrix width equals the rank)
- **Projections**: ``X.obsm["projections"]`` - Cell projections for each component (matrix width equals the rank)
- **Weighted Projections**: ``X.obsm["weighted_projections"]`` - Weighted projections for each cell across all components, determining how each cell relates to each component pattern
- **PaCMAP Embedding**: ``X.obsm["X_pf2_PaCMAP"]`` - 2D visualization embedding of cell projections (if ``doEmbedding=True``)

Saving and Loading Factorization Results
-----------------------------------------

Single-cell datasets often contain tens of thousands of cells and genes, resulting in multi-gigabyte files when saving raw expression matrices alongside decomposition results. To enable lightweight sharing and fast downstream analysis, RISE includes built-in functions for exporting and loading factorizations:

* **Optimized Product Quantization (OPQ)**: Compresses the continuous per-cell projection matrix (``X.obsm["projections"]``) to compact 8-bit quantization codes while guaranteeing high reconstruction fidelity (:math:`R^2 \ge 0.99`).
* **Deterministic Reconstruction**: Factor matrices (``Pf2_A``, ``Pf2_B``, ``Pf2_weights``, ``Pf2_C``) are stored in single precision (``float32``), and weighted projections are deterministically reconstructed on load via :math:`\text{projections} \times \text{Pf2\_B}`, eliminating redundant data on disk.
* **Barcode Index Compression**: String cell barcodes are compressed into chunked gzip byte matrices to minimize HDF5 overhead.

**1. Exporting Factors to Disk**

Use ``export_factors`` to save decomposition factors and cell projections to an ``.h5ad`` file:

.. code-block:: python

    from RISE import export_factors

    # Export factors and compressed projections
    export_factors(
        X,
        filename="thomson_factors.h5ad",
        fidelity_threshold=0.99,  # Target R^2 reconstruction fidelity (default: 0.99)
        random_state=42,
    )

This produces a compact file (~4-5 MB for ~30,000 cells) containing all factor matrices, cell metadata, and PaCMAP coordinates, without bundling the uncompressed raw count matrix.

**2. Loading Factors for Fast Downstream Analysis**

Use ``load_factors`` to load the saved decomposition. Projections are decoded from OPQ codebooks and weighted projections are automatically recalculated:

.. code-block:: python

    from RISE import load_factors
    from RISE.plotting import plot_condition_factors, plot_labels_pacmap

    # Load factor results into an AnnData object
    factors = load_factors("thomson_factors.h5ad")

    # Access reconstructed factor matrices and projections
    print("Condition factors shape:", factors.uns["Pf2_A"].shape)
    print("Gene factors shape:", factors.varm["Pf2_C"].shape)
    print("Cell projections shape:", factors.obsm["projections"].shape)
    print("Weighted projections shape:", factors.obsm["weighted_projections"].shape)
    print("OPQ reconstruction fidelity R^2:", factors.uns["opq_fidelity"])

    # Generate factor and embedding plots directly
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_condition_factors(factors, ax=ax, cond="Condition", log_transform=True)
    plt.show()

**3. On-Demand Attachment of Raw Expression Data**

When you need to visualize individual gene expression on top of embeddings (e.g., using ``plot_gene_pacmap``) or perform differential expression, supply ``raw_path`` to ``load_factors``:

.. code-block:: python

    # Load factors and automatically match & attach raw expression matrix
    X_full = load_factors(
        "thomson_factors.h5ad",
        raw_path="path/to/raw_dataset.h5ad",  # Supports AnnData and IVCSR / VCSC files
    )

    # Cell barcodes and gene IDs are matched automatically
    print("Expression matrix shape:", X_full.X.shape)

    # Now gene expression plotting functions work seamlessly
    from RISE.plotting import plot_gene_pacmap

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_gene_pacmap("MS4A1", X_full, ax=ax)
    plt.show()

Visualizing the Factors
------------------------

**Visualize Condition Factors**

Examine how each experimental condition contributes to the identified patterns. Log-transforming these factors allows for easier interpretation of condition-specific effects.
Several plotting functions are available in the ``RISE.plotting`` package.

.. code-block:: python

    from RISE.plotting import plot_condition_factors

    fig, ax = plt.subplots(figsize=(8, 8))

    plot_condition_factors(X, ax=ax, cond="Condition", log_transform=True)
    plt.tight_layout()
    plt.show()

.. figure:: _static/tutorial_images/step5_condition_factors.png
   :align: center
   :width: 700px
   
   **Condition Factors Heatmap.** This heatmap shows how each experimental condition (rows) contributes to each component (columns). 

**Visualize Cell Embedding**

Explore the latent space of cells using nonlinear dimensionality reduction methods such as PaCMAP. Label cells by cell type or experimental condition to understand clustering patterns.

.. code-block:: python

    from RISE.plotting import plot_labels_pacmap

    fig, ax = plt.subplots(figsize=(8, 8))

    plot_labels_pacmap(X, labelType="Cell Type", ax=ax)
    plt.tight_layout()
    plt.show()

.. figure:: _static/tutorial_images/step6_cell_embedding.png
   :align: center
   :width: 700px

**Visualize Eigen-State Factors**

Analyze how each cell state contributes to the identified patterns. Each eigen-state represents a summary of similar cells with a distinct expression profile.

.. code-block:: python

    from RISE.plotting import plot_eigenstate_factors

    fig, ax = plt.subplots(figsize=(4, 4))

    plot_eigenstate_factors(X, ax=ax)
    plt.ylabel("Eigen-state")
    plt.tight_layout()
    plt.show()

.. figure:: _static/tutorial_images/step7_eigenstate_factors.png
   :align: center
   :width: 400px
   
   **Eigen-state Factors Heatmap.** This heatmap shows how each eigen-state (representing groups of similar cells) is related to each component. High values indicate strong association with a component.

**Visualize Gene Factors**

Identify which genes are highly weighted in each component, revealing coordinated gene modules. Adjust weight values to focus on genes that contribute significantly to the patterns.

.. code-block:: python

    from RISE.plotting import plot_gene_factors

    fig, ax = plt.subplots(figsize=(7, 8))

    plot_gene_factors(X, ax=ax, weight=0.2, trim=True)
    plt.tight_layout()
    plt.show()

.. figure:: _static/tutorial_images/step8_gene_factors.png
   :align: center
   :width: 650px
   
   **Gene Factors Heatmap.** This heatmap shows which genes (rows) are associated with each component (columns). The weight parameter filters out genes with low contributions for easier interpretation.

Interpreting a Component
-------------------------

**Investigate Gene Associations for a Component**

Overlay specific gene expression on the cell embedding to see which cells express genes of interest for a particular component.

.. code-block:: python

    from RISE.plotting import plot_gene_pacmap

    fig, ax = plt.subplots(figsize=(8, 8))

    gene = "MS4A1"  # B cell marker
    plot_gene_pacmap(gene, X, ax=ax, clip_outliers=0.9995)
    plt.tight_layout()
    plt.show()

.. figure:: _static/tutorial_images/step9_gene_expression.png
   :align: center
   :width: 700px
   
   **Gene Expression on PaCMAP.** This visualization overlays MS4A1 gene expression (a B cell marker) onto the PaCMAP embedding. Cells are colored by expression level, revealing which populations express the gene.

**Investigate Cell Associations for a Component**

Visualize how cells contribute to specific components using weighted projections, revealing subpopulations with distinct expression patterns.

.. code-block:: python

    from RISE.plotting import plot_wp_pacmap

    fig, ax = plt.subplots(figsize=(8, 8))

    plot_wp_pacmap(X, cmp=10, ax=ax, cbarMax=0.9)
    plt.tight_layout()
    plt.show()

.. figure:: _static/tutorial_images/step10_weighted_projections.png
   :align: center
   :width: 700px
   
   **Weighted Projections for Component 10.** This plot shows which cells contribute most strongly to component 10. Cells with high weighted projections (bright colors) are most representative of that component's pattern.
