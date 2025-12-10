Tutorial for PARAFAC2-RISE on scRNA-seq Data
=============================================

Overview
--------

Effective exploration and analysis tools are vital for the extraction of insights from single-cell data. However, current techniques for modeling single-cell studies performed across experimental conditions (e.g., samples) require restrictive assumptions or do not adequately deconvolute condition-to-condition variation from cell-to-cell variation. 

**RISE** (Reduction and Insight in Single-cell Exploration) is an adaptation of the tensor decomposition method PARAFAC2 that enables the dimensionality reduction and analysis of single-cell data across conditions. RISE enables associations of gene variation patterns with patients or perturbations while connecting each coordinated change to single cells without requiring cell-type annotations.

We demonstrate the benefits of RISE across distinct examples of single-cell RNA-sequencing experiments of peripheral immune cells: pharmacologic drug perturbations and systemic lupus erythematosus patient samples. The theoretical grounding of RISE suggests a unified framework for many single-cell data modeling tasks while providing an intuitive dimensionality reduction approach for multi-sample single-cell studies across biological contexts.

Installation
------------

To add RISE to your Python package, add the following line to your ``requirements.txt`` and remake your virtual environment::

    git+https://github.com/meyer-lab/parafac2.git@main

Input Requirements
------------------

Your AnnData object must meet the following requirements:

1. **Condition Index**: Include an observations column ``condition_unique_idxs`` that is a 0-indexed array indicating which condition each cell is derived from, along with the cell barcode. Condition 1 cells are indexed as 0, Condition 2 as 1, and so on.

2. **Preprocessing**: Your AnnData object must be preprocessed (doublets removed, genes filtered, normalized, and log-transformed) before running the algorithm. The ``prepare_dataset`` function from ``parafac2.normalize`` can assist with preprocessing including assigning ``condition_unique_idx`` and gene filtering, and ``doubletdetection`` can help remove doublets.

The PARAFAC2 Algorithm
----------------------

The function ``parafac2_nd`` is the PARAFAC2 algorithm with various parameters that can be altered, such as:

- ``rank``: Number of components
- ``tolerance``: Convergence threshold
- ``max_iter``: Maximum number of iterations
- ``random_state``: Random seed for reproducibility

Outputs
-------

The output of ``parafac2_nd`` includes the original AnnData object with added results and the reconstruction error (R²X). The following are added to the AnnData object:

- **Weights**: ``X.uns["Pf2_weights"]`` - The weights for each component
- **Condition Factors**: ``X.uns["Pf2_A"]`` - Factors with respect to conditions
- **Eigen-state Factors**: ``X.uns["Pf2_B"]`` - Eigen-state factors
- **Gene Factors**: ``X.varm["Pf2_C"]`` - Gene factors (matrix width equals the rank)
- **Projections**: ``X.obsm["projections"]`` - Cell projections for each component (matrix width equals the rank)
- **Weighted Projections**: ``X.obsm["weighted_projections"]`` - Weighted projections for each cell across all components, determining how each cell relates to each component pattern

We recommend implementing an embedding algorithm such as PaCMAP or UMAP on ``X.obsm["projections"]`` to visualize cell-to-cell heterogeneity, creating a new column such as ``X.obsm["embedding"]``.

Tutorial Workflow
-----------------

This tutorial demonstrates the complete RISE workflow:

**Step 1: Import and Prepare the Dataset**

Import your dataset as an AnnData object with preprocessed data.

**Step 2: Assess Variance Explained by RISE and PCA**

Determine the optimal component/rank by plotting the variance explained (R²X) across different ranks for both RISE and PCA. This helps balance model complexity with explanatory power.

.. code-block:: python

    from RISE.figures.commonFuncs.plotGeneral import plot_r2x
    import matplotlib.pyplot as plt

    ranks = [1, 5, 10, 15, 20, 25, 30]
    fig, ax = plt.subplots(figsize=(5, 5))

    plot_r2x(X, ranks, ax)
    plt.tight_layout()
    plt.show()

.. figure:: _static/tutorial_images/step2_r2x.png
   :align: center
   :width: 500px
   
   **Variance Explained (R²X) Plot.** This plot shows the variance explained for both RISE (PARAFAC2) and PCA across different ranks. Choose a rank where RISE captures more variance than PCA, indicating that tensor decomposition better models the multi-condition structure.

.. note::
   To see the actual output plots with figures, view the **Interactive Tutorial** section below which includes the Jupyter notebook with all executed cells and their visual outputs.

**Step 3: Evaluate Factor Stability with Factor Match Score (FMS)**

Measure the reproducibility of the RISE factorization across different ranks. An FMS above ~0.6 indicates stable components.

.. code-block:: python

    from RISE.figures.figureS4 import plot_fms_diff_ranks

    fig, ax = plt.subplots(figsize=(5, 5))
    rank_list = list(ranks)

    plot_fms_diff_ranks(X, ax, ranksList=rank_list, runs=3)
    plt.tight_layout()
    plt.show()

.. figure:: _static/tutorial_images/step3_fms.png
   :align: center
   :width: 500px
   
   **Factor Match Score (FMS) Plot.** The FMS measures stability of components across different ranks. Higher scores (above ~0.6) indicate reproducible factorization.

**Step 4: Perform RISE Factorization**

Based on the variance explained and FMS, select a rank and perform the RISE factorization. This decomposes the data into condition, eigenstate (cell), and gene factors.

.. code-block:: python

    from RISE.factorization import pf2

    rank = 20
    X = pf2(X=X, rank=rank, doEmbedding=True, tolerance=1e-9, 
            max_iter=500, random_state=42)

Setting ``doEmbedding=True`` automatically computes PaCMAP embeddings of the cell projections, which will be stored in ``X.obsm["embedding"]`` for visualization.

**Step 5: Visualize Condition Factor**

Examine how each experimental condition contributes to the identified patterns. Log-transforming these factors allows for easier interpretation of condition-specific effects.

.. code-block:: python

    from RISE.figures.commonFuncs.plotFactors import plot_condition_factors

    fig, ax = plt.subplots(figsize=(8, 8))

    plot_condition_factors(X, ax=ax, cond="Condition", log_transform=True)
    plt.tight_layout()
    plt.show()

.. figure:: _static/tutorial_images/step5_condition_factors.png
   :align: center
   :width: 700px
   
   **Condition Factors Heatmap.** This heatmap shows how each experimental condition (rows) contributes to each component (columns). Positive values (red) indicate upregulation, negative values (blue) indicate downregulation.

**Step 6: Visualize Cell Embedding**

Explore the latent space of cells using nonlinear dimensionality reduction methods such as PaCMAP. Label cells by cell type or experimental condition to understand clustering patterns.

.. code-block:: python

    from RISE.figures.commonFuncs.plotPaCMAP import plot_labels_pacmap

    fig, ax = plt.subplots(figsize=(8, 8))

    plot_labels_pacmap(X, labelType="Cell Type", ax=ax)
    plt.tight_layout()
    plt.show()

.. figure:: _static/tutorial_images/step6_cell_embedding.png
   :align: center
   :width: 700px
   
   **PaCMAP Cell Embedding.** The embedding shows cells in 2D space where similar cells cluster together. Cells are colored by cell type, revealing how different populations are distributed in the latent space.

**Step 7: Visualize Eigen-State Factor**

Analyze how each cell state contributes to the identified patterns. Each eigen-state represents a summary of similar cells with a distinct expression profile.

.. code-block:: python

    from RISE.figures.commonFuncs.plotFactors import plot_eigenstate_factors

    fig, ax = plt.subplots(figsize=(4, 4))

    plot_eigenstate_factors(X, ax=ax)
    plt.ylabel("Eigen-state")
    plt.tight_layout()
    plt.show()

.. figure:: _static/tutorial_images/step7_eigenstate_factors.png
   :align: center
   :width: 400px
   
   **Eigen-state Factors Heatmap.** This heatmap shows how each eigen-state (representing groups of similar cells) loads onto each component. High values indicate strong association with a component.

**Step 8: Visualize Gene Factor**

Identify which genes are highly weighted in each component, revealing coordinated gene modules. Adjust weight values to focus on genes that contribute significantly to the patterns.

.. code-block:: python

    from RISE.figures.commonFuncs.plotFactors import plot_gene_factors

    fig, ax = plt.subplots(figsize=(7, 8))

    plot_gene_factors(X, ax=ax, weight=0.2, trim=True)
    plt.tight_layout()
    plt.show()

.. figure:: _static/tutorial_images/step8_gene_factors.png
   :align: center
   :width: 650px
   
   **Gene Factors Heatmap.** This heatmap shows which genes (rows) are associated with each component (columns). The weight parameter filters out genes with low contributions for easier interpretation.

**Step 9: Investigate Gene Associations for a Component**

Overlay specific gene expression on the cell embedding to see which cells express genes of interest for a particular component.

.. code-block:: python

    from RISE.figures.commonFuncs.plotPaCMAP import plot_gene_pacmap

    fig, ax = plt.subplots(figsize=(8, 8))

    gene = "MS4A1"  # B cell marker
    plot_gene_pacmap(gene, X, ax=ax, clip_outliers=0.9995)
    plt.tight_layout()
    plt.show()

.. figure:: _static/tutorial_images/step9_gene_expression.png
   :align: center
   :width: 700px
   
   **Gene Expression on PaCMAP.** This visualization overlays MS4A1 gene expression (a B cell marker) onto the PaCMAP embedding. Cells are colored by expression level, revealing which populations express the gene.

**Step 10: Investigate Cell Associations for a Component**

Visualize how cells contribute to specific components using weighted projections, revealing subpopulations with distinct expression patterns.

.. code-block:: python

    from RISE.figures.commonFuncs.plotPaCMAP import plot_wp_pacmap

    fig, ax = plt.subplots(figsize=(8, 8))

    plot_wp_pacmap(X, cmp=10, ax=ax, cbarMax=0.9)
    plt.tight_layout()
    plt.show()

.. figure:: _static/tutorial_images/step10_weighted_projections.png
   :align: center
   :width: 700px
   
   **Weighted Projections for Component 10.** This plot shows which cells contribute most strongly to component 10. Cells with high weighted projections (bright colors) are most representative of that component's pattern.
