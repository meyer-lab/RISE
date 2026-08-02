API Reference
=============

Core Functions
--------------

Factorization
~~~~~~~~~~~~~

.. py:function:: pf2(X: anndata.AnnData, rank: int, random_state=1, doEmbedding: bool = True, tolerance=1e-6, max_iter: int = 100)
   :module: RISE.factorization

   Perform PARAFAC2 tensor decomposition on single-cell RNA-seq data.
   
   This is the main function for running RISE analysis. It decomposes the
   multi-condition single-cell data into condition factors, eigen-state factors,
   and gene factors, revealing patterns across experimental conditions.
   
   :param X: Preprocessed AnnData object containing single-cell RNA-seq data. Must have X.obs["condition_unique_idxs"] indicating which condition each cell belongs to (0-indexed).
   :type X: anndata.AnnData
   :param rank: Number of components to extract. Determines the complexity of the decomposition. Typically chosen based on variance explained and Factor Match Score analysis.
   :type rank: int
   :param random_state: Random seed for reproducibility of the decomposition.
   :type random_state: int, optional
   :param doEmbedding: If True, automatically computes PaCMAP embedding of cell projections and stores in X.obsm["X_pf2_PaCMAP"].
   :type doEmbedding: bool, optional
   :param tolerance: Convergence threshold for the optimization algorithm.
   :type tolerance: float, optional
   :param max_iter: Maximum number of iterations for the optimization algorithm.
   :type max_iter: int, optional
   :returns: The input AnnData object with added RISE decomposition results in X.uns["Pf2_weights"], X.uns["Pf2_A"], X.uns["Pf2_B"], X.varm["Pf2_C"], X.obsm["projections"], X.obsm["weighted_projections"], and X.obsm["X_pf2_PaCMAP"]
   :rtype: anndata.AnnData

Preprocessing
~~~~~~~~~~~~~

.. py:function:: prepare_dataset(X: anndata.AnnData, condition_name: str, geneThreshold: float, deviance: bool = False)
   :module: parafac2.normalize

   Preprocess single-cell RNA-seq data for RISE analysis.
   
   This function performs gene filtering, normalization, scaling, and log transformation.
   It also creates the condition_unique_idxs column required for RISE decomposition.
   
   :param X: AnnData object containing raw count data in sparse matrix format
   :type X: anndata.AnnData
   :param condition_name: Name of the column in X.obs that specifies experimental conditions for each cell
   :type condition_name: str
   :param geneThreshold: Minimum mean expression threshold for gene filtering
   :type geneThreshold: float
   :param deviance: If True, applies deviance transformation instead of log normalization
   :type deviance: bool, optional
   :returns: Preprocessed AnnData object with condition_unique_idxs and gene means
   :rtype: anndata.AnnData

Visualization Functions
-----------------------

General Plotting
~~~~~~~~~~~~~~~~

.. py:function:: plot_r2x(data: anndata.AnnData, rank_vec, ax: matplotlib.axes.Axes)
   :module: RISE.plotting

   Plot variance explained (R²X) for RISE and PCA across different ranks.
   
   This visualization helps determine the optimal number of components by showing
   how variance explained increases with rank. The elbow point where the curve
   flattens indicates a good balance between model complexity and explanatory power.
   
   :param data: Preprocessed AnnData object containing single-cell RNA-seq data.
   :type data: anndata.AnnData
   :param rank_vec: Array of rank values to test (e.g., [1, 5, 10, 15, 20, 25, 30]).
   :type rank_vec: array-like of int
   :param ax: Matplotlib axes object to plot on.
   :type ax: matplotlib.axes.Axes

Factor Plotting
~~~~~~~~~~~~~~~

.. py:function:: plot_condition_factors(data: anndata.AnnData, ax: matplotlib.axes.Axes, cond: str = 'Condition', log_transform: bool = True, cond_group_labels: pandas.Series = None, ThomsonNorm=False, color_key=None, group_cond=False)
   :module: RISE.plotting

   Plot condition factors as a heatmap showing how conditions contribute to components.
   
   This visualization shows how each experimental condition (rows) contributes to each RISE component (columns). High values indicate strong association between a condition and a component's pattern.
   
   :param data: AnnData object with RISE decomposition results. Must contain data.uns["Pf2_A"] and data.obs[cond].
   :type data: anndata.AnnData
   :param ax: Matplotlib axes object to plot on.
   :type ax: matplotlib.axes.Axes
   :param cond: Name of column in data.obs containing condition labels.
   :type cond: str, optional
   :param log_transform: If True, applies log10 transformation to condition factors before plotting.
   :type log_transform: bool, optional
   :param cond_group_labels: Series mapping conditions to group labels for colored row annotations.
   :type cond_group_labels: pandas.Series, optional
   :param ThomsonNorm: If True, normalizes factors using only control conditions.
   :type ThomsonNorm: bool, optional
   :param color_key: Custom colors for condition group labels.
   :type color_key: list, optional
   :param group_cond: If True and cond_group_labels provided, sorts conditions by group.
   :type group_cond: bool, optional

.. py:function:: plot_eigenstate_factors(data: anndata.AnnData, ax: matplotlib.axes.Axes)
   :module: RISE.plotting

   Plot eigen-state factors as a heatmap showing cell state patterns.
   
   Eigen-state factors represent the underlying cell state patterns across components. Each row represents an eigen-state and each column represents a component.
   
   :param data: AnnData object with RISE decomposition results. Must contain data.uns["Pf2_B"].
   :type data: anndata.AnnData
   :param ax: Matplotlib axes object to plot on.
   :type ax: matplotlib.axes.Axes

.. py:function:: plot_gene_factors(data: anndata.AnnData, ax: matplotlib.axes.Axes, weight=0.08, trim=True)
   :module: RISE.plotting

   Plot gene factors as a heatmap showing which genes contribute to each component.
   
   This visualization reveals coordinated gene modules by showing which genes (rows) are highly weighted in each component (columns).
   
   :param data: AnnData object with RISE decomposition results. Must contain data.varm["Pf2_C"].
   :type data: anndata.AnnData
   :param ax: Matplotlib axes object to plot on.
   :type ax: matplotlib.axes.Axes
   :param weight: Minimum absolute weight threshold for including genes.
   :type weight: float, optional
   :param trim: If True, filters genes based on the weight parameter.
   :type trim: bool, optional

PaCMAP Visualization
~~~~~~~~~~~~~~~~~~~~

.. py:function:: plot_labels_pacmap(X: anndata.AnnData, labelType: str, ax: matplotlib.axes.Axes, condition=None, cmap: str = 'tab20', color_key=None)
   :module: RISE.plotting

   Plot PaCMAP embedding colored by categorical labels (cell type or condition).
   
   This visualization shows the overall structure of the cell embedding, revealing how cells cluster by cell type or experimental condition.
   
   :param X: AnnData object with RISE decomposition results. Must contain X.obsm["X_pf2_PaCMAP"] and X.obs[labelType].
   :type X: anndata.AnnData
   :param labelType: Name of column in X.obs containing categorical labels to color by.
   :type labelType: str
   :param ax: Matplotlib axes object to plot on.
   :type ax: matplotlib.axes.Axes
   :param condition: If provided, only highlights cells from these specific conditions.
   :type condition: list of str, optional
   :param cmap: Matplotlib colormap name for coloring categories.
   :type cmap: str, optional
   :param color_key: Custom list of colors for categories.
   :type color_key: list, optional

.. py:function:: plot_gene_pacmap(gene: str, X: anndata.AnnData, ax: matplotlib.axes.Axes, clip_outliers=0.9995)
   :module: RISE.plotting

   Plot PaCMAP embedding colored by gene expression levels.
   
   This visualization overlays gene expression onto the PaCMAP embedding of cells, revealing which cell populations express specific genes.
   
   :param gene: Name of gene to visualize. Must be present in X.var_names.
   :type gene: str
   :param X: AnnData object with RISE decomposition results. Must contain X.obsm["X_pf2_PaCMAP"].
   :type X: anndata.AnnData
   :param ax: Matplotlib axes object to plot on.
   :type ax: matplotlib.axes.Axes
   :param clip_outliers: Quantile threshold for clipping extreme expression values.
   :type clip_outliers: float, optional

.. py:function:: plot_wp_pacmap(X: anndata.AnnData, cmp: int, ax: matplotlib.axes.Axes, cbarMax: float = 1.0)
   :module: RISE.plotting

   Plot PaCMAP embedding colored by weighted projections for a component.
   
   This visualization shows which cells contribute most strongly to a specific component by coloring them according to their weighted projections.
   
   :param X: AnnData object with RISE decomposition results. Must contain X.obsm["X_pf2_PaCMAP"] and X.obsm["weighted_projections"].
   :type X: anndata.AnnData
   :param cmp: Component number to visualize (1-indexed).
   :type cmp: int
   :param ax: Matplotlib axes object to plot on.
   :type ax: matplotlib.axes.Axes
   :param cbarMax: Maximum value for the color scale.
   :type cbarMax: float, optional

Factor Stability
~~~~~~~~~~~~~~~~

.. py:function:: plot_fms_diff_ranks(X: anndata.AnnData, ax: matplotlib.axes.Axes, ranksList: list[int], runs: int)
   :module: RISE.plotting

   Plot Factor Match Score (FMS) across different ranks to assess stability.
   
   FMS measures the reproducibility of PARAFAC2 decomposition results across multiple runs. Values above ~0.6 indicate stable, reproducible components.
   
   :param X: Preprocessed AnnData object containing single-cell RNA-seq data.
   :type X: anndata.AnnData
   :param ax: Matplotlib axes object to plot on.
   :type ax: matplotlib.axes.Axes
   :param ranksList: List of rank values to test (e.g., [1, 5, 10, 15, 20, 25, 30]).
   :type ranksList: list of int
   :param runs: Number of independent runs per rank to use for FMS calculation.
   :type runs: int


