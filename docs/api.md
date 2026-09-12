# API Reference

## Factorization

::: scrise.factorization
    options:
      members:
        - pf2
        - correct_conditions
        - order_components_by_energy
        - canonical_component_signs
        - match_components_across_ranks

## Factor Import/Export

::: scrise.factor_io
    options:
      members:
        - export_factors
        - load_factors

## Rank Selection

::: scrise.rank_selection
    options:
      members:
        - bicv

## Annotation Alignment

::: scrise.annotation_alignment

### Alignment Statistics

::: scrise.alignment_stats
    options:
      members:
        - compute_auroc_per_cell_type
        - compute_tau
        - compute_eta_squared
        - compute_kruskal_epsilon_squared

## Quantization & Compression

::: scrise.opq

## Preprocessing

::: parafac2.normalize
    options:
      members:
        - prepare_dataset

## Visualization Functions

### General Plotting

::: scrise.plotting.general
    options:
      members:
        - plot_r2x

### Factor Plotting

::: scrise.plotting.factors
    options:
      members:
        - plot_condition_factors
        - plot_eigenstate_factors
        - plot_gene_factors

### PaCMAP Visualization

::: scrise.plotting.pacmap
    options:
      members:
        - plot_labels_pacmap
        - plot_gene_pacmap
        - plot_wp_pacmap

### Rank Selection Plotting

::: scrise.plotting.rank_selection
    options:
      members:
        - plot_bicv_r2x

### Factor Stability

::: scrise.plotting.stability
    options:
      members:
        - plot_fms_diff_ranks

### Cell-Type Alignment Plotting

::: scrise.plotting.annotation_alignment
