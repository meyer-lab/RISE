# API Reference

## Factorization

::: scrise.factorization
    options:
      members:
        - pf2
        - export_factors
        - load_factors
        - correct_conditions

## Rank Selection

::: scrise.rank_selection
    options:
      members:
        - bicv
        - optimize_rank

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
        - plot_rank_optimization

### Factor Stability

::: scrise.plotting.stability
    options:
      members:
        - plot_fms_diff_ranks
