"""
Plotting and visualization functions for RISE.
"""

from .factors import (
    plot_condition_factors,
    plot_eigenstate_factors,
    plot_gene_factors,
)
from .general import (
    avegene_per_status,
    cell_count_perc_df,
    cell_count_perc_lupus_df,
    gene_plot_cells,
    plot_avegene_per_category,
    plot_avegene_per_celltype,
    plot_cell_gene_corr,
    plot_r2x,
    rotate_xaxis,
    rotate_yaxis,
)
from .pacmap import (
    plot_gene_pacmap,
    plot_labels_pacmap,
    plot_wp_pacmap,
)
from .rank_selection import (
    plot_bicv_r2x,
    plot_rank_optimization,
)
from .stability import (
    calculateFMS,
    plot_fms_diff_ranks,
    plot_fms_percent_drop,
)

__all__ = [
    "avegene_per_status",
    "calculateFMS",
    "cell_count_perc_df",
    "cell_count_perc_lupus_df",
    "gene_plot_cells",
    "plot_avegene_per_category",
    "plot_avegene_per_celltype",
    "plot_bicv_r2x",
    "plot_cell_gene_corr",
    "plot_condition_factors",
    "plot_eigenstate_factors",
    "plot_fms_diff_ranks",
    "plot_fms_percent_drop",
    "plot_gene_factors",
    "plot_gene_pacmap",
    "plot_labels_pacmap",
    "plot_r2x",
    "plot_rank_optimization",
    "plot_wp_pacmap",
    "rotate_xaxis",
    "rotate_yaxis",
]
