from parafac2.normalize import prepare_dataset

from . import plotting
from .alignment_stats import compute_tau
from .annotation_alignment import (
    CellTypeAlignmentResults,
    ComponentAlignmentResult,
    cell_type_alignment,
    score_cell_type_alignment,
)
from .factor_io import export_factors, load_factors
from .factorization import (
    canonical_component_signs,
    correct_conditions,
    match_components_across_ranks,
    order_components_by_energy,
    pf2,
    rise_pca_r2x,
)
from .opq import OPQQuantizer, find_optimal_opq
from .rank_selection import bicv

__version__ = "1.3.0"

__all__ = [
    "CellTypeAlignmentResults",
    "ComponentAlignmentResult",
    "OPQQuantizer",
    "__version__",
    "bicv",
    "canonical_component_signs",
    "cell_type_alignment",
    "compute_tau",
    "correct_conditions",
    "export_factors",
    "find_optimal_opq",
    "load_factors",
    "match_components_across_ranks",
    "order_components_by_energy",
    "pf2",
    "plotting",
    "prepare_dataset",
    "rise_pca_r2x",
    "score_cell_type_alignment",
]
