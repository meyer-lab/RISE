from parafac2.normalize import prepare_dataset

from . import plotting
from .annotation_alignment import (
    CellTypeAlignmentResults,
    ComponentAlignmentResult,
    cell_type_alignment,
    compute_tau,
    score_cell_type_alignment,
)
from .factorization import (
    canonical_component_signs,
    correct_conditions,
    export_factors,
    load_factors,
    match_components_across_ranks,
    order_components_by_energy,
    pf2,
    rise_pca_r2x,
)
from .opq import OPQQuantizer, find_optimal_opq
from .rank_selection import bicv

__version__ = "1.2.0"

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
