# Handling Unequal Cell Counts Across Conditions

## The problem

RISE fits a PARAFAC2 decomposition by alternating least squares (ALS). At each
iteration, every condition slice $X_i$ contributes to the shared factor updates
through matricized-tensor-times-Khatri-Rao-product (MTTKRP) terms computed as a
raw sum over the cells in that slice. Because these contributions scale directly
with the number of cells (and therefore with total sequencing depth) in a
condition, decompositions can behave poorly when conditions have very disparate
cell counts.

This shows up most often in experimental designs where one condition
aggregates many more cells than the others — for example, pooling all control
or non-targeting cells from a Perturb-seq experiment into a single condition,
while each perturbation condition retains only a few hundred to a few thousand
cells. In that setting:

- The shared factors ($\mathbf{B}$ and $\mathbf{C}$) are dominated by the
  large slice, since it contributes disproportionately to every MTTKRP update.
- The condition factor $\mathbf{A}$ ends up correlated with cell count and
  sequencing depth rather than with the underlying biological effect.
- Conditions with genuinely strong biological effects but relatively few
  cells (e.g. essential-gene knockdowns with a strong growth phenotype) can be
  suppressed or buried underneath the oversized condition.

This is purely an artifact of slice size, not biology, and it can persist even
after the [post-fit read-depth correction](#fix-2-post-fit-read-depth-correction)
described below, since that correction addresses total counts per condition
rather than the number of cells contributing to the ALS updates.

## Fix 1: Slice normalization during fitting

The most direct fix is to prevent large slices from dominating the fit in the
first place. [`scrise.pf2`][scrise.factorization.pf2] exposes a
`normalize_slices` argument for this purpose:

```python
from scrise import pf2

X = pf2(X, rank=20, normalize_slices=True)
```

When `normalize_slices=True`, each mean-centered condition slice is weighted
by the inverse of its own Frobenius norm,
$w_i = 1 / \lVert X_i - \mathbf{1}\mu^T \rVert_F$, before it contributes to
the mode-0, mode-1, and mode-2 factor updates. This puts every condition on
comparable footing during fitting regardless of its cell count, so components
reflect the relative strength of each condition's biological signal rather
than how many cells were sequenced for it. The underlying data $X_i$ is never
modified, and the reported reconstruction error ($R^2X$) is always computed
against the unweighted residuals, so it remains directly comparable across
runs with and without `normalize_slices`.

`normalize_slices` defaults to `False` to preserve existing behavior. Enable
it whenever your dataset has conditions with substantially different cell
counts, and especially when one condition (such as a pooled control) is much
larger than the rest.

## Fix 2: Post-fit read-depth correction

RISE also provides a post-hoc correction,
[`correct_conditions`][scrise.factorization.correct_conditions], that can be
applied after fitting:

```python
from scrise import correct_conditions

X.uns["Pf2_A"] = correct_conditions(X)
```

This function regresses the geometric mean of each condition's factor
weights (across components) against that condition's total read count, and
divides the condition factors by the fitted trend. It removes the portion of
condition-factor magnitude that is linearly explained by sequencing depth,
without needing to refit the tensor decomposition.

Because this correction is fit after the decomposition, it cannot undo
distortions that already propagated into the shared factors ($\mathbf{B}$ and
$\mathbf{C}$) during ALS — it only rescales the condition factors
$\mathbf{A}$ after the fact. It is most useful as a lightweight adjustment
for moderate depth differences, or as a complement to `normalize_slices`.

## Recommendation

- For datasets with substantially unequal cell counts per condition (for
  example, a large pooled control condition alongside much smaller
  perturbation conditions), fit with `normalize_slices=True`. This addresses
  the imbalance where it originates, in the ALS updates themselves.
- `correct_conditions` can still be applied afterward as an additional,
  cheap correction for residual read-depth effects, but it is not a
  substitute for `normalize_slices` when cell count imbalance is severe,
  since it cannot correct the shared factors.
