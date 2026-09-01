# Unambiguous Component Ordering

## The problem

PARAFAC2 does not assign a natural order to components — a fit of rank $R$
returns $R$ components in an arbitrary order, and nothing in the model
itself says which one is "first." Some ordering has to be imposed after the
fact so that components can be compared, plotted, and referred to
consistently.

RISE previously ordered components by the Gini coefficient (variance-to-mean
ratio) of the condition factor $\mathbf{A}$. This ordering is a heuristic:
it is not particularly stable across refits, and — more importantly — it
gives no guarantee that, when the rank is increased from $N$ to $N+1$, the
first $N$ components of the new fit line up with the $N$ components of the
old fit. In practice, benchmarking has shown that when going from rank $N$
to rank $N+1$, $N$ of the components are usually carried over essentially
unchanged, with one new component appearing. A good ordering scheme should
make that behavior explicit: existing components should stay where they
are, and a newly added component should be appended at the high end of the
ordering, rather than being shuffled in among the others.

## Why cross-rank correspondence is well posed

This expectation is not just a convenience — it is backed by
**Harshman's Uniqueness Theorem** for PARAFAC2: given at least three
conditions and factors of full column rank, the decomposition is unique up
to only two ambiguities, permutation of the components and a sign/scale
indeterminacy shared between two of the three factor matrices. There is no
rotational freedom, unlike ordinary PCA or matrix factorization. That
uniqueness is what makes "does component $X$ at rank $N$ correspond to
component $Y$ at rank $N-1$" a well-defined question in the first place,
rather than an artifact of the optimizer's initialization or convergence
path — provided the components are first placed on a common, deterministic
footing (i.e. a fixed sign and a fixed ordering rule).

## The energy-based ordering

RISE orders components by their intrinsic energy,

$$
e_r = \lVert \mathbf{A}[:, r] \rVert \cdot \lVert \mathbf{C}[:, r] \rVert,
$$

the product of the condition-factor and gene-factor column norms for
component $r$. This quantity is directly determined by the fit: unlike the
Gini coefficient, it is not distorted by the arbitrary rescaling that can
occur between $\mathbf{A}$ and $\mathbf{C}$ during optimization (their
product is fixed by the fit, but how that product is split between the two
factors is not). Components are sorted from highest to lowest energy, so
that low-energy components — typically the ones that appear only once the
rank is increased — land at the high end of the ordering, while
established, high-energy components stay near the front. This is
implemented in [`RISE.order_components_by_energy`][RISE.factorization.order_components_by_energy]
and is applied automatically inside [`RISE.pf2`][RISE.factorization.pf2].

## Sign convention

Harshman's theorem only fixes the decomposition up to sign, so before
components can be compared — whether for ordering, or for any downstream
matching across ranks or refits — a canonical sign has to be chosen.
RISE adopts the convention that the largest-magnitude entry of each
component's gene factor ($\mathbf{C}$) column should be positive. When a
component's sign is flipped to satisfy this convention, the condition
factor ($\mathbf{A}$) column for that component is flipped correspondingly,
so that the reconstructed decomposition is unchanged; the eigen-state
factor ($\mathbf{B}$) is left as the unflipped reference. This is
implemented in
[`RISE.canonical_component_signs`][RISE.factorization.canonical_component_signs].

## Matching components across ranks

As a small, optional addition, RISE also provides
[`RISE.match_components_across_ranks`][RISE.factorization.match_components_across_ranks],
which implements the cross-rank matching primitive described above: given
the (sign-canonicalized) gene factors of a rank-$N$ fit and a rank-$(N+1)$
fit, it performs Hungarian maximum-weight matching on cosine similarity and
reports which components matched (above a similarity threshold, default
0.6) and which rank-$(N+1)$ component had no good match — i.e. the
candidate for the newly added component. This is a lightweight matching
utility, not a full cross-rank benchmarking pipeline; it is intended for
ad hoc checks of whether the energy ordering's expectation (that $N$
components carry over when moving to rank $N+1$) actually holds for a given
dataset.

```python
from RISE import match_components_across_ranks

matched_pairs, unmatched_high = match_components_across_ranks(
    C_low=X_rank_n.varm["Pf2_C"],
    C_high=X_rank_n_plus_1.varm["Pf2_C"],
)
```

Where the energy ordering and this matching disagree — for example, a
component that is not the lowest-energy one at rank $N+1$ turns out to be
the unmatched one — that disagreement is itself useful information: it
tends to flag a genuinely unstable or overfit component, rather than one
that can be silently reordered away.
