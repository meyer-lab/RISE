"""
S3d: Plot samples along two components to see patient separation
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: see if SLE/healthy samples can be stratified along strongly predictive Pf2 components
# (they can, at least when you do 13 and 26)

# load functions/modules ----
from .common import (
    subplotLabel,
    getSetup,
    openPf2
)
from .commonFuncs.plotLupus import (
    plot2DSeparationByComp
)
from ..imports.scRNA import load_lupus_data
import numpy as np
import pandas as pd

def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12,12), # fig size
                     (2,2) # grid size
                     )

    # Add subplot labels
    subplotLabel(ax)

    rank = 40
    group_to_predict = "SLE_status" # group to predict 

    lupus_tensor, obs = load_lupus_data() 

    group_labs = (
            obs[['sample_ID', group_to_predict]]
            .drop_duplicates()
        )

    group_labs = group_labs.set_index('sample_ID')

    _, factors, _, = openPf2(rank = rank, dataName = 'lupus', optProjs=True)

    factor_A = pd.DataFrame(factors[0],
                            columns = [f"Cmp. {i}" for i in np.arange(1, rank + 1)],
                            index = lupus_tensor.condition_labels)

    merged = factor_A.merge(group_labs, left_index=True, right_index=True)

    # components can be varied; including these for now because 13 and 26 seemed
    # to have high weights in logistic regression, as do 32 and 29. 32 and 13 both
    # had positive weights while 26 and 29 were negative
    comps_to_test = [('Cmp. 13', 'Cmp. 26'),
                     ('Cmp. 32', 'Cmp. 26'),
                     ('Cmp. 13', 'Cmp. 32'),
                     ('Cmp. 13', 'Cmp. 29')]
    
    for i, pair in enumerate(comps_to_test):
        plot2DSeparationByComp(merged, pair, group_to_predict, ax[i])

    return f