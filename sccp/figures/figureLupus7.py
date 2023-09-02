"""
S3b: Logistic Regression (and maybe SVM) on Pf2 Factor matrix A output
article: https://www.science.org/doi/10.1126/science.abf1970
data: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174188
"""

# GOAL: run logisitc regression to see which components are best able to predict disease status

# load functions/modules ----
from .common import (
    subplotLabel,
    getSetup,
    openPf2,
    plotROCAcrossGroups
)
from ..imports.scRNA import load_lupus_data


def makeFigure():
    """Get a list of the axis objects and create a figure."""
    # Get list of axis objects
    ax, f = getSetup((12, 6), # fig size
                     (1, 1) # grid size
                     )

    # Add subplot labels
    subplotLabel(ax)

    rank = 40


    _, factors, _ = openPf2(rank = rank, dataName = 'lupus', optProjs=True)

    _, obs = load_lupus_data() 

    status = (
              obs[['sample_ID', 'SLE_status', 'Processing_Cohort']]
             .drop_duplicates()
              )
    
    group_labs = status.set_index('sample_ID')[['SLE_status', 'Processing_Cohort']]
        
    A_matrix = factors[0]   


    plotROCAcrossGroups(A_matrix, group_labs, ax[0],
                        pred_group='SLE_status',
                        cv_group='Processing_Cohort')

    
    return f