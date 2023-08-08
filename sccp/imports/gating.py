import umap 
import numpy as np
import pandas as pd
import umap.plot
import os
from os.path import join
from ..figures.common import openPf2

path_here = os.path.dirname(os.path.dirname(__file__))
 
 
def gateThomsonCells(rank, saveCellTypes=False):
    """Manually gates cell types for Thomson UMAP"""
    if saveCellTypes is True:
        _, _, projs = openPf2(rank, "Thomson")
        
        pf2Points = umap.UMAP(random_state=1).fit_transform(projs)

        umap1 = pf2Points[:, 0]
        umap2 = pf2Points[:, 1]

        cells = np.zeros(len(umap1))

        df = pd.DataFrame(data={"UMAP1": umap1, "UMAP2": umap2, "Cell Type": cells})

        idx = df.index[(df["UMAP1"] >= 5)].tolist()
        df.loc[idx, "Cell Type"] = "DCs"

        idx = df.index[(df["UMAP2"] >= 9.5)].tolist()
        df.loc[idx, "Cell Type"] = "B Cells"

        idx = df.index[(df["UMAP1"] >= -5) & (df["UMAP1"] <= 5) &
                    (df["UMAP2"] >= -3) & (df["UMAP2"] <= 5)].tolist()
        df.loc[idx, "Cell Type"] = "Monocytes"


        idx = df.index[(df["UMAP1"] <= -.75) &
                    (df["UMAP2"] >= 5) & (df["UMAP2"] <= 9.5)].tolist()
        df.loc[idx, "Cell Type"] = "NK Cells"

        idx = df.index[(df["UMAP1"] >= -.75) &
                    (df["UMAP2"] >= 5) & (df["UMAP2"] <= 9.5)].tolist()
        df.loc[idx, "Cell Type"] = "T Cells"

        np.save(join(path_here, "data/Thomson/Thomson_CellTypes.npy"),  df["Cell Type"].values)
        
    else:
        celltypes = np.load(join(path_here, "data/Thomson/Thomson_CellTypes.npy"), allow_pickle=True)
        
        return celltypes
        
        
