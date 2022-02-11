

import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, join
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.metrics.cluster import adjusted_rand_score,rand_score
from sklearn.model_selection import KFold
from pathlib import Path
from sklearn.mixture import GaussianMixture


def GMMpca(ax,gmmtype,zflowDF,maxcluster,ksplit):

    arr = np.arange(1,5,1)
    celltypelist = zflowDF.CellType.values
    totalDF = zflowDF.drop(columns=['CellType','pSTAT5']) # Creating matrix that will be used in GMM model
    clusternumb = np.arange(2,maxcluster) # Amount of clusters

    for ii in range(len(celltypelist)): # Changing cell types to a number
        if celltypelist[ii] == 'None':
            one = 1
            celltypelist[ii] = one
        elif celltypelist[ii] == 'Treg':
            two = 2
            celltypelist[ii] = two
        else: #Thelper
            three = 3 
            celltypelist[ii] = three

    pcagmmDF = pd.DataFrame(columns=["Component","Cluster","Score"])

    for jj in range(len(arr)):
        # Iterating through different number of components to determine which is best
        pca = PCA(n_components=arr[jj])
        pcaDF = pca.fit_transform(totalDF)
        kf = KFold(n_splits=ksplit) # Cross validation for amount of splits
        gmm_labels = np.zeros([len(pcaDF)])
        bestguess = np.zeros([len(clusternumb)])
        for kk in range(len(clusternumb)):
            GMM = GaussianMixture(n_components = clusternumb[kk],covariance_type = 'full',tol = .001,max_iter = 5000,
            reg_covar = 1e-3)

            # Running GMM model on PCA dataset

            if gmmtype == "RandScore":
                for train_index, test_index in kf.split(pcaDF):
                    trainX = pcaDF[train_index,:]
                    GMM.fit(trainX)
                    gmm_labels[test_index] = GMM.predict(pcaDF[test_index,:])

                bestguess = rand_score(celltypelist,gmm_labels) # Comparing the cell type with the GMM predicted 
                pcagmmDF = pcagmmDF.append(pd.DataFrame({"Component":[arr[jj]],"Cluster":[clusternumb[kk]],"Score": [bestguess]}))

            else: 
                gmm_score = []
                for train_index, test_index in kf.split(pcaDF):
                    trainX = pcaDF[train_index,:]
                    GMM.fit(trainX)
                    gmm_labels[test_index] = GMM.predict(pcaDF[test_index,:])
                    gmm_score.append(GMM.score(pcaDF[test_index,:]))

                score_arr = np.mean(gmm_score)
                pcagmmDF = pcagmmDF.append(pd.DataFrame({"Component":[arr[jj]],"Cluster":[clusternumb[kk]],"Score": [score_arr]}))


    for ll in range(len(arr)):
        randDF = pcagmmDF.loc[pcagmmDF.Component == arr[ll]]
        ax.plot(randDF.Cluster.values,randDF.Score.values,label = arr[ll])


    ax.legend(title = "Component Number", loc = 'best')
    xlabel = "Cluster Number"
    ylabel = "Score"
    ax.set(xlabel=xlabel, ylabel=ylabel)

