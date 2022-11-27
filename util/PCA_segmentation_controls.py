#%%

import numpy as np
import pandas as pd
from hmmlearn import hmm

import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


from kinematics import dlc_kinematics
from iteration import dlc_iter
from single_ts import dlc_ts

ts=dlc_ts()
kin=dlc_kinematics()
iter=dlc_iter()

def nl_hmm_scores(dct,t_point, range_comps, kin):

    score=[] #could be improved

    for ky in list(dct.keys()):


        features=kin.get_3Dembd_train(dct, ky, kin, t_point)


        for n_com in range(2,range_comps):

            model = hmm.GaussianHMM(n_components=n_com)
            model.fit(features)

            labels=model.predict(features)
            sc_r=metrics.silhouette_score(features, labels)

            score.append((ky, sc_r, n_com, '3D Embedding'))

    return score
        


def pca_hmm_scores(dct,t_point, range_comps, ts, dim=12):


    score=[]

    for ky in list(dct.keys()):


        features=ts.pca_12D_srs(dct, ky, t_point)
        pca = PCA()                                     #could  be optimized by n_comp
        x = StandardScaler().fit_transform(features)
        principalComponents = pca.fit_transform(x)


        for n_com in range(2,range_comps):

            model = hmm.GaussianHMM(n_components=n_com)
            model.fit(principalComponents[:,:dim])

            labels=model.predict(principalComponents[:,:dim])
            sc_r=metrics.silhouette_score(principalComponents[:,:dim], labels)

            score.append((ky, sc_r, n_com, '12D-PCA Embedding'))

    return score





def pca_spectrum(dct,t_points, ts):

    eigen_s=[] #could be improved
    for ky in list(dct.keys()):


        features=ts.pca_12D_srs(dct, ky, t_points)
        pca = PCA()
        x = StandardScaler().fit_transform(features)
        pca.fit_transform(x)
        eigens=pca.explained_variance_ratio_


        n_com=0 
        for eigen in eigens:

            eigen_s.append((ky, eigen, n_com, '12D-PCA Embedding'))
            n_com+=1

    return eigen_s





