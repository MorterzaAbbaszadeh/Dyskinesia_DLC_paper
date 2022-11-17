
#global init
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from hmmlearn import hmm
import pickle


#local init
import sys
sys.path.insert(1, r'C:\dlc_projects\Analysis\git_repo\util')

from iteration import dlc_iter
from kinematics import dlc_kinematics
iter=dlc_iter()
kin=dlc_kinematics()




#initialize parameters
t_points=[30] #training time point
dim=3 #training dimension
n_com=2 #number of HMM states

evaluate=['LID']
iter.treatments=evaluate #set treatment groups
file='C:\dlc_projects\Analysis\git_repo\data\d_base.npy'




#core functions
def get_train_features(dct, t_point, kin, dim=3):
    features=np.empty((0,dim))
    print(features)

    for ky in list(dct.keys()):
        features=np.vstack((features,kin.get_3Dembd_train(dct,  ky, kin, t_point)))

    return features



def train_hmm(file, t_points, dim, n_com, dct):

    #get training input
    features=np.empty((0,dim))
    for t_point in t_points:
        features=np.vstack((features, get_train_features(dct, t_point, kin)))


    #train on the data set with 2 components
    model = hmm.GaussianHMM(n_components=n_com, n_iter=14500)
    model.fit(features)
    return model



#prepare the dataset
dct=iter.get_treatments(file)
model=train_hmm(file, t_points, dim, n_com, dct)

with open("3DHMM.pkl", "wb") as file: pickle.dump(model, file)
