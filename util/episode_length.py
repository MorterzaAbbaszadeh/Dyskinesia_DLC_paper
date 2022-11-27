
#%% initiate
import sys
sys.path.insert(1, r'C:\dlc_projects\Analysis\Currencodes\DLC_refact\util')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
import sklearn.metrics as metrics
from iteration import dlc_iter
from kinematics import dlc_kinematics

from hmmlearn import hmm
iter=dlc_iter()
kin=dlc_kinematics()






#%% Funcs



def leng(labels):

    labels[0]=0
    labels[-1]=0
    d_label=labels[1:]-labels[:-1]
    bgins=np.where(d_label==1)[0]
    ends=np.where(d_label==-1)[0]

    return ends-bgins



def leng_decode(dct, remodel, ky, kin, t_point):

    features=kin.get_3Dembd_train(dct, ky, kin, t_point)
    
    labels=remodel.predict(features)

    return leng(labels)


def norm_leng_decode(dct, remodel, ky, kin, t_point):

    features=kin.get_3Dembd_train(dct, ky, kin, t_point)
    
    labels=remodel.predict(features)

    lengs=leng(labels)

    if lengs.shape[0] >0 :
        return lengs/np.max(lengs)
    else:      
        return lengs

def lengths(dct, remodel, kin, t_point):


    lengs={}
    for ky in dct.keys():


        dlc_dct=dct[ky]['traces']

        kys=list(dlc_dct.keys())
        treatment=dct[ky]['treatment']
        id=dct[ky]['id']
        i=0

        

        for  t  in t_point:
            t=str(t)

            lengs[treatment+id+t]={}
            lengs[treatment+id+t]['treatment']= treatment
            lengs[treatment+id+t]['lengs']=leng_decode(dct, remodel, ky, kin, t)

            i+=1


    return lengs


def norm_lengths(dct, remodel, kin, t_point):


    norm_lengs={}
    for ky in dct.keys():


        dlc_dct=dct[ky]['traces']

        kys=list(dlc_dct.keys())
        treatment=dct[ky]['treatment']
        id=dct[ky]['id']
        i=0

        

        for  t  in t_point:
            t=str(t)

            norm_lengs[treatment+id+t]={}
            norm_lengs[treatment+id+t]['treatment']= treatment
            norm_lengs[treatment+id+t]['lengs']=norm_leng_decode(dct, remodel, ky, kin, t)

            i+=1


    return norm_lengs

def sort_treat_histo(lengs, treat):
    histo=np.empty(1)
    for ky in lengs.keys():
        if lengs[ky]['treatment']==treat:
            
            histo=np.concatenate((histo,np.array(lengs[ky]['lengs'])))
    return histo[1:]





''''n, bins, patches = plt.hist(x=lid_histo, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85, density=True)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Episode Length')
plt.ylabel('Probability')
plt.title('LID Episode Length Histogram')
plt.xlim(0, 200)
maxfreq = n.max()


'''