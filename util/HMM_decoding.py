

import sys
sys.path.insert(1, r'C:\dlc_projects\Analysis\Currencodes\DLC_refact\util')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from iteration import dlc_iter
from kinematics import dlc_kinematics


iter=dlc_iter()
kin=dlc_kinematics()







def decode(dct, remodel, ky, kin, t_point):

    features=kin.get_3Dembd_train(dct, ky, kin, t_point )
    
    labels=remodel.predict(features)

    return len(labels[labels==1])/len(labels)



def time_spent(dct, remodel, kin):
    characters=[]
    for ky in dct.keys():
        dlc_dct=dct[ky]['traces']
        
        kys=list(dct[ky]['traces'].keys())

        treatment=dct[ky]['treatment']
        

        i=0
        while i < len(kys):

            characters.append([decode(dct, remodel, ky, kin, kys[i])]+
                                                            [kys[int(i)], treatment, ky]) #could add self.ar0 here
            i+=1
    return characters



