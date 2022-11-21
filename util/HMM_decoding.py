

import sys
sys.path.insert(1, r'C:\dlc_projects\Analysis\git_repo\util')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from iteration import dlc_iter
from kinematics import dlc_kinematics


iter=dlc_iter()
kin=dlc_kinematics()







def decode(dct, remodel, ky, kin, t_point):

 

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






def time_spent(dct, remodel, kin):
    characters=[]
    for ky in dct.keys():
        
        
        kys=list(dct[ky]['traces'].keys())

        treatment=dct[ky]['treatment']


        i=0
        while i < len(kys):

            characters.append([decode(dct, remodel, ky, kin, kys[i])]+
                                                            [kys[int(i)], treatment, ky]) #could add self.ar0 here
            i+=1
    return characters



def inside_out(dct, model, kin, t_point):
    
    in_and_out=[]


    for ky in dct.keys():
        

        treatment=dct[ky]['treatment']


        features=kin.get_3Dembd_train(dct, ky, kin, t_point )
        labels=model.predict(features)
        

        inside= features[[labels==1][0],:]
        outside= features[[labels==0][0],:]
        inside_ar, inside_head_ang, inside_absrot_speed=np.mean(inside[:,0]),np.mean(inside[:,1]), np.mean(inside[:,2])
        outside_ar, outside_head_ang, outside_absrot_speed=np.mean(outside[:,0]),np.mean(outside[:,1]), np.mean(outside[:,2])

        in_and_out.append((inside_ar, inside_head_ang, inside_absrot_speed,ky, treatment, 'inside'))
        in_and_out.append((outside_ar,outside_head_ang, outside_absrot_speed,ky, treatment, 'outside'))


    return in_and_out
