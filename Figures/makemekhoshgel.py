

#%% The initialization

import numpy as np
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt
import scipy.signal as sgn
import seaborn as sns
from sklearn.cluster import  KMeans
import os
from sklearn.metrics import roc_curve, auc

def retrivedata(case,trg):
    df=pd.read_csv(case, skiprows=[0,2])[trg].dropna()
    return np.array(df)

def angs(x1,y1,x2,y2):
    theta=[]

    for i in range(0, len(x1)-1):
        if y2[i]-y1[i]>0:
            theta.append(np.arccos((x2[i]-x1[i])/(np.sqrt((x2[i]-x1[i])**2+(y2[i]-y1[i])**2)))*57.32)
        elif y2[i]-y1[i]<0:
                theta.append(180+(np.arccos((x1[i]-x2[i])/(np.sqrt((x2[i]-x1[i])**2+(y2[i]-y1[i])**2)))*57.32))
    return np.array(theta)


def smooth_diff(theta):  
    grad=np.gradient(theta)
    for i in range(0,len(grad)):
        if grad[i]>15 or grad[i]<-15 :
            grad[i]=grad[i-1]
    sm_dtheta=sgn.savgol_filter(grad, 25, 2)
    return sm_dtheta

def thet_head(x1,y1,x2,y2,x3,y3,x4,y4):         #1:tail, 2:mid_head, 3 HeadR, 4:HeadL
    u=np.array([x2-x1, y2-y1])                  #main body vector
    v=np.array([x4-x3, y4-y3])                  #head vector
    dotp=u[0]*v[0]+u[1]*v[1]                    #dot product
    si_u=np.sqrt(u[0]**2+u[1]**2)
    si_v=np.sqrt(v[0]**2+v[1]**2)
    thet_head=np.arccos(abs(dotp/(si_u*si_v)))*np.pi #*57.32   #(abs(dotp/(si_u*si_v)))*57.32 
    
    sm_head=np.pad(thet_head, 50, 'edge')                #pad the signal by its edge
    sm_head=sgn.savgol_filter(sm_head, 45, 4)        
    return sm_head[50:-50]-np.pi/2

def ar(x1,y1,x2,y2):
    ar=np.sqrt(np.square(x2-x1)+np.square(y2-y1))
    sm_ar=sgn.savgol_filter(ar, 45, 2)
    return np.array(sm_ar)


def steps(x, y):
    stp=[]
    for i in range(len(x)-1):
        stp.append(np.sqrt(np.square(x[i+1]-x[i])+np.square(x[i+1]-x[i])))
    sm_stp=sgn.savgol_filter(stp, 45, 2)
    return np.array(sm_stp)

#%%polar Plots LID


case=r"C:\dlc_projects\Analysis\exdata\lid1_3p.csv"
m_head_x=(retrivedata(case,'headR')+retrivedata(case, 'headL'))/2
m_head_y=(retrivedata(case, 'headR.1')+retrivedata(case, 'headL.1'))/2
tail_x=retrivedata(case, 'tail')
tail_y=retrivedata(case, 'tail.1')



ang_ht=np.array(angs(tail_x, tail_y, m_head_x, m_head_y))
d_ang=smooth_diff(ang_ht)
rad=ar(tail_x, tail_y, m_head_x, m_head_y)
rad=rad/np.max(rad)
LID_momen=steps(m_head_x, m_head_y)


strt=4000
fin=5000
plt.polar(ang_ht[strt:fin],rad[strt:fin], ls='', marker='o',color='darkorange', alpha=0.4)
plt.grid(linewidth=1, ls='--')
plt.title('LID', loc='left')
plt.ylim(0,1)


# %% SKF Example


case=r"C:\dlc_projects\Analysis\exdata\skf_3p.csv"
m_head_x=(retrivedata(case,'headR')+retrivedata(case, 'headL'))/2
m_head_y=(retrivedata(case, 'headR.1')+retrivedata(case, 'headL.1'))/2
tail_x=retrivedata(case, 'tail')
tail_y=retrivedata(case, 'tail.1')


strt=4000
fin=5000
ang_ht=np.array(angs(tail_x, tail_y, m_head_x, m_head_y))
d_ang=smooth_diff(ang_ht)
rad=ar(tail_x, tail_y, m_head_x, m_head_y)
rad=rad/np.max(rad)

plt.polar(ang_ht[strt:fin],rad[strt:fin], ls='', marker='o',color='blue', alpha=0.3)
plt.grid(linewidth=1, ls='--')
plt.title('SKF', loc='left')
plt.ylim(0,1)
SKF_momen=steps(m_head_x, m_head_y)

# %% SUM

case=r"C:\dlc_projects\Analysis\exdata\sum2_3p.csv"
m_head_x=(retrivedata(case,'headR')+retrivedata(case, 'headL'))/2
m_head_y=(retrivedata(case, 'headR.1')+retrivedata(case, 'headL.1'))/2
tail_x=retrivedata(case, 'tail')
tail_y=retrivedata(case, 'tail.1')


strt=3000
fin=4000
ang_ht=np.array(angs(tail_x, tail_y, m_head_x, m_head_y))
d_ang=smooth_diff(ang_ht)
rad=ar(tail_x, tail_y, m_head_x, m_head_y)
rad=rad/np.max(rad)

plt.polar(ang_ht[strt:fin],rad[strt:fin], ls='', marker='o',color='green', alpha=0.3)
plt.grid(linewidth=1, ls='--')
plt.title('SUM', loc='left')
plt.ylim(0,1)

SUM_momen=steps(m_head_x, m_head_y)

# %% LES


case=r"C:\dlc_projects\Analysis\exdata\les2_3p.csv"
m_head_x=(retrivedata(case,'headR')+retrivedata(case, 'headL'))/2
m_head_y=(retrivedata(case, 'headR.1')+retrivedata(case, 'headL.1'))/2
tail_x=retrivedata(case, 'tail')
tail_y=retrivedata(case, 'tail.1')

strt=3000
fin=4000
time_p=np.arange(0,len(tail_x)-1)/60
ang_ht=np.array(angs(tail_x, tail_y, m_head_x, m_head_y))
d_ang=smooth_diff(ang_ht)
rad=ar(tail_x, tail_y, m_head_x, m_head_y)
rad=rad/np.max(rad)

plt.polar(ang_ht[strt:fin],rad[strt:fin], ls='', marker='o',color='red', alpha=0.3)
plt.grid(linewidth=1, ls='--')
plt.title('Vehicle', loc='left')
plt.ylim(0,1)
BASE_momen=steps(m_head_x, m_head_y)



# %%

fig, ax=plt.subplots(2,2,squeeze=False)

ax[0][0].plot(time_p[2500:3500],LID_momen[2500:3500], label='L-Dopa', color='darkorange')
ax[0][0].set_title('L-Dopa')
ax[0][0].set_ylim(0,9)


ax[1][0].plot(time_p[2500:3500],SKF_momen[2500:3500], label='SKF', color='b')
ax[1][0].set_title('SKF')
ax[1][0].set_ylim(0,9)
ax[1][0].set_ylabel('Angular Moment')
ax[1][0].set_xlabel('Time (sec)')



ax[0][1].plot(time_p[2500:3500],SUM_momen[2500:3500], label='Sumanirole', color='green')
ax[0][1].set_title('Sumanirole')
ax[0][1].set_ylim(0,9)



ax[1][1].plot(time_p[2500:3500],BASE_momen[2500:3500], label='Vehicle', color='r')
ax[1][1].set_title('Vehicle')
ax[1][1].set_ylim(0,9)

sns.despine()
fig.tight_layout(pad=0.5)




#%%polar Plots LID


case=r"C:\dlc_projects\Analysis\exdata\LD6100.csv"
m_head_x=(retrivedata(case,'headR')+retrivedata(case, 'headL'))/2
m_head_y=(retrivedata(case, 'headR.1')+retrivedata(case, 'headL.1'))/2
tail_x=retrivedata(case, 'tail')
tail_y=retrivedata(case, 'tail.1')



ang_ht=np.array(angs(tail_x, tail_y, m_head_x, m_head_y))
d_ang=smooth_diff(ang_ht)
rad=ar(tail_x, tail_y, m_head_x, m_head_y)
rad=rad/np.max(rad)
LID_momen=steps(m_head_x, m_head_y)


strt=2000
fin=5000
plt.polar(ang_ht[strt:fin:2],rad[strt:fin:2], ls='', marker='o',color='limegreen', alpha=0.4)
plt.grid(linewidth=1, ls='--')
plt.title('LD6', loc='left')
plt.ylim(0,1)



#%%polar Plots LID


case=r"C:\dlc_projects\Analysis\exdata\D2A1_3p.csv"
m_head_x=(retrivedata(case,'headR')+retrivedata(case, 'headL'))/2
m_head_y=(retrivedata(case, 'headR.1')+retrivedata(case, 'headL.1'))/2
tail_x=retrivedata(case, 'tail')
tail_y=retrivedata(case, 'tail.1')



ang_ht=np.array(angs(tail_x, tail_y, m_head_x, m_head_y))
d_ang=smooth_diff(ang_ht)
rad=ar(tail_x, tail_y, m_head_x, m_head_y)
rad=rad/np.max(rad)
LID_momen=steps(m_head_x, m_head_y)


strt=1000
fin=4000
plt.polar(ang_ht[strt:fin:2],rad[strt:fin:2], ls='', marker='o',color='plum', alpha=0.4)
plt.grid(linewidth=1, ls='--')
plt.title('D2A', loc='left')
plt.ylim(0,1)



'''  
                                        The head_ang vs 

   '''
# %%


#%%polar Plots LID


case=r"C:\dlc_projects\Analysis\exdata\lid1_3p.csv"
m_head_x=(retrivedata(case,'headR')+retrivedata(case, 'headL'))/2
m_head_y=(retrivedata(case, 'headR.1')+retrivedata(case, 'headL.1'))/2
tail_x=retrivedata(case, 'tail')
tail_y=retrivedata(case, 'tail.1')

headl_l_x=retrivedata(case, 'headL')
headl_l_y=retrivedata(case, 'headL.1')

headl_r_x=retrivedata(case,'headR')
headl_r_y=retrivedata(case,'headR.1')

ang_ht=np.array(angs(tail_x, tail_y, m_head_x, m_head_y))
d_ang=smooth_diff(ang_ht)
rad=ar(tail_x, tail_y, m_head_x, m_head_y)
rad=rad/np.max(rad)
LID_momen=steps(m_head_x, m_head_y)



theta_head=thet_head(tail_x,tail_y,m_head_x,m_head_y,headl_r_x,headl_r_y,headl_l_x,headl_l_y)      #1:tail, 2:mid_head, 3 HeadR, 4:HeadL



strt=4000
fin=5000


plt.axes(projection = 'polar')
plt.polar(theta_head[strt:fin],rad[strt:fin], ls='', marker='o',color='darkorange', alpha=0.1)


plt.grid(linewidth=1, ls='--')
plt.title('LID', loc='left')
plt.ylim(0,1)

# %% les

case=r"C:\dlc_projects\Analysis\exdata\les2_3p.csv"
m_head_x=(retrivedata(case,'headR')+retrivedata(case, 'headL'))/2
m_head_y=(retrivedata(case, 'headR.1')+retrivedata(case, 'headL.1'))/2
tail_x=retrivedata(case, 'tail')
tail_y=retrivedata(case, 'tail.1')

headl_l_x=retrivedata(case, 'headL')
headl_l_y=retrivedata(case, 'headL.1')

headl_r_x=retrivedata(case,'headR')
headl_r_y=retrivedata(case,'headR.1')

ang_ht=np.array(angs(tail_x, tail_y, m_head_x, m_head_y))
d_ang=smooth_diff(ang_ht)
rad=ar(tail_x, tail_y, m_head_x, m_head_y)
rad=rad/np.max(rad)
LID_momen=steps(m_head_x, m_head_y)



theta_head=thet_head(tail_x,tail_y,m_head_x,m_head_y,headl_r_x,headl_r_y,headl_l_x,headl_l_y)      #1:tail, 2:mid_head, 3 HeadR, 4:HeadL



strt=4000
fin=5000


plt.axes(projection = 'polar')
plt.polar(theta_head[strt:fin],rad[strt:fin], ls='', marker='o',color='red', alpha=0.1)


plt.grid(linewidth=1, ls='--')
plt.title('Vehicle', loc='left')
plt.ylim(0,1)


# %%
