

#%%


import numpy as np
import pandas as pd
import seaborn as sns
import dlcppross as dlcp
import scipy.signal as sgn

from iteration import dlc_iter
from kinematics import dlc_kinematics





#%%


file='d_base.npy'
groups=['LID', 'SKF', 'SUM', 'D2A', ]

kinematic=dlc_kinematics() 
iter=dlc_iter()



funcs= (kinematic.npy_ar, kinematic.npy_rot_speed, kinematic.npy_thet_head, kinematic.npy_translation)
df_ready=iter.get_sns(file, groups, funcs)


#%%


k_df=pd.DataFrame(df_ready, columns=[i.__name__ for i in funcs]+['time','treatment','id']).dropna()
k_df['ttreatment'] = k_df['treatment'].replace(['SKF'],'D1Ag')
k_df['treatment'] = k_df['treatment'].replace(['SUM'],'D2Ag')
k_df.head()

# %%


sns.boxplot(x='time', y='npy_ar', hue='treatment', data=k_df, showfliers=False )
sns.despine()


# %%

sns.boxplot(x='time', y='npy_rot_speed', hue='treatment', data=k_df, showfliers=False )
sns.despine()


# %%

sns.boxplot(x='time', y='npy_thet_head', hue='treatment', data=k_df, showfliers=False )
plt.ylabel('Head Angle')
sns.despine()



# %%

sns.boxplot(x='time', y='npy_translation', hue='treatment', data=k_df, showfliers=False )
sns.despine()






'''                             D1-Controls
'''


#%%


file='d_base.npy'
groups=[ 'SKF', 'D1A', 'SUM']

kinematic=dlc_kinematics() 
iter=dlc_iter()


funcs= (kinematic.npy_ar, kinematic.npy_rot_speed, kinematic.npy_thet_head, kinematic.npy_translation)
df_ready2=iter.get_sns(file, groups, funcs)


#%%

k_df2=pd.DataFrame(df_ready2, columns=[i.__name__ for i in funcs]+['time','treatment','id']).dropna()

k_df2['treatment'] = k_df2['treatment'].replace(['SKF'],'D1-Ag')
k_df2['treatment'] = k_df2['treatment'].replace(['SUM'],'D2-Ag')
k_df2['treatment'] = k_df2['treatment'].replace(['D1A'],'D1-Antagonist')

k_df2 = k_df2[k_df2.time != '100']
k_df2 = k_df2[k_df2.time != '05']
k_df2 = k_df2[k_df2.time != '90']
k_df2 = k_df2[k_df2.time != '00']

# %%
import matplotlib.pyplot as plt


sns.pointplot(x='time', y='npy_translation', hue='treatment', data=k_df2, showfliers=False )

plt.ylabel('Total translation of center point')
sns.despine()




'''                             D2-Controls
'''


#%%


file='d_base.npy'
groups=[ 'LID', 'D2A']

kinematic=dlc_kinematics() 
iter=dlc_iter()


funcs= (kinematic.npy_ar, kinematic.npy_rot_speed, kinematic.npy_thet_head, kinematic.npy_translation)
df_ready2=iter.get_sns(file, groups, funcs)


#%%

k_df2=pd.DataFrame(df_ready2, columns=[i.__name__ for i in funcs]+['time','treatment','id']).dropna()

k_df2['treatment'] = k_df2['treatment'].replace(['LID'],'LD3')
k_df2['treatment'] = k_df2['treatment'].replace(['D2A'],'D2-Antagonist')


k_df2 = k_df2[k_df2.time != '100']
k_df2 = k_df2[k_df2.time != '05']
k_df2 = k_df2[k_df2.time != '90']
k_df2 = k_df2[k_df2.time != '00']

# %%
import matplotlib.pyplot as plt


sns.pointplot(x='time', y='npy_translation', hue='treatment', data=k_df2, showfliers=False )

plt.ylabel('Total translation of center point')
sns.despine()


# %%


sns.pointplot(x='time', y='npy_translation', hue='treatment', data=k_df2, showfliers=False )

plt.ylabel('Total translation of center point')
sns.despine()
