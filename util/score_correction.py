
#%%
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.signal as sgn
import matplotlib.pyplot as plt


from visualization_config import visual_config
vis=visual_config()

#%%





times=np.linspace(20, 140, 7)
exc_f=r'C:\dlc_projects\Analysis\git_repo\data\DLC_dyskinesia_UpdatedFile_100322.xlsx'
ex=pd.read_excel(exc_f,sheet_name=None)

pages={'LD-3mg': list(ex.keys())[2],
        'D2Ag':list(ex.keys())[0] ,
        'D1Ag':list(ex.keys())[1]}



#df=pd.read_excel(exc_f, sheet_name='LD 3')

#df.melt(id_vars=['Minute', 'AIMS'])

f_df=pd.DataFrame()
for ky in list(pages.keys()):
    df=pd.read_excel(exc_f, sheet_name=pages[ky])
    df=df.melt(id_vars=['Minute', 'AIMS'])
    df['treatment']=ky

    f_df=pd.concat((f_df,df))


#f_df.to_csv( r'C:\dlc_projects\Analysis\git_repo\data\dyskinesia_scores.csv')





'''                                         Do Stuff
'''




# %% Over all Aims Scores

scores_df=pd.read_csv(r'C:\dlc_projects\Analysis\git_repo\data\dyskinesia_scores.csv')

scores_df.head()

#%%

ax1=sns.pointplot(x='Minute', y='value', hue='treatment',
             data=scores_df.loc[scores_df['AIMS'].isin(['Ax'])], palette=vis.treatment_colors, ci=95 )



xlabel='Time after treatment(Min)'
ylabel='Axial Score'
 

xlim_right=6.7
ax1=vis.hmm_plots(ax1, xlabel, ylabel, xlim_right)
plt.savefig('Treatment_timespent.svg')
plt.draw()



# %%

ax2=sns.pointplot(x='Minute', y='value', hue='treatment', data=scores_df.loc[scores_df['AIMS'].isin(['O'])], palette=vis.treatment_colors, ci=95 )



xlabel='Time after treatment(Min)'
ylabel='Orofacial Score'

xlim_right=6.7

ax2=vis.hmm_plots(ax2, xlabel, ylabel, xlim_right,legend=True)
plt.savefig('Treatment_timespent.svg')
plt.draw()


# %%
