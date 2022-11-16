

#%%
from os import walk, listdir, path
import dlcppross as dlcp
import numpy as np
import pandas as pd

cases=[]
pths=[r"C:\dlc_projects\video_rep\Main", r"C:\dlc_projects\video_rep\Controls"]
parts=['headR','headR.1', 'headL','headL.1', 'tail', 'tail.1']


#%%

def retrivedata(case):                      #load the csv file and omit the column titles
    df=pd.read_csv(case, skiprows=[0,2]).replace(np.nan,0) #drop na if any
    return df

def get_ar(case):
    m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2
    m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
    tail_x=dlcp.retrivedata(case, 'tail')
    tail_y=dlcp.retrivedata(case, 'tail.1') 

    return dlcp.ar(m_head_x,m_head_y,tail_x,tail_y)


#for base R
#data structure
# d_base/id/{trace_in_timepoint}


def const_dict(root, parts):

    dct={}
    dct['traces']={}

    for fname in listdir(root):
        

        if fname[-3:]=='csv':

            #print(fname)
            case=path.join(root,fname)
            df=retrivedata(case)
        

            

            dct['treatment']=fname[0:3]
            dct['id']=fname[3:5]

            if fname[5:7]=='00':
                dct['ar0']=np.mean(get_ar(case))
                time='10'
            elif fname[5:7]=='05':
                dct['ar0']=np.mean(get_ar(case))
                time='10'
            elif fname[5:7]=='11':
                time='100'
            else:
                time=fname[5:7]

            
            dct['traces'][time]={}

            for pt in parts:
                dct['traces'][time][pt]=df[pt].values


    return dct


d_base={}
for pth in pths:
    

    for root, dirs, files in walk(pth):
        for fn in listdir(root):

            if fn[-3:]=='csv':
                print(root)
                u=const_dict(root, parts)
                d_base[u['id']+'_'+u['treatment']]=u
                break


#%%

with open('d_base.npy', 'wb') as f:
    np.save(f, d_base , allow_pickle=True)
# %%
