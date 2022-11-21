#%%
import numpy as np
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import dlcppross as dlcp
from scipy import stats
import sklearn.metrics as metrics

from sklearn.covariance import EmpiricalCovariance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



#%% get the scores standard


pth='/home/morteza/dlc_projects/videorep'



score=[]
for root, dirs, files in os.walk(pth):
    for fname in files:
        if fname.endswith('.csv') and fname.startswith('LID') and fname[5:7]=='30' :
            case=root+'/'+fname


            m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2
            m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
            tail_x=dlcp.retrivedata(case, 'tail')
            tail_y=dlcp.retrivedata(case, 'tail.1')

            ang_head=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
            d_ang=dlcp.smooth_diff(ang_head)
            d_ang2=d_ang.reshape(-1, 1)
            

            for i in range(2,15):
                n_com=i
                model1 = hmm.GaussianHMM(n_components=n_com, covariance_type="full", n_iter=4500)
                model1.fit(d_ang2)
                label1=model1.predict(d_ang2)
                sc_r=metrics.silhouette_score(d_ang2, label1)
                score.append((fname[3:5], sc_r, i, 'LID'))


scores=pd.DataFrame(score, columns=['case_name', 'score', 'n_clusters', 'treatment'])

# %%
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_palette(sns.color_palette(flatui))
ax=sns.lineplot(x='n_clusters', y='score',data=scores)
plt.ylim(0,0.6)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
sns.despine()

# %%



#%% get the scores head_tail


pth='/home/morteza/dlc_projects/videorep'



score=[]
for root, dirs, files in os.walk(pth):
    for fname in files:
        if fname.endswith('.csv') and fname.startswith('LID') and fname[5:7]=='30' :
            case=root+'/'+fname


            m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2
            m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
            tail_x=dlcp.retrivedata(case, 'tail')
            tail_y=dlcp.retrivedata(case, 'tail.1')

            head_ang=dlcp.thet_head(tail_x,tail_y, m_head_x,m_head_y,dlcp.retrivedata(case,'headL'),dlcp.retrivedata(case,'headL.1'),
                dlcp.retrivedata(case,'headR'),dlcp.retrivedata(case,'headR.1'))
            head_ang2=head_ang.reshape(-1, 1)
            for i in range(2,15):
                n_com=i
                model1 = hmm.GaussianHMM(n_components=n_com, covariance_type="full", n_iter=4500)
                model1.fit(head_ang2)
                label1=model1.predict(head_ang2)
                sc_r=metrics.silhouette_score(head_ang2, label1)
                score.append((fname[3:5], sc_r, i, 'LID'))


scores=pd.DataFrame(score, columns=['case_name', 'score', 'n_clusters', 'treatment'])

# %% HEAD
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_palette(sns.color_palette(flatui))
ax=sns.lineplot(x='n_clusters', y='score',data=scores)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
sns.despine()


#%% Multi variable clustering



pth='/home/morteza/dlc_projects/video_rep/Main/LID'



score=[]
for root, dirs, files in os.walk(pth):
    for fname in files:
        if fname.endswith('.csv') and fname.startswith('LID') and fname[5:7]=='40' :
            case=root+'/'+fname
            print(fname)

            x_head_l=dlcp.retrivedata(case,'headR')
            d_x_head_l=dlcp.smooth_diff(x_head_l)
            x_head_r=dlcp.retrivedata(case, 'headL')
            d_x_head_r=dlcp.smooth_diff(x_head_r)
            y_head_l=dlcp.retrivedata(case,'headR.1')
            d_y_head_l=dlcp.smooth_diff(y_head_l)
            y_head_r=dlcp.retrivedata(case, 'headL.1')
            d_y_head_r=dlcp.smooth_diff(y_head_r)
            tail_x=dlcp.retrivedata(case, 'tail')
            d_tail_x=dlcp.smooth_diff(tail_x)
            tail_y=dlcp.retrivedata(case, 'tail.1')
            d_tail_y=dlcp.smooth_diff(tail_y)


            ary=np.column_stack((tail_x,tail_y, x_head_l,x_head_r,y_head_l, y_head_r,
                                d_tail_x,d_tail_y, d_x_head_l,d_x_head_r,d_y_head_l, d_y_head_r))
            pca = PCA()
            x = StandardScaler().fit_transform(ary)
            principalComponents = pca.fit_transform(x)
            
            for i in range(2,15):
                n_com=i
                model1 = hmm.GaussianHMM(n_components=n_com, covariance_type="full", n_iter=4500)
                model1.fit(ary)
                label1=model1.predict(ary)
                sc_r=metrics.silhouette_score(ary, label1)
                print(sc_r)
                score.append((fname[3:5], sc_r, i, 'LID'))


scores=pd.DataFrame(score, columns=['case_name', 'score', 'n_clusters', 'treatment'])


# %% Multi Variable

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_palette(sns.color_palette(flatui))
ax=sns.lineplot(x='n_clusters', y='score',data=scores, ci=95)
plt.ylim(0,0.6)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
sns.despine()

#scores.to_csv('/home/morteza/dlc_projects/Analysis/Currencodes/12D_cluster_scores.csv')
#%%Correlation matrix for the 12D


x = StandardScaler().fit_transform(ary)
cov_o = EmpiricalCovariance().fit(x)
vect_cov=cov_o.covariance_ 

#Heat Map
sns.heatmap(vect_cov, cmap="viridis",vmin=-1, vmax=1)
plt.xlabel('Dimensions')
plt.ylabel('Dimensions')


#%% take the PCA of the 12D space




pca = PCA()
x = StandardScaler().fit_transform(ary)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2',
             'principal component 3', 'principal component 4','principal component 5', 'principal component 6',
                'principal component 7', 'principal component 8',
             'principal component 9', 'principal component 10','principal component 11', 'principal component 12' ])
             #,'principal component 7', 'principal component 8',
              #              'principal component 9', 'principal component 10','principal component 11', 'principal component 12','principal component 13', 'principal component 14','principal component 15', 'principal component 16'])


#y= StandardScaler().fit_transform(principalDf)
cov_o = EmpiricalCovariance().fit(principalDf)
vect_cov=cov_o.covariance_ 


#Heat Map
sns.heatmap(vect_cov, cmap="viridis",vmin=-0.1, vmax=1)
plt.xlabel('Principal Component')
plt.ylabel('Principal Component')



#%% Posture clustering


#pth='/home/morteza/dlc_projects/video_rep/Main/LID'
pth='/home/morteza/dlc_projects/video_rep/Additional'


score=[]
for root, dirs, files in os.walk(pth):
    for fname in files:
        if fname.endswith('.csv') and fname.startswith('SHM'):# and fname[5:7]=='40' :
            case=root+'/'+fname
            print(fname)
            m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2
            m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
            tail_x=dlcp.retrivedata(case, 'tail')
            tail_y=dlcp.retrivedata(case, 'tail.1')

            #posture
            body_ar=dlcp.ar(tail_x, tail_y, m_head_x, m_head_y)

            body_ar=body_ar/np.quantile(body_ar,0.95)
            head_ang=dlcp.thet_head(dlcp.retrivedata(case,'tail'),dlcp.retrivedata(case,'tail.1'),
                m_head_x,m_head_y,dlcp.retrivedata(case,'headR'),dlcp.retrivedata(case,'headR.1'),
                dlcp.retrivedata(case,'headL'),dlcp.retrivedata(case,'headL.1'))
            


            #movement
            ang_head=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
            d_ang=-dlcp.smooth_diff(ang_head)*50
            

            features=np.array([head_ang,d_ang,body_ar]).swapaxes(1,0)

            for i in range(2,15):
                n_com=i
                model1 = hmm.GaussianHMM(n_components=n_com, covariance_type="full", n_iter=4500)
                model1.fit(features)
                label1=model1.predict(features)
                sc_r=metrics.silhouette_score(features, label1)
                print(sc_r)
                score.append((fname[3:5], sc_r, i, 'LID'))


scores=pd.DataFrame(score, columns=['case_name', 'score', 'n_clusters', 'treatment'])




# %%
flatui = [ "#e74c3c", "#34495e", "#2ecc71"] #"#9b59b6", "#3498db", "#95a5a6",
sns.set_palette(sns.color_palette(flatui))
ax=sns.lineplot(x='n_clusters', y='score',data=scores)
plt.ylim(-0.1,0.6)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
sns.despine()


#scores.to_csv('/home/morteza/dlc_projects/Analysis/Currencodes/3param_cluster_scores.csv')



#%% K-means features, capable of reading tweets.


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
scaler = StandardScaler()
pth='/home/morteza/dlc_projects/video_rep/Main/SKF'



score=[]
for root, dirs, files in os.walk(pth):
    for fname in files:
        if fname.endswith('.csv') and fname.startswith('SKF') and fname[5:7]=='40' :
            case=root+'/'+fname
            print(fname)


            m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2
            m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
            tail_x=dlcp.retrivedata(case, 'tail')
            tail_y=dlcp.retrivedata(case, 'tail.1')
            body_ar=dlcp.ar(tail_x, tail_y, m_head_x, m_head_y)
            body_ar=body_ar/np.mean(body_ar)
            ang_head=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
            d_ang=dlcp.smooth_diff(ang_head)

            head_ang=dlcp.thet_head(dlcp.retrivedata(case,'tail'),dlcp.retrivedata(case,'tail.1'),
                m_head_x,m_head_y,dlcp.retrivedata(case,'headR'),dlcp.retrivedata(case,'headR.1'),
                dlcp.retrivedata(case,'headL'),dlcp.retrivedata(case,'headL.1'))

            features=np.array([body_ar,head_ang]).swapaxes(1,0)
            scaled_features = scaler.fit_transform(features)
           
            for i in range(2,15):
                kmeans = KMeans(n_clusters=i, random_state=0).fit(features)
                sc_r=metrics.silhouette_score(features, kmeans.labels_)
                score.append((fname[3:5], sc_r, i, 'LID'))





#scores=pd.DataFrame(score, columns=['case_name', 'score', 'n_clusters', 'treatment'])

# %%
flatui = ["#3498db"]#"#9b59b6", "#3498db", "#95a5a6", "#e74c3c",
sns.set_palette(sns.color_palette(flatui))


ax=sns.lineplot(x='n_clusters', y='score',data=scores)
plt.ylim(0,0.6)



plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')



plt.xlabel('Time From Drug Treatment (min)', fontsize='large')
plt.ylabel('Mean Body Length (Norm.)', fontsize='large')

sns.despine()


#%%
#scores.to_csv('/home/morteza/dlc_projects/Analysis/Currencodes/3D_Posture_clustering_scores.csv')



#%% Multi variable clustering PCA



pth='/home/morteza/dlc_projects/video_rep/Main/LID'



score=[]
for root, dirs, files in os.walk(pth):
    for fname in files:
        if fname.endswith('.csv') and fname.startswith('LID') and fname[5:7]=='20' :
            case=root+'/'+fname
            print(fname)

            x_head_l=dlcp.retrivedata(case,'headR')
            d_x_head_l=dlcp.smooth_diff(x_head_l)
            x_head_r=dlcp.retrivedata(case, 'headL')
            d_x_head_r=dlcp.smooth_diff(x_head_r)
            y_head_l=dlcp.retrivedata(case,'headR.1')
            d_y_head_l=dlcp.smooth_diff(y_head_l)
            y_head_r=dlcp.retrivedata(case, 'headL.1')
            d_y_head_r=dlcp.smooth_diff(y_head_r)
            tail_x=dlcp.retrivedata(case, 'tail')
            d_tail_x=dlcp.smooth_diff(tail_x)
            tail_y=dlcp.retrivedata(case, 'tail.1')
            d_tail_y=dlcp.smooth_diff(tail_y)


            ary=np.column_stack((tail_x,tail_y, x_head_l,x_head_r,y_head_l, y_head_r,
                                d_tail_x,d_tail_y, d_x_head_l,d_x_head_r,d_y_head_l, d_y_head_r))
            pca = PCA()
            x = StandardScaler().fit_transform(ary)
            principalComponents = pca.fit_transform(x)
            for i in range(2,15):
                n_com=i
                model1 = hmm.GaussianHMM(n_components=n_com, covariance_type="full", n_iter=4500)
                model1.fit(ary)
                label1=model1.predict(ary)
                sc_r=metrics.silhouette_score(ary, label1)
                print(sc_r)
                score.append((fname[3:5], sc_r, i, 'LID'))


scores=pd.DataFrame(score, columns=['case_name', 'score', 'n_clusters', 'treatment'])

#%%
flatui = ["#3498db"]#"#9b59b6", "#3498db", "#95a5a6", "#e74c3c",
sns.set_palette(sns.color_palette(flatui))


ax=sns.lineplot(x='n_clusters', y='score',data=scores)
plt.ylim(0,0.6)



plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')


plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
sns.despine()

#%%


#scores.to_csv('12DPCA_clusteringscore.csv')

# %% TSNE clustering 
from sklearn.manifold import TSNE



pth='/home/morteza/dlc_projects/video_rep/Main/LID'

model = TSNE(learning_rate=100)

score=[]
for root, dirs, files in os.walk(pth):
    for fname in files:
        if fname.endswith('.csv') and fname.startswith('LID') and fname[5:7]=='40' :
            case=root+'/'+fname
            print(fname)


            m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2
            m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
            tail_x=dlcp.retrivedata(case, 'tail')
            tail_y=dlcp.retrivedata(case, 'tail.1')
            body_ar=dlcp.ar(tail_x, tail_y, m_head_x, m_head_y)
            body_ar=body_ar/np.mean(body_ar)
            ang_head=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
            d_ang=dlcp.smooth_diff(ang_head)*50

            head_ang=dlcp.thet_head(dlcp.retrivedata(case,'tail'),dlcp.retrivedata(case,'tail.1'),
                m_head_x,m_head_y,dlcp.retrivedata(case,'headR'),dlcp.retrivedata(case,'headR.1'),
                dlcp.retrivedata(case,'headL'),dlcp.retrivedata(case,'headL.1'))

            features=np.array([body_ar,head_ang]).swapaxes(1,0)
            transformed_features = model.fit_transform(features)

            for i in range(2,15):
                n_com=i
                kmeans = KMeans(n_clusters=i, random_state=0).fit(principalComponents)
                sc_r=metrics.silhouette_score(principalComponents, kmeans.labels_)
                score.append((fname[3:5], sc_r, i, 'LID'))






scores=pd.DataFrame(score, columns=['case_name', 'score', 'n_clusters', 'treatment'])

# %%
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_palette(sns.color_palette(flatui))
ax=sns.lineplot(x='n_clusters', y='score',data=scores)
plt.ylim(0,0.6)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
sns.despine()



# %%
pth='/home/morteza/dlc_projects/video_rep/Main'



score=[]
for root, dirs, files in os.walk(pth):
    for fname in files:
        if fname.endswith('.csv') and fname.startswith('SKF') and fname[5:7]=='40' :
            case=root+'/'+fname

            m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2
            m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
            tail_x=dlcp.retrivedata(case, 'tail')
            tail_y=dlcp.retrivedata(case, 'tail.1')

            head_ang=dlcp.thet_head(dlcp.retrivedata(case,'tail'),dlcp.retrivedata(case,'tail.1'),
                m_head_x,m_head_y,dlcp.retrivedata(case,'headR'),dlcp.retrivedata(case,'headR.1'),
                dlcp.retrivedata(case,'headL'),dlcp.retrivedata(case,'headL.1'))

            head_ang2=head_ang.reshape(-1,1)
            #d_head=dlcp.smooth_diff(head_ang)*50



            #features=np.array([head_ang,d_head]).swapaxes(1,0)

            for i in range(2,15):
                n_com=i
                model1 = hmm.GaussianHMM(n_components=n_com)#, covariance_type="full", n_iter=4500)
                model1.fit(head_ang2)
                label1=model1.predict(head_ang2)
                sc_r=metrics.silhouette_score(head_ang2, label1)
                score.append((fname[3:5], sc_r, i, 'LID'))


scores=pd.DataFrame(score, columns=['case_name', 'score', 'n_clusters', 'treatment'])

# %%


flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.set_palette(sns.color_palette(flatui))
ax=sns.lineplot(x='n_clusters', y='score',data=scores)
#plt.ylim(0.5,0.8)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
sns.despine()






# %% posture d_reduction



pth='/home/morteza/dlc_projects/video_rep/Main/LID'



score=[]
for root, dirs, files in os.walk(pth):
    for fname in files:
        if fname.endswith('.csv') and fname.startswith('LID') and fname[5:7]=='20' :
            case=root+'/'+fname
            print(fname)

            m_head_x=(dlcp.retrivedata(case,'headR')+dlcp.retrivedata(case, 'headL'))/2
            m_head_y=(dlcp.retrivedata(case, 'headR.1')+dlcp.retrivedata(case, 'headL.1'))/2
            tail_x=dlcp.retrivedata(case, 'tail')
            tail_y=dlcp.retrivedata(case, 'tail.1')

            #posture
            body_ar=dlcp.ar(tail_x, tail_y, m_head_x, m_head_y)

            body_ar=body_ar/np.quantile(body_ar,0.95)
            head_ang=dlcp.thet_head(dlcp.retrivedata(case,'tail'),dlcp.retrivedata(case,'tail.1'),
                m_head_x,m_head_y,dlcp.retrivedata(case,'headR'),dlcp.retrivedata(case,'headR.1'),
                dlcp.retrivedata(case,'headL'),dlcp.retrivedata(case,'headL.1'))
            


            #movement
            ang_head=np.array(dlcp.angs(tail_x, tail_y, m_head_x, m_head_y))
            d_ang=-dlcp.smooth_diff(ang_head)*50
            features=np.array([body_ar,head_ang]).swapaxes(1,0)



            pca = PCA()
            x = StandardScaler().fit_transform(features)
            principalComponents = pca.fit_transform(x)
            for i in range(2,15):
                n_com=i
                model1 = hmm.GaussianHMM(n_components=n_com, covariance_type="full", n_iter=4500)
                model1.fit(principalComponents)
                label1=model1.predict(principalComponents)
                sc_r=metrics.silhouette_score(principalComponents, label1)
                print(sc_r)
                score.append((fname[3:5], sc_r, i, 'LID'))


scores=pd.DataFrame(score, columns=['case_name', 'score', 'n_clusters', 'treatment'])

#%%
flatui = ["#3498db"]#"#9b59b6", "#3498db", "#95a5a6", "#e74c3c",
sns.set_palette(sns.color_palette(flatui))


ax=sns.lineplot(x='n_clusters', y='score',data=scores)
plt.ylim(0,0.6)



plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')


plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
sns.despine()
