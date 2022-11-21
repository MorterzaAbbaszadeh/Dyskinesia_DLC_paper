
#a class that stores visualization settings and functions for matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np

class visual_config():
    def __init__(self):
        
        #colors
        pal=sns.color_palette("tab10")
        self.treatment_color_list=pal.as_hex()

        self.treatment_colors={
                                'LD-3mg':self.treatment_color_list[0],
                                'D1Ag':self.treatment_color_list[1],
                                'D2Ag':self.treatment_color_list[2],
                                'D1Ant':self.treatment_color_list[3],
                                'D2Ant':self.treatment_color_list[4],
                                'D2KO':self.treatment_color_list[5]    }

        self.gradient_colors=sns.color_palette("RdPu", 10)


        self.compare_colors_sr = ["#9b59b6", "#3498db",
                    "#95a5a6", "#e74c3c", 
                        "#34495e", "#2ecc71"]
        self.compare_colors=sns.set_palette(sns.color_palette(self.compare_colors_sr))

                                #palette=palette

        self.heatmap_cmp= sns.cubehelix_palette(light=1.05, as_cmap=True)
       
       
        #fonts
        self.label_font = {  'weight' : 'normal',
                                'size'   : 14           }

        self.title_font = {     'weight' : 'bold',
                                'size'   : 16           }

        self.tick_font = {     'weight' : 'normal',
                                'size'   : 10            }

        self.heatmap_font= 8



        self.legend={
                                'fontsize':10,
                                'title':None,
                                'fancybox':True, 
                                'edgecolor':None, 
                                'frameon':False
                                                         }
                                

        pass



    def kinematic_box(self, ax, xlabel, ylabel, legend=False):
        

        ax.set_xlim(-0.5, 8.5)
        
        ax.set_xlabel(xlabel, fontdict=self.label_font)
        ax.set_ylabel(ylabel, fontdict=self.label_font)


        sns.despine(ax=ax)
        
        if isinstance(legend, dict):
            plt.legend(**legend)
        else:
            ax.legend([],[], frameon=False, fontsize=self.tick_font['size'])

         #make this local

            
        return ax

    def hmm_plots(self, ax, xlabel, ylabel, legend=False):
        

        ax.set_xlim(-0.5, 8.5)
        
        ax.set_xlabel(xlabel, fontdict=self.label_font)
        ax.set_ylabel(ylabel, fontdict=self.label_font)


        sns.despine(ax=ax)
        
        if isinstance(legend, dict):
            plt.legend(**legend)
        else:
            ax.legend([],[], frameon=False, fontsize=self.tick_font['size'])

         #make this local

            
        return ax

    def barplots(self, ax, xlabel, ylabel, legend=False):
    

        ax.set_xlim(-0.5, 8.5)
        
        ax.set_xlabel(xlabel, fontdict=self.label_font)
        ax.set_ylabel(ylabel, fontdict=self.label_font)


        sns.despine(ax=ax)
        
        if isinstance(legend, dict):
            plt.legend(**legend)
        else:
            ax.legend([],[], frameon=False, fontsize=self.tick_font['size'])

            #make this local

            
            return ax

    #messy with alot of constants
    def head_polar_plot(self,ax, head_ang, ar,lower_cut, upper_cut, treatment, alph=0.1 ):
        
        
        
        ax.plot( head_ang[lower_cut:upper_cut],ar[lower_cut:upper_cut],
                            
                                ls='', marker='o',color=self.treatment_colors[treatment], alpha=alph)


        ax.grid(linewidth=1, ls='--')

        ax.set_xticklabels(['270', '', '0', '', '90', '', '180', ''])
        ax.set_ylim(0,1.5) #normalized body
   
        ax.set_rticks([0, 0.5, 1, 1.5])
       
        ax.spines['polar'].set_visible(False)


        return ax


    def presence_heat_maps(self, axs,cent_x,cent_y, lower_cut, upper_cut ):



        cbar_kws = {'format':'%.1e'}


        axs.plot(cent_x[lower_cut:upper_cut],cent_y[lower_cut:upper_cut],linewidth=0.3, color='k', alpha=0.5)#,cumulative=True,
    
        arena=plt.Circle((320, 320),320, fill=False, linestyle='--', linewidth=0.5)
        axs.add_patch(arena)
        axs.set_xlim(0, 640)
        axs.set_ylim(0, 640)
        sns.despine(top=True, bottom=True, left=True, right=True)
        axs.set_xticks([],[])
        axs.set_yticks([],[])
        axs.set_aspect(1)

        sns.kdeplot(cent_x[lower_cut:upper_cut],cent_y[lower_cut:upper_cut],shade=True, 
                    cmap=self.heatmap_cmp, cbar=True, cbar_kws = cbar_kws, ax=axs)#,cumulative=True,
       
        return axs
    
    def sns_plots(self, ax, xlabel, ylabel, legend=False):
    
        
        
        ax.set_xlabel(xlabel, fontdict=self.label_font)
        ax.set_ylabel(ylabel, fontdict=self.label_font)


        sns.despine(ax=ax)
        
        if isinstance(legend, dict):
            plt.legend(**legend)
        else:
            ax.legend([],[], frameon=False, fontsize=self.tick_font['size'])

         #make this local

            
        return ax
    
    def visualize_ts_state(self, ax, dct,  ID_n, time_point,strt, fini, fetch, model, cut, fps):


        
        head_ang, ar, rot_speed, translate=fetch.time_srs(dct, ID_n,  time_point)
        features=np.vstack(([ar[:cut], rot_speed[:cut], head_ang[:cut]])).swapaxes(1,0)
        labels=model.predict(features)

        time=np.linspace(0,cut, cut)/fps



        ar=features[:,0]
        rot_speed=features[:,1]
        head_ang=features[:,2]

        label_loc=[-0.08, 0.5]

        ax[0].plot(time, head_ang, linewidth=0.5)
        ax[0].scatter(time[labels==0], head_ang[labels==0], color=self.compare_colors_sr[0])
        ax[0].scatter(time[labels==1], head_ang[labels==1], color=self.compare_colors_sr[1])
        ax[0].set_ylabel(r'$\phi $', fontdict=self.label_font)
        ax[0].yaxis.set_label_coords(label_loc[0], label_loc[1])
        ax[0].set_xlim(strt, fini)
        ax[0].tick_params(bottom=False, top=False, left=True, right=False)
        ax[0].set_xticklabels([])
        sns.despine(top=True, bottom=True, right=True, ax=ax[0])


        ax[1].plot(time, ar, linewidth=0.5)
        ax[1].scatter(time[labels==0], ar[labels==0],color=self.compare_colors_sr[0])
        ax[1].scatter(time[labels==1], ar[labels==1],color=self.compare_colors_sr[1])
        ax[1].set_ylabel(r'R', fontdict=self.label_font)
        ax[1].yaxis.set_label_coords(label_loc[0], label_loc[1])
        ax[1].set_xlim(strt, fini)
        ax[1].tick_params(bottom=False, top=False, left=True, right=False)
        ax[1].set_xticklabels([])
        sns.despine(top=True, bottom=True, right=True, ax=ax[1])


        ax[2].plot(time, rot_speed, linewidth=0.5)
        ax[2].scatter(time[labels==0], rot_speed[labels==0], color=self.compare_colors_sr[0])
        ax[2].scatter(time[labels==1], rot_speed[labels==1], color=self.compare_colors_sr[1])
        ax[2].set_ylabel(r'd$\theta$/dt', fontdict=self.label_font)
        ax[2].yaxis.set_label_coords(label_loc[0], label_loc[1])
        ax[2].set_xlim(strt, fini)
        ax[2].set_xlabel('Time (sec)', fontdict=self.label_font)
        sns.despine(top=True, right=True, ax=ax[2])


        

        return ax

    
    def cluster_bar_plots(self,ax, characters, params, n=7):
        for i in range(len(ax)):
            param=list(params.keys())[i]
            sns.barplot(x='Type', y=param,data=characters, ax=ax[i], ci=95/np.sqrt(n))
            ax[i].set_xlim(-1, 2)
            ax[i].set_xticklabels(['in', 'out'],  rotation = 45, fontdict=self.label_font)
            ax[i].set_ylabel(params[param], fontdict=self.label_font)
            ax[i].set_xlabel('')
            if i==2: #specific to cluster vis
                ax[i].plot([-1, 2], [0, 0], color='k', linestyle='--', alpha=0.5)
            
            sns.despine(right=True, top=True, ax=ax[i])
            plt.tight_layout()

        return ax
