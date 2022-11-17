
#a class that stores visualization settings and functions for matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

class visual_config():
    def __init__(self):
        
        #colors
        pal=sns.color_palette("tab10")
        color_list=pal.as_hex()

        self.treatment_colors={
                                'LD-3mg':color_list[0],
                                'D1Ag':color_list[1],
                                'D2Ag':color_list[2],
                                'D1Ant':color_list[3],
                                'D2Ant':color_list[4],
                                'D2KO':color_list[5]    }

        self.gradient_colors=sns.color_palette("RdPu", 10)

                                #palette=palette


        #fonts
        self.label_font = {  'weight' : 'normal',
                                'size'   : 14           }

        self.title_font = {     'weight' : 'bold',
                                'size'   : 16           }

        self.tick_font = {     'weight' : 'normal',
                                'size'   : 10            }


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