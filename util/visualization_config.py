
#a class that stores visualization settings and functions for matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

class visual_config():
    def __init__(self):
        
        #colors
        pal=sns.color_palette("tab10")
        color_list=pal.as_hex()

        self.treatment_colors={
                                'LD-3':color_list[0],
                                'D1Ag':color_list[1],
                                'D2Ag':color_list[2],
                                'D1Ant':color_list[3],
                                'D2Ant':color_list[4],
                                'D2KO':color_list[5]    }

        self.gradient_colors=sns.color_palette("RdPu", 10)

                                #palette=palette


        #fonts
        self.label_font = {  'weight' : 'normal',
                                'size'   : 12           }

        self.title_font = {     'weight' : 'bold',
                                'size'   : 14           }

        self.tick_font = {     'weight' : 'normal',
                                'size'   : 9            }


        self.legend={
                                'fontdict':self.tick_font,
                                'title':None,
                                'fancybox':True, 
                                'edgecolor':None, 
                                'frameon':False
                                                         }
                                

        pass



    def kinematic_box(self, ax, xlabel, ylabel, fonting, legend=False):
        

        with plt.rc('font', **fonting):
            ax.set_xlim(-0.5, 8.5)
            
            ax.set_xlabel(xlabel, fontsize=fonting.ax_size)
            ax.set_ylabel(ylabel, fontsize=fonting.ax_size)


            sns.despine(ax=ax)
            
            if isinstance(legend, dict):
                plt.legend(**legend)
            else:
                ax.legend([],[], frameon=False)
            
        return ax

