

import sys
sys.path.insert(1, r'C:\dlc_projects\Analysis\Currencodes\DLC_refact')
import numpy as np
from kinematics import dlc_kinematics


kinematics=dlc_kinematics()



class dlc_ts:

    def __init__(self):

        self.body_map   =   {  

                'x_rhead':'headR',
                'y_rhead':'headR.1',
                'x_lhead':'headL',
                'y_lhead':'headL.1',
                'x_tail':'tail',
                'y_tail':'tail.1',

                }

        self.treatments =['LID', 'SUM', 'SKF', 'D1A', 'D2A', 'D2K']



    '''fetch precise dict keys from NPY database'''


    # datat base: animal_ID/treatment,
    #                       traces/time points/-xhand
    #                                           -yhand
    #                                          -etc 
    # work flow: get_treatment > for a specific time point (t_p) and animal index in the database (ID_n)


    #this reads the specific treatment types from the dataset     

    def get_treatments(self, file):

        with open(file, 'rb') as f:
            dct=np.load(f, allow_pickle=True).item()

        for ky in list(dct.keys()):
            if not dct[ky]['treatment'] in self.treatments:
                dct.pop(ky)

        return dct


    
    
    def time_srs(self, dct, ID_n:'the animal idx', t_p:'the time point') -> 'thet_head, ar, rot_speed, translate':

        ky=list(dct.keys())[ID_n]                #select the animal

        self.ar0=dct[ky]['ar0']

        dlc_dct=dct[ky]['traces'][str(t_p)]     #get the x,y data set
        treatment=dct[ky]['treatment']

        #calculate the outputs
        thet_head=kinematics.npy_thet_head(dlc_dct)
        ar=kinematics.npy_ar( dlc_dct)/self.ar0
        rot_speed=kinematics.npy_rot_speed(dlc_dct)
        translate=kinematics.npy_translation(dlc_dct)



        return thet_head, ar, rot_speed, translate
    



if __name__=='__main__':
    
    dlc_ts()