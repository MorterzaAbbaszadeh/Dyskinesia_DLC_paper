import numpy as np




class dlc_iter:

    def __init__(self):

        self.body_map   =   {  

                'x_rhead':'headR',
                'y_rhead':'headR.1',
                'x_lhead':'headL',
                'y_lhead':'headL.1',
                'x_tail':'tail',
                'y_tail':'tail.1',

                }

        self.treatments =   ['LID', 'SUM', 'SKF', 'D1A', 'D2A', 'D2K']



    '''fetch precise dict keys from NPY database'''


    #datat base: animal_ID/treatment,
    #                       traces/time points/-xhand
    #                                           -yhand
    #                                          -etc 
    #work flow: get_treatment >for each animal in treatment >iterate a functionon over the trace dict


    #this reads the specific treatment types from the dataset     
    def get_treatments(self, file):

        with open(file, 'rb') as f:
            dct=np.load(f, allow_pickle=True).item()

        for ky in list(dct.keys()):
            if not dct[ky]['treatment'] in self.treatments:
                dct.pop(ky)

        return dct


    #seaborn
    #iterates over the recording time_points of each specific animal
    def time_iter(self, dct, funcs):

        characters=[]


        for ky in dct.keys():
            

            self.ar0=dct[ky]['ar0']

            dlc_dct=dct[ky]['traces']
            
            kys=list(dct[ky]['traces'].keys())

            treatment=dct[ky]['treatment']
            

            i=0
            while i < len(kys):

                characters.append([np.mean(f(dlc_dct[kys[int(i)]])) for f in funcs]+
                                                                [kys[int(i)], treatment, ky]) #could add self.ar0 here
                i+=1
    
        return characters



    def func_iter(self, dct:'dct of animals', funcs , t_point):

        characters=[f(dct['traces'][str(t_point)]) for f in funcs]
    
        return characters



    def get_singleanimal_ts(self, file:'full dct', groups, animal_id, funcs):
        self.treatments=groups

        return





    '''
                                Get functions

    '''

    #get it for seaborn
    def get_sns(self, file, groups, funcs):
        self.treatments=groups
        return self.time_iter(self.get_treatments(file), funcs)



    def get_ts(self, file, groups, funcs):
        self.treatments=groups
        return self.func_iter(self.get_treatments(file), funcs)



    def get_singleanimal_ts(self, file:'full dct', groups, animal_id, funcs):
        self.treatments=groups

        return





if __name__=='__main__':
    
    dlc_iter()