import numpy as np
import scipy.signal as sgn



class dlc_kinematics:
    

    def __init__(self):

        self.sg_window=45
        self.sg_order=4
        self.pad=50
        
        self.body_map   =   {  

                'x_rhead':'headR',
                'y_rhead':'headR.1',
                'x_lhead':'headL',
                'y_lhead':'headL.1',
                'x_tail':'tail',
                'y_tail':'tail.1',

                }

        



    def angs(self, x1,y1,x2,y2): #rendering vectro angles from 0 to 360

        theta=np.zeros(len(x1))
        

        i=0
        while i<len(x1):

            if y2[i]-y1[i]>0:
                theta[i]=np.arccos((x2[i]-x1[i])/(np.sqrt((x2[i]-x1[i])**2+(y2[i]-y1[i])**2)))*57.32

            elif y2[i]-y1[i]<0:                     # if the vector falls in third or forth quarters add 180 degs to result
                    theta[i]=180+(np.arccos((x1[i]-x2[i])/(np.sqrt((x2[i]-x1[i])**2+(y2[i]-y1[i])**2)))*57.32)
            i+=1


        return np.concatenate((theta, theta[-1:])) #compensates for size difference in differentiation



    def ar(self, x1,y1,x2,y2):

        ar=np.sqrt(np.square(x2-x1)+np.square(y2-y1))

        ar=np.pad(ar, 50, 'edge')
        
        return sgn.savgol_filter(ar, 45, 4)[50:-50] 
    



    def translation(self, x, y):                                #calculate the euclidian distance between positions of a point in two frames. 
       
       
        stp=np.zeros(len(x))

        i=0
        while i<len(x)-1:
            stp[i]=np.sqrt(np.square(x[i+1]-x[i])+np.square(y[i+1]-y[i]))
            i+=1

        #stp=np.concatenate((stp, stp[-1]))

        stp=np.pad(stp, 50, 'edge')                #pad the signal by its edge
        

        return sgn.savgol_filter(stp, 45, 4)[50:-50]




    def tot_dist(self, x, y):                                #calculate the euclidian distance between positions of a point in two frames. 

        return self.translation(x, y).sum()


    def pos_rots(self, x1,y1,x2,y2):
    
        angs=self.npy_rot_speed(x1,y1,x2,y2)


        pos_rot=0
        neg_rot=0

        c_ang=0


        for ang in angs[:3600]:
            c_ang=c_ang+ang
            if c_ang >= 365:
                pos_rot=pos_rot+1
                c_ang=0
            elif c_ang <= -365:
                neg_rot=neg_rot+1
                c_ang=0
                

        return (pos_rot/len(angs), neg_rot/len(angs))




    def thet_head(self, x1,y1,x2,y2,x3,y3,x4,y4):         #1:tail, 2:mid_head, 3 HeadR, 4:HeadL
        
        
        u=np.array([x2-x1, y2-y1])                  #main body vector
        v=np.array([x4-x3, y4-y3])                  #head vector


        dotp=u[0]*v[0]+u[1]*v[1]                    #dot product


        si_u=np.sqrt(u[0]**2+u[1]**2)
        si_v=np.sqrt(v[0]**2+v[1]**2)



        thet_head=np.arccos(abs(dotp/(si_u*si_v)))*57.32   #(abs(dotp/(si_u*si_v)))*57.32 
        sm_head=np.pad(thet_head, 50, 'edge')                #pad the signal by its edge
           
        return 90-sgn.savgol_filter(sm_head, 45, 4)[50:-50]




    def smooth_diff(self, theta): 

        diff=np.diff(theta)

        for i in range(0,len(diff)):
            if diff[i]>30 or diff[i]<-30 :
                diff[i]=diff[i-1]

        diff=np.append(diff,diff[-1])
        diff=np.pad(diff, 50, 'edge')                #pad the signal by its edge
        
        return sgn.savgol_filter(diff, 45, 4)[50:-50]  


    ''' auxilary functions to interface the NPY database  '''



    #dct is dct[trace][trace][t_p]
    def npy_angs(self, dct:'dct[trace][t_p]'):

        x1=(dct[self.body_map['x_rhead']]+dct[self.body_map['x_lhead']])/2
        y1=(dct[self.body_map['y_rhead']]+dct[self.body_map['y_lhead']])/2

        x2=dct[self.body_map['x_tail']]
        y2=dct[self.body_map['y_tail']]

        return self.angs(x1,y1,x2,y2)
    

    def npy_ar(self, dct:'dct[trace][t_p]'):
        
        x1=(dct[self.body_map['x_rhead']]+dct[self.body_map['x_lhead']])/2
        y1=(dct[self.body_map['y_rhead']]+dct[self.body_map['y_lhead']])/2

        x2=dct[self.body_map['x_tail']]
        y2=dct[self.body_map['y_tail']]

        return self.ar(x1,y1,x2,y2)


    def npy_translation(self, dct:'dct[trace][t_p]'):


        x=(dct[self.body_map['x_rhead']]+dct[self.body_map['x_lhead']])/2
        y=(dct[self.body_map['y_rhead']]+dct[self.body_map['y_lhead']])/2



        return self.translation(x,y)
    

    def npy_thet_head(self, dct:'dct[trace][t_p]'):


        x1=dct[self.body_map['x_tail']]
        y1=dct[self.body_map['y_tail']]



        x3=dct[self.body_map['x_rhead']]
        y3=dct[self.body_map['y_rhead']]


        x4=dct[self.body_map['x_lhead']]
        y4=dct[self.body_map['y_lhead']]


        x2=(x3+x4)/2
        y2=(y3+y4)/2


        
        
        return self.thet_head(x1,y1,x2,y2,x3,y3,x4,y4)



    def npy_rot_speed(self, dct):
        return self.smooth_diff(self.npy_angs(dct))


    def get_3Dembd_train(self, dct:'full treat dct', ky:'animal ID', kin:'Kinm. class', t_point:'training time')->'HMM embedding at t_point':

        t_point=str(t_point)
        ar=self.npy_ar(dct[ky]['traces'][t_point])/dct[ky]['ar0']
        head_ang= self.npy_thet_head(dct[ky]['traces'][t_point])
        rot_speed=self.npy_rot_speed(dct[ky]['traces'][t_point])
        

        cut=min([len(ar), len(head_ang), len(rot_speed)])

        features=np.vstack(([ar[:cut], rot_speed[:cut], head_ang[:cut]])).swapaxes(1,0)
        
        return features




    #huh?
    def get_3Dembd(self, dct:'full treat dct', ky:'animal ID', kin:'Kinm. class')->'HMM embedding at t_point':

        
        ar=self.npy_ar(dct)/dct[ky]['ar0']
        head_ang= self.npy_thet_head(dct)
        rot_speed=self.npy_rot_speed(dct)
        

        cut=min([len(ar), len(head_ang), len(rot_speed)])

        features=np.vstack(([ar[:cut], rot_speed[:cut], head_ang[:cut]])).swapaxes(1,0)
        
        return features


    def get_4Dembd_train(self, dct:'full treat dct', ky:'animal ID', kin:'Kinm. class', t_point:'training time')->'HMM embedding at t_point':

        t_point=str(t_point)
        ar=self.npy_ar(dct[ky]['traces'][t_point])/dct[ky]['ar0']
        head_ang= self.npy_thet_head(dct[ky]['traces'][t_point])
        rot_speed=self.npy_rot_speed(dct[ky]['traces'][t_point])
        trans=self.npy_translation(dct[ky]['traces'][t_point])
        

        cut=min([len(ar), len(head_ang), len(rot_speed)])

        features=np.vstack(([ar[:cut], rot_speed[:cut], head_ang[:cut], trans[:cut]])).swapaxes(1,0)
        
        return features




    
    
    



if __name__=='__main__':
    #dunno why
    dlc_kinematics()
