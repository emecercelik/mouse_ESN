# =================================================================================================================================================
#                                       Import modules
import pickle
import random
import numpy as np
from numpy import linalg as LA

def PickleIt(data,fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def GetPickle(fileName):
    with open(fileName, 'rb') as f:
        data = pickle.load(f)
    return data

def ActFunc(n,maxim,minim,value,off):
    return np.exp((-n/(3*(maxim-minim)))*(value-off)**2)

def NeuronFunc(x):
    return (1.+np.tanh(x-1.6))*.5 # 4.=9 input

def InputFunc(x,dim,nAct,maxAcc,minAcc):
    rangeAcc=maxAcc-minAcc
    off=np.arange(minAcc+rangeAcc/(2.*nAct),maxAcc,rangeAcc/nAct)
    res=np.array([[ActFunc(nAct,maxAcc,minAcc,x[j],off[i]) for i in range(len(off))] for j in range(len(x))])
    res=res.reshape(res.size,1)
    for i in range(res.size):
        if res[i]<1e-2:
            res[i]=.0
    return res


bpy.context.scene.game_settings.fps=50.
dt=1000./bpy.context.scene.game_settings.fps

# =================================================================================================================================================
#                                       Creating muscles

FF=1.5
FF2=.05
muscle_ids = {}
[ muscle_ids["wrist.L_FLEX"], muscle_ids["wrist.L_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_wrist.L",  attached_object_name = "obj_forearm.L",  maxF = FF2)
[ muscle_ids["wrist.R_FLEX"], muscle_ids["wrist.R_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_wrist.R",  attached_object_name = "obj_forearm.R",  maxF = FF2)
[ muscle_ids["forearm.L_FLEX"], muscle_ids["forearm.L_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_forearm.L",  attached_object_name = "obj_upper_arm.L",  maxF = FF)
[ muscle_ids["forearm.R_FLEX"], muscle_ids["forearm.R_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_forearm.R",  attached_object_name = "obj_upper_arm.R",  maxF = FF)

[ muscle_ids["upper_arm.L_FLEX"], muscle_ids["upper_arm.L_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_upper_arm.L",  attached_object_name = "obj_shoulder.L",  maxF = FF)
[ muscle_ids["upper_arm.R_FLEX"], muscle_ids["upper_arm.R_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_upper_arm.R",  attached_object_name = "obj_shoulder.R",  maxF = FF)
[ muscle_ids["shin_lower.L_FLEX"], muscle_ids["shin_lower.L_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_shin_lower.L",  attached_object_name = "obj_shin.L",  maxF = FF2)
[ muscle_ids["shin_lower.R_FLEX"], muscle_ids["shin_lower.R_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_shin_lower.R",  attached_object_name = "obj_shin.R",  maxF = FF2)

[ muscle_ids["shin.L_FLEX"], muscle_ids["shin.L_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_shin.L",  attached_object_name = "obj_thigh.L",  maxF = FF)
[ muscle_ids["shin.R_FLEX"], muscle_ids["shin.R_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_shin.R",  attached_object_name = "obj_thigh.R",  maxF = FF)
[ muscle_ids["thigh.L_FLEX"], muscle_ids["thigh.L_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_thigh.L",  attached_object_name = "obj_hips",  maxF = FF2)
[ muscle_ids["thigh.R_FLEX"], muscle_ids["thigh.R_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_thigh.R",  attached_object_name = "obj_hips",  maxF = FF2)


# =================================================================================================================================================
#                                       Network creation

#np.random.seed(np.random.randint(0,10000))
dim=3 # input dimension
nAct=10 # Active point for each dimension
nInp=nAct*dim # Total input size
nRes=1000 # Reservoir element number
nOut=6 #Output number

wInp=np.random.rand(nRes,nInp+nOut)

wRes=2.*np.random.rand(nRes*nRes,1)
indInh=np.random.randint(0,wRes.size,size=wRes.size*.1)
wRes[indInh]*=-1.
wRes=wRes.reshape(nRes,nRes)
wRes*=1.25/max(abs((LA.eigvals(wRes))))

wOut=np.random.rand(nOut,nInp+nRes)
out=np.random.rand(nOut,1)
a=.3 # leak coeff 
x=np.random.rand(nRes,1)
u=np.random.rand(nInp,1)

eps=1.

Record_or_Test=0    #0: Record (Gather Data), 1: Test  

if Record_or_Test==0:
##    aa=GetPickle('paramStatic1')
##    wInp=aa[0]
##    wRes=aa[1]
    PickleIt([wInp,wRes],'paramStatic')
    Record=np.zeros((1,(nInp+nRes+nOut)))
elif Record_or_Test==1: # Test case
    aa=GetPickle('paramStatic1')
    wInp=aa[0]
    wRes=aa[1]
    wOut=GetPickle('wout')

ax_avg=0.1
ay_avg=0.1
az_avg=0.1

ax=.1
ay=.1
az=.1

out2=out
out3=out
z2=np.vstack((x,u))
z3=z2

# =================================================================================================================================================
#                                       Evolve function
def evolve():
    global x,wInp,u,wRes,wOut,a,nInp,nAct,nOut,counter,out,eps
    global out,out2,out3,z2,z3
    global ax,ay,az,ax_avg,ay_avg,az_avg
    global Record, Record_or_Test
    
    print("Step:", i_bl, "  Time:{0:.2f}".format(t_bl),'   Acc:{0:8.2f}  {1:8.2f}  {2:8.2f}'.format(ax,ay,az),\
          'Acc:{0:8.2f} {1:8.2f} {2:8.2f}'.format(ax_avg,ay_avg,az_avg))
    # ------------------------------------- Visual ------------------------------------------------------------------------------------------------
    #visual_array     = getVisual(camera_name = "Meye", max_dimensions = [256,256])
    #scipy.misc.imsave("test_"+('%05d' % (i_bl+1))+".png", visual_array)
    # ------------------------------------- Olfactory ---------------------------------------------------------------------------------------------
    olfactory_array  = getOlfactory(olfactory_object_name = "obj_nose", receptor_names = ["smell1", "plastic1"])
    # ------------------------------------- Taste -------------------------------------------------------------------------------------------------
    taste_array      = getTaste(    taste_object_name =     "obj_mouth", receptor_names = ["smell1", "plastic1"], distance_to_object = 1.0)
    # ------------------------------------- Vestibular --------------------------------------------------------------------------------------------
    vestibular_array = getVestibular(vestibular_object_name = "obj_head")
    #print (vestibular_array)
    # ------------------------------------- Sensory -----------------------------------------------------------------------------------------------
    # ------------------------------------- Proprioception ----------------------------------------------------------------------------------------
    #~ spindle_FLEX = getMuscleSpindle(control_id = muscle_ids["forearm.L_FLEX"])
    #~ spindle_EXT  = getMuscleSpindle(control_id = muscle_ids["forearm.L_EXT"])
    # ------------------------------------- Neural Simulation -------------------------------------------------------------------------------------

    ax=vestibular_array[3]
    ay=vestibular_array[4]
    az=vestibular_array[5]

    if Record_or_Test==0: # Record Case
        ax_avg=(ax_avg*(i_bl)+ax)/(i_bl+1)
        ay_avg=(ay_avg*(i_bl)+ay)/(i_bl+1)
        az_avg=(az_avg*(i_bl)+az)/(i_bl+1)
        z4=z3
        z3=z2
        z2=np.vstack((x,u))

        out4=out3
        out3=out2
        out2=out
        if ay_avg<-1. and ax_avg<2. and ax_avg>-2.: # Record the outputs if acceleration is positive
            z=np.vstack((x,u))
            Record=np.vstack((Record,np.vstack((z,out)).T))
            #Record=np.vstack((Record,np.vstack((z2,out2)).T))
            #Record=np.vstack((Record,np.vstack((z3,out3)).T))
            #Record=np.vstack((Record,np.vstack((z4,out4)).T))
            #print(Record.shape,np.vstack((z,out)).T.shape)    
        if np.mod(i_bl,100)==0: # Save once in 1000 steps
            PickleIt(Record,'paramWalking')
    else:
        ax_avg=(ax_avg*(i_bl)+ax)/(i_bl+1)
        ay_avg=(ay_avg*(i_bl)+ay)/(i_bl+1)
        az_avg=(az_avg*(i_bl)+az)/(i_bl+1)


    x=(1-a)*x+a*NeuronFunc(wInp.dot(np.vstack((u,out)))+wRes.dot(x)) # States get output as input
    #x=(1-(a))*x+a*NeuronFunc(wInp.dot(u)+wRes.dot(x)) # States dont have outputs as inputs
    

    #input_array=[ax,ay,az,ax2,ay2,az2,ax3,ay3,az3]
    input_array1=[ax,ay,az]
    #input_array2=[sF1[0],sF1[1],sF2[0],sF2[1],sF3[0],sF3[1],sF4[0],sF4[1]] # Inputs are only muscle positions

    input_array1=InputFunc(input_array1,len(input_array1),nAct,100.,-100.)
    #input_array2=InputFunc(input_array2,len(input_array2),nAct,2.,-2.) # Convert muscle positions to inputs of the ESN
    
    u=input_array1

    u=u.reshape(nInp,1) # Reshape inputs (n,1)
    x=x.reshape(nRes,1)
    z=np.vstack((x,u))

    if Record_or_Test==1: # Test case with using Regressed wOut
        out=wOut.dot(z) # Calculate outputs with using wOut
        #out=out/np.amax(out) # This scale needed not to diverge ???
        out.reshape(nOut,1)
    elif Record_or_Test==0: # Record case with random outputs
        out=(np.random.rand(nOut,1)-.5)*2
            
    
    
    
    # ------------------------------------- Muscle Activation -------------------------------------------------------------------------------------

    speed_ = 20.0


    
    act_tmp         = 0.5 + 0.5*np.sin(speed_*t_bl)
    anti_act_tmp    = 1.0 - act_tmp
    act_tmp_p1      = 0.5 + 0.5*np.sin(speed_*t_bl - np.pi*0.5)
    anti_act_tmp_p1 = 1.0 - act_tmp_p1
    act_tmp_p2      = 0.5 + 0.5*np.sin(speed_*t_bl + np.pi*0.5)
    anti_act_tmp_p2 = 1.0 - act_tmp_p2

    act1= 0.5 + 0.5*np.sin(out[1]*t_bl) #0.5 + 0.5*np.sin(speed_*t_bl+out[0])
    anti_act1=1.-act1
    act2= 0.5 + 0.5*np.sin(out[3]*t_bl) #0.5 + 0.5*np.sin(speed_*t_bl+out[1])
    anti_act2=1.-act2
    act3= 0.5 + 0.5*np.sin(out[5]*t_bl) #0.5 + 0.5*np.sin(speed_*t_bl+out[1])
    anti_act3=1.-0.8*act3

    act4= 0.5 + 0.5*np.sin(out[0]*t_bl) #0.5 + 0.5*np.sin(speed_*t_bl+out[0])
    anti_act4=1.-act1
    act5= 0.5 + 0.5*np.sin(out[2]*t_bl) #0.5 + 0.5*np.sin(speed_*t_bl+out[1])
    anti_act5=1.-act2
    act6= 0.5 + 0.5*np.sin(out[4]*t_bl) #0.5 + 0.5*np.sin(speed_*t_bl+out[1])
    anti_act6=1.-0.8*act3






    
    controlActivity(control_id = muscle_ids["wrist.L_FLEX"], control_activity = .4) #act4)# fL
    controlActivity(control_id = muscle_ids["wrist.L_EXT"] , control_activity = .6) #anti_act4)# fL
    controlActivity(control_id = muscle_ids["wrist.R_FLEX"], control_activity = .4)
    controlActivity(control_id = muscle_ids["wrist.R_EXT"] , control_activity = .6)
    controlActivity(control_id = muscle_ids["forearm.L_FLEX"], control_activity = 0.8*act_tmp)#act5)#fAL) #   #fL
    controlActivity(control_id = muscle_ids["forearm.L_EXT"] , control_activity = 1.-0.8*act_tmp) #act5)#anti_fAL) # #fL
    controlActivity(control_id = muscle_ids["forearm.R_FLEX"], control_activity = 0.8*anti_act_tmp) #
    controlActivity(control_id = muscle_ids["forearm.R_EXT"] , control_activity = 1.-0.8*anti_act_tmp) #
    controlActivity(control_id = muscle_ids["upper_arm.L_FLEX"], control_activity = 1.0*act_tmp_p1)#act6)#uAL) # #fL
    controlActivity(control_id = muscle_ids["upper_arm.L_EXT"] , control_activity = 1.-1.0*act_tmp_p1)#act6)#anti_uAL) # #fL
    controlActivity(control_id = muscle_ids["upper_arm.R_FLEX"], control_activity = 1.0*anti_act_tmp_p1) #
    controlActivity(control_id = muscle_ids["upper_arm.R_EXT"] , control_activity = 1.-1.0*anti_act_tmp_p1) #
    controlActivity(control_id = muscle_ids["shin_lower.L_FLEX"], control_activity = 0.8*act6)#0.8*anti_act_tmp)
    controlActivity(control_id = muscle_ids["shin_lower.L_EXT"] , control_activity = anti_act6)#1-0.8*anti_act_tmp)
    controlActivity(control_id = muscle_ids["shin_lower.R_FLEX"], control_activity = 0.8*act3)#0.8*act_tmp)#
    controlActivity(control_id = muscle_ids["shin_lower.R_EXT"] , control_activity = anti_act3)#1.-0.8*act_tmp)#
    controlActivity(control_id = muscle_ids["shin.L_FLEX"], control_activity = act5)#0.5*anti_act_tmp_p1)
    controlActivity(control_id = muscle_ids["shin.L_EXT"] , control_activity = anti_act5)#1.-0.5*anti_act_tmp_p1)
    controlActivity(control_id = muscle_ids["shin.R_FLEX"], control_activity = act2)#0.5*act_tmp_p1)#
    controlActivity(control_id = muscle_ids["shin.R_EXT"] , control_activity = anti_act2)#1.-0.5*act_tmp_p1)#
    controlActivity(control_id = muscle_ids["thigh.L_FLEX"], control_activity = act4)#0.5*anti_act_tmp)
    controlActivity(control_id = muscle_ids["thigh.L_EXT"] , control_activity = anti_act4)#1.-0.5*anti_act_tmp)
    controlActivity(control_id = muscle_ids["thigh.R_FLEX"], control_activity = act1)#0.5*act_tmp)#
    controlActivity(control_id = muscle_ids["thigh.R_EXT"] , control_activity = anti_act1)#1.-0.5*act_tmp)#

