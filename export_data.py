import os
# import re
import sys
import numpy as np
import scipy.io as io
import scipy.ndimage.filters as filters

sys.path.append('./motion/datasets')
# sys.path.append('./motionsynth_data/motion')

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import pickle
from itertools import combinations
from visualization.plot_animation import plot_animation

def get_txtfiles(directory):
    return [os.path.join(directory,f) for f in sorted(list(os.listdir(directory)))
    if os.path.isfile(os.path.join(directory,f))
    and f.endswith('.txt')] 

def softmax(x, **kw):
    softness = kw.pop('softness', 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))

def softmin(x, **kw):
    return -softmax(-x, **kw)
    
def process_file(filename, window=80, window_step=50):
    
    #anim, names, frametime = BVH.load(filename)
    """ Raw 3D Data"""
    envroot_based_positions = np.loadtxt(filename,delimiter=' ',dtype='float32')

    """ Remove Uneeded Joints """
    positions = envroot_based_positions[...]

    if(positions.shape[-1] != 22*3):
        print("mayday! : feature dim is not 66! %i", positions.shape[-1])
    """ Slide over windows """
    windows = []
    cnt =0
    for j in range(0, len(positions)-window//8, window_step):
    
        """ If slice too small pad out by repeating start and end poses """
        slice = positions[j:j+window]
        if len(slice) < window:
            left  = slice[:1].repeat((window-len(slice))//2 + (window-len(slice))%2, axis=0) # start pose 하나를 중첩시킨다
            right = slice[-1:].repeat((window-len(slice))//2, axis=0) # end pose 하나를 중첩시킨다
            slice = np.concatenate([left, slice, right], axis=0)
        
        if len(slice) != window: raise Exception()
        
        windows.append(slice)
        cnt +=1
        
    return np.asarray(windows)
def process_file_pos_fromRotation(filename, window=80, window_step=50):
    
    #anim, names, frametime = BVH.load(filename)
    """ Raw 3D Data"""
    envroot_based_positions = np.loadtxt(filename,delimiter=' ',dtype='float32')

    """ Remove Uneeded Joints """
    envroot_based_positions = envroot_based_positions.reshape(len(envroot_based_positions),-1,9)
    positions = envroot_based_positions[...,:3]
    positions = positions.reshape(len(envroot_based_positions),-1)
    # positions = positions[...,np.newaxis]
    # positions = positions.reshape(-1,22,3) 
        
    """ Add Velocity, RVelocity, Foot Contacts to vector """
    # positions = positions.reshape(len(positions), -1)
    # positions = np.concatenate([positions, velocity[:,:,0]], axis=-1)
    # positions = np.concatenate([positions, velocity[:,:,2]], axis=-1)
    # positions = np.concatenate([positions, rvelocity], axis=-1)
        
    """ Slide over windows """
    windows = []
    cnt =0
    for j in range(0, len(positions)-window//8, window_step):
    
        """ If slice too small pad out by repeating start and end poses """
        slice = positions[j:j+window]
        if len(slice) < window:
            left  = slice[:1].repeat((window-len(slice))//2 + (window-len(slice))%2, axis=0) # start pose 하나를 중첩시킨다
            right = slice[-1:].repeat((window-len(slice))//2, axis=0) # end pose 하나를 중첩시킨다
            slice = np.concatenate([left, slice, right], axis=0)
        
        if len(slice) != window: raise Exception()
        
        windows.append(slice)
        cnt +=1
        
    return windows    

def get_files(directory):
    return [os.path.join(directory,f) for f in sorted(list(os.listdir(directory)))
    if os.path.isfile(os.path.join(directory,f))
    and f.endswith('.bvh') and f != 'rest.bvh'] 

def get_txtfiles(directory):
    return [os.path.join(directory,f) for f in sorted(list(os.listdir(directory)))
    if os.path.isfile(os.path.join(directory,f))
    and f.endswith('.txt')] 



def create_Autoreg_OutputData3(seqlen,data):
    # sequence data
    joint_data = data[:,:,:] # joint data
    # current pose (output)
    n_frames = joint_data.shape[1]
    new_x = concat_sequence_3(1, joint_data[:,seqlen:n_frames,:])
    # control autoreg(10) 
    autoreg_seq = concat_sequence_3(seqlen,joint_data[:,:n_frames-1,:])
    return new_x, autoreg_seq

def create_Sequence_CondOnly_OutputData3(seqlen,data,condinfo):
    # sequence data
    joint_data = data[:,:,:-condinfo] # joint data
    control_data = data[:,:,-condinfo:] # control data 
    # current pose (output)
    n_frames = joint_data.shape[1]
    new_x = concat_sequence_3(1, joint_data[:,seqlen:n_frames,:])
    # control autoreg(10) + control(11 or 1)
    autoreg_control = concat_sequence_3(seqlen +1, control_data)
    single_control = concat_sequence_3(1, control_data[:,seqlen:n_frames,:])
    
    #
    autoreg_seq = concat_sequence_3(seqlen,joint_data[:,:n_frames-1,:])
    autoreg_seq_control = np.concatenate((autoreg_seq,autoreg_control),axis=-1)
    autoreg_seq_single_control = np.concatenate((autoreg_seq,single_control),axis=-1)

    return new_x, autoreg_control,single_control , autoreg_seq_control, autoreg_seq_single_control

def create_Sequence_OutputData3(seqlen,data,condinfo):
    # sequence data
    joint_data = data[:,:,:-condinfo] # joint data
    control_data = data[:,:,-condinfo:] # control data 
    # current pose (output)
    n_frames = joint_data.shape[1]
    new_x = concat_sequence_3(1, joint_data[:,seqlen:n_frames,:])
    # control autoreg(10) + control(11 or 1)
    autoreg_seq = concat_sequence_3(seqlen,joint_data[:,:n_frames-1,:])
    autoreg_control = concat_sequence_3(seqlen +1, control_data)
    new_cond = np.concatenate((autoreg_seq,autoreg_control),axis=-1)

    return new_x, new_cond

def concat_sequence_3(seqlen, data):
    """ 
    Concatenates a sequence of features to one.
    """
    nn,n_timesteps,n_feats = data.shape
    L = n_timesteps-(seqlen-1)
    inds = np.zeros((L, seqlen)).astype(int)

    #create indices for the sequences we want
    rng = np.arange(0, n_timesteps)
    for ii in range(0,seqlen):  
        inds[:, ii] = np.transpose(rng[ii:(n_timesteps-(seqlen-ii-1))])  

    #slice each sample into L sequences and store as new samples 
    cc=data[:,inds,:].copy()

    #print ("cc: " + str(cc.shape))

    #reshape all timesteps and features into one dimention per sample
    dd = cc.reshape((nn, L, seqlen*n_feats))
    #print ("dd: " + str(dd.shape))
    return dd

def partial_fit(data,scaler):
    shape = data.shape
    flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
    scaler.partial_fit(flat)
    
    return scaler

def inv_standardize(data, scaler):      
    shape = data.shape
    flat = data.reshape((shape[0]*shape[1], shape[2]))
    scaled = scaler.inverse_transform(flat).reshape(shape)
    return scaled 
def standardize(data, scaler):
    shape = data.shape
    flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
    scaled = scaler.transform(flat).reshape(shape)
    return scaled

def augmentedRootInCircle(data):
    # x,z
    theta_radian = [ 0,          np.math.pi/4,   np.math.pi/2,     3 * np.math.pi/4,
                     np.math.pi, 5*np.math.pi/4, -1*np.math.pi/2, -1 * np.math.pi/4 ] 
    #
    data_origin = data.copy()
    data_augment = data.copy()
    radius = 0.4
    #
    nn, timesteps, feature = data.shape  
    
    for k in range(len(theta_radian)):
        trans = np.zeros((nn,timesteps,3))
        trans[:,:,0] = data_origin[:,:,-3] + radius * np.math.cos(theta_radian[k])
        trans[:,:,2] = data_origin[:,:,-2] + radius * np.math.sin(theta_radian[k])
        #
        batch_trans = np.tile(trans[:,:,:],(1,1,22))
        #
        data_trans = data_origin.copy()
        data_trans[...,:-3] = data_origin[...,:-3] - batch_trans
        #data_trans[:,0,-3] = data_origin[:,0,-3] - trans[:,0,0]
        #data_trans[:,0,-2] = data_origin[:,0,-2] - trans[:,0,2]
        
        data_augment = np.concatenate((data_augment,data_trans),axis=0)
    #
    return data_augment        

    
#load data
framerate = 20
data_root = "/root/home/project/data/locomotion/"

train_X = np.load(os.path.join(data_root, 'all_locomotion_train_'+str(20)+'fps_aug.npz'))['clips'].astype(np.float32)
test_X = np.load(os.path.join(data_root, 'all_locomotion_test_'+str(20)+'fps.npz'))['clips'].astype(np.float32)
valid_X = train_X[:100,...]
train_X = train_X[100:,...]
# masking
scaler = StandardScaler()
    
# # scaler update
scaler = partial_fit(train_X, scaler)

# # scaler 저장하기
joblib.dump(scaler,os.path.join(data_root,'loco_augmentation_npz/mixamo.pkl'))

# scaler 불러오기
scaler = joblib.load(os.path.join(data_root,'loco_augmentation_npz/mixamo.pkl'))

#train
train_X = standardize(train_X,scaler)
valid_X = standardize(valid_X,scaler)
test_X = standardize(test_X,scaler)

# test data 늘리기
n_test = test_X.shape[0]
n_tiles = 1+100//n_test
all_test_data = np.tile(test_X.copy(), (n_tiles,1,1))

# result data
datafilecnt = 0
datafilecnt_train = 0
datafilecnt_valid = 0
seqlen = 10

info_Condition = 3
train_seqX, train_seqControl, train_singleControl, train_seqControl_autoreg, train_singleControl_autoreg = create_Sequence_CondOnly_OutputData3(seqlen,train_X.copy(),info_Condition)
for i in range(0,train_X.shape[0]):
    print("datafilecnt_train"+str(datafilecnt_train))
    np.savez_compressed(os.path.join(data_root,'loco_augmentation_npz/train_scaled_seqX_'+str(datafilecnt_train)+'.npz'), clips = train_seqX[i,...])

    np.savez_compressed(os.path.join(data_root,'loco_augmentation_npz/train_scaled_singleControl_'+str(datafilecnt_train)+'.npz'), clips = train_singleControl[i,...])
    np.savez_compressed(os.path.join(data_root,'loco_augmentation_npz/train_scaled_seqControl_'+str(datafilecnt_train)+'.npz'), clips = train_seqControl[i,...])

    np.savez_compressed(os.path.join(data_root,'loco_augmentation_npz/train_scaled_singleControlAutoreg_'+str(datafilecnt_train)+'.npz'), clips = train_singleControl_autoreg[i,...])
    np.savez_compressed(os.path.join(data_root,'loco_augmentation_npz/train_scaled_seqControlAutoreg_'+str(datafilecnt_train)+'.npz'), clips = train_seqControl_autoreg[i,...])
    
    datafilecnt_train += 1

valid_seqX, valid_seqControl, valid_singleControl, valid_seqControl_autoreg, valid_singleControl_autoreg = create_Sequence_CondOnly_OutputData3(seqlen,valid_X.copy(),info_Condition)
for i in range(0,valid_X.shape[0]):
    print("datafilecnt_valid"+str(datafilecnt_valid))
    np.savez_compressed(os.path.join(data_root,'loco_augmentation_npz/valid_scaled_seqX_'+str(datafilecnt_valid)+'.npz'), clips = valid_seqX[i,...])

    np.savez_compressed(os.path.join(data_root,'loco_augmentation_npz/valid_scaled_singleControl_'+str(datafilecnt_valid)+'.npz'), clips = valid_singleControl[i,...])
    np.savez_compressed(os.path.join(data_root,'loco_augmentation_npz/valid_scaled_seqControl_'+str(datafilecnt_valid)+'.npz'), clips = valid_seqControl[i,...])

    np.savez_compressed(os.path.join(data_root,'loco_augmentation_npz/valid_scaled_singleControlAutoreg_'+str(datafilecnt_valid)+'.npz'), clips = valid_singleControl_autoreg[i,...])
    np.savez_compressed(os.path.join(data_root,'loco_augmentation_npz/valid_scaled_seqControlAutoreg_'+str(datafilecnt_valid)+'.npz'), clips = valid_seqControl_autoreg[i,...])
    datafilecnt_valid += 1   

for i_te in range(0, all_test_data.shape[0]):
    print("datafilecnt"+str(datafilecnt))
    np.savez_compressed(os.path.join(data_root,'loco_augmentation_npz/test_scaled_'+str(datafilecnt)+'.npz'), clips = all_test_data[i_te,...])
    datafilecnt += 1        
        