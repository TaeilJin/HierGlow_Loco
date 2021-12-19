import os
import numpy as np
from .motion_data_separate import TrainDataset_Inpainting, TestDataset_Inpainting, ValidDataset_Inpainting
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from visualization.plot_animation import plot_animation, plot_animation_withRef,save_animation_BVH
import joblib
from torch.utils.data import DataLoader, Dataset
from scipy.spatial.distance import pdist
from sklearn.metrics import mean_squared_error
from scipy.spatial.transform import Rotation as R

def inv_standardize(data, scaler):      
    shape = data.shape
    flat = data.reshape((shape[0]*shape[1], shape[2]))
    scaled = scaler.inverse_transform(flat).reshape(shape)
    return scaled        

def fit_and_standardize(data):
    shape = data.shape
    flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
    scaler = StandardScaler().fit(flat)
    scaled = scaler.transform(flat).reshape(shape)
    return scaled, scaler

def standardize(data, scaler):
    shape = data.shape
    flat = data.copy().reshape((shape[0]*shape[1], shape[2]))
    scaled = scaler.transform(flat).reshape(shape)
    return scaled
  

class LocomotionSP():

    def __init__(self, hparams, is_training):
    
        data_root = hparams.Dir.data_root
        
        self.frame_rate = hparams.Data.framerate
        #load data
        self.scaler =StandardScaler()
        self.scaler = joblib.load(os.path.join(data_root,'loco_augmentation_npz/mixamo.pkl'))

        # data root 에서 
        self.seqlen = hparams.Data.seqlen
        feature = 3
        self.joint = 21 *feature
        
        self.train_dataset = TrainDataset_Inpainting(os.path.join(data_root,'loco_augmentation_npz'),hparams.Data.seqlen, hparams.Data.dropout)
        self.test_dataset = TestDataset_Inpainting(os.path.join(data_root,'loco_augmentation_npz'),hparams.Data.seqlen, hparams.Data.dropout)
        self.validation_dataset = ValidDataset_Inpainting(os.path.join(data_root,'loco_augmentation_npz'),hparams.Data.seqlen, hparams.Data.dropout)
                
        self.n_x_channels = self.joint
        self.n_cond_channels = self.n_x_channels*hparams.Data.seqlen + 3*(hparams.Data.seqlen + 1 + hparams.Data.n_lookahead)

        self.ee_idx = [15,16,17,
                    27,28,29,
                    39,40,41 ]

        self.n_test = self.test_dataset.__len__()

    def save_APD_Score(self, control_data, K_motion_data, totalClips,filename):
        np.savez(filename + "_APD_testdata.npz", clips=K_motion_data)
        #K_motion_data = np.load("../data/results/locomotion/MG/log_20211103_1638/0_sampled_temp100_0k_APD_testdata.npz")['clips'].astype(np.float32)
        K, nn, ntimesteps, feature = K_motion_data.shape
        total_APD_score = np.zeros(nn)
        if totalClips != K:
            print("wrong! different motions")
        else :
            for nBatch in range(nn):
                k_motion_data = K_motion_data[:,nBatch,...]
                batch_control_data = control_data[nBatch:nBatch+1,...]
                k_control_data = np.repeat(batch_control_data,K,axis=0)

                apd_score = self.calculate_APD(k_control_data,k_motion_data)

                total_APD_score[nBatch] = apd_score
            print(f'APD of_{nn}_motion:_{total_APD_score.shape}_:{total_APD_score}_mean:{np.mean(total_APD_score)}')    
            np.savez(filename + "_APD_score.npz", clips=total_APD_score)

    def save_APD_Score_withRef(self, reference_data, control_data, K_motion_data, totalClips,filename):
        #np.savez(filename + "score_HG_reduce_dance_(-2,2)_testdata.npz", clips=K_motion_data)
        
        K_motion_data = np.load("../data/results/test_Dance_Full/0_sampled_temp100_0kscore_HG_dance_(-2,2)_testdata.npz")['clips'].astype(np.float32)
        
        #K_motion_data = np.load("../data/results/locomotion/log_20211116_2105/0_sampled_temp100_0kscore_HG_nonSP_dance_(-2,2)_testdata.npz")['clips'].astype(np.float32)
        #K_motion_data = np.load("../data/results/locomotion/log_20211116_2315/0_sampled_temp100_0kscore_HG_reduce_dance_(-2,2)_testdata.npz")['clips'].astype(np.float32)
        
        #K_motion_data_del = np.delete(K_motion_data,[5,19,31,37],1)
        K, nn, ntimesteps, feature = K_motion_data.shape
        total_APD_score = np.zeros(nn)
        total_ADE_score = np.zeros(nn)
        total_FDE_score = np.zeros(nn)
        total_EED_score = np.zeros(nn)
        if totalClips != K:
            print("wrong! different motions")
        else :

            for nBatch in range(nn):

                apd_score =0
                eed_score =0

                if nBatch != 5 and nBatch !=9 and nBatch != 10 and nBatch !=19 and nBatch !=20 and nBatch != 31 and nBatch != 37 and nBatch !=57 :
                    k_motion_data = K_motion_data[:,nBatch,...]
                    batch_reference_data = reference_data[nBatch:nBatch+1,...]
                    batch_control_data = control_data[nBatch:nBatch+1,...]
                    k_control_data = np.repeat(batch_control_data,K,axis=0)
                    k_gt_data = np.repeat(batch_reference_data,K,axis=0)

                    # after scaler
                    animation_data = np.concatenate((k_motion_data,k_control_data), axis=2)
                    anim_clip = inv_standardize(animation_data, self.scaler)
                    gt_data = np.concatenate((k_gt_data,k_control_data), axis=2)
                    ref_clip = inv_standardize(gt_data, self.scaler)

                    # go to the world
                    for i in range(K):
                        rot = R.from_quat([0,0,0,1])
                        translation = np.array([[0,0,0]])
                        translations = np.zeros((anim_clip.shape[1],3))
                        
                        joints, root_dx, root_dz, root_dr = anim_clip[i,:,:-3], anim_clip[i,:,-3], anim_clip[i,:,-2], anim_clip[i,:,-1] # K test data
                        gt_joints, root_dx, root_dz, root_dr = ref_clip[i,:,:-3], ref_clip[i,:,-3], ref_clip[i,:,-2], ref_clip[i,:,-1] # duplicated gt data
                        
                        joints = joints.reshape((len(joints), -1, 3))
                        gt_joints = gt_joints.reshape((len(gt_joints), -1, 3))
                        
                        for p in range(len(joints)):
                            joints[p,:,:] = rot.apply(joints[p])
                            joints[p,:,0] = joints[p,:,0] + translation[0,0]
                            joints[p,:,2] = joints[p,:,2] + translation[0,2]

                            gt_joints[p,:,:] = rot.apply(gt_joints[p])
                            gt_joints[p,:,0] = gt_joints[p,:,0] + translation[0,0]
                            gt_joints[p,:,2] = gt_joints[p,:,2] + translation[0,2]

                            rot = R.from_rotvec(np.array([0,-root_dr[p],0])) * rot
                            translation = translation + rot.apply(np.array([root_dx[p], 0, root_dz[p]]))
                            translations[p,:] = translation
                        
                        joints = joints.reshape((len(joints),-1))
                        gt_joints = gt_joints.reshape((len(gt_joints),-1))
                        
                        motion_clip = joints /1.59
                        gt_clip = gt_joints /1.59

                        # get score
                        apd_score += self.calculate_APD(motion_clip)
                        eed_score += self.caclulate_EED(gt_clip,motion_clip)
                        ade_score = self.calculate_ADE(gt_clip,motion_clip)
                        #fde_score = self.calculate_FDE(gt_clip,motion_clip)
                        
                total_APD_score[nBatch] = np.mean(apd_score)
                #total_ADE_score[nBatch] = ade_score
                #total_FDE_score[nBatch] = fde_score
                total_EED_score[nBatch] = np.mean(eed_score)


            total_APD_score = np.delete(total_APD_score,[5,9,10,19,20,31,37,57],0)
            total_ADE_score = np.delete(total_ADE_score,[5,9,10,19,20,31,37,57],0)
            total_FDE_score = np.delete(total_FDE_score,[5,9,10,19,20,31,37,57],0)
            total_EED_score = np.delete(total_EED_score,[5,9,10,19,20,31,37,57],0)
            print(f'APD of_{nn}_motion:_{total_APD_score.shape}_:{total_APD_score}_mean:{np.mean(total_APD_score)}')    
            np.savez(filename + "_APD_score.npz", clips=total_APD_score)
            print(f'ADE of_{nn}_motion:_{total_ADE_score.shape}_:{total_ADE_score}_mean:{np.mean(total_ADE_score)}')    
            np.savez(filename + "_ADE_score.npz", clips=total_ADE_score)
            # print(f'FDE of_{nn}_motion:_{total_FDE_score.shape}_:{total_FDE_score}_mean:{np.mean(total_FDE_score)}')    
            # np.savez(filename + "_FDE_score.npz", clips=total_FDE_score)
            print(f'EED of_{nn}_motion:_{total_EED_score.shape}_:{total_EED_score}_mean:{np.mean(total_EED_score)}')    
            np.savez(filename + "_FDE_score.npz", clips=total_EED_score)

    def calculate_FDE(self, gt_clip, motion_clip):
        
        diff = motion_clip - gt_clip # (K,70,63)
        
        # k number's l2 (diff) 
        dist = np.linalg.norm(diff, axis=1)[:,-1]
        return dist.min()

    def caclulate_EED(self, gt_clip, motion_clip):
        
        motion_clip_ee = motion_clip[:,self.ee_idx]
        gt_clip_ee = gt_clip[:,self.ee_idx]

        motion_clip_ee = np.reshape(motion_clip,(motion_clip.shape[0],-1))
        gt_clip_ee = np.reshape(gt_clip,(gt_clip.shape[0],-1))
        
        MSE = mean_squared_error(motion_clip_ee,gt_clip_ee)
        MSE_10 = np.sqrt(MSE)
        return np.mean(MSE_10)

        



        



    def calculate_ADE(self, gt_clip, motion_clip):
        
        diff = motion_clip-gt_clip # (K,70,63)
        
        # k number's l2 (diff) 
        dist = np.linalg.norm(diff, axis=1).mean(axis=-1)
        return dist.min()
           

    def calculate_APD(self, motion_clip):
        
        motion_clip = np.reshape(motion_clip,(motion_clip.shape[0],-1))
        
        dist = pdist(motion_clip)

        apd = dist.mean().item()

        # #check
        # apd =0
        # n_clips = min(self.n_test, anim_clip.shape[0])
        # for i in range(0,n_clips):
        #     filename_ = f'test_{str(i)}.mp4'
        #     print('writing:' + filename_)
        #     parents = np.array([0,1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20]) - 1
        #     plot_animation(anim_clip[i,self.seqlen:,:], parents, filename_, fps=self.frame_rate, axis_scale=60)

        return (apd)

    def save_animation_withRef_withBVH(self, control_data, motion_data, refer_data, logdir, filename):
        animation_data = np.concatenate((motion_data,control_data), axis=2)
        reference_data = np.concatenate((refer_data,control_data), axis=2)

        anim_clip = inv_standardize(animation_data, self.scaler)
        ref_clip = inv_standardize(reference_data,self.scaler)
        np.savez(filename + ".npz", clips=anim_clip)
        n_clips = min(self.n_test, anim_clip.shape[0])
        for i in range(0,n_clips):
            filename_ = f'{logdir}/{str(i)}_{filename}.mp4'
            print('writing:' + filename_)
            parents = np.array([0,1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20]) - 1
            
            plot_animation_withRef(anim_clip[i,self.seqlen:,:],ref_clip[i,self.seqlen:,:], parents, filename_, fps=self.frame_rate, axis_scale=2.0)
            save_animation_BVH(anim_clip[i,self.seqlen:,:],parents,filename_)

    def save_animation_withBVH(self, control_data, motion_data, logdir, filename):
        animation_data = np.concatenate((motion_data,control_data), axis=2)
        anim_clip = inv_standardize(animation_data, self.scaler)
        np.savez(filename + ".npz", clips=anim_clip)
        n_clips = min(self.n_test, anim_clip.shape[0])
        for i in range(0,n_clips):
            filename_ = f'{logdir}/{str(i)}_{filename}.mp4'
            print('writing:' + filename_)
            parents = np.array([0,1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20]) - 1
            
            plot_animation(anim_clip[i,self.seqlen:,:], parents, filename_, fps=self.frame_rate, axis_scale=2.0)
            save_animation_BVH(anim_clip[i,self.seqlen:,:],parents,filename_)
     
    def save_animation_withRef(self, control_data, motion_data, refer_data, filename):
        animation_data = np.concatenate((motion_data,control_data), axis=2)
        reference_data = np.concatenate((refer_data,control_data), axis=2)

        anim_clip = inv_standardize(animation_data, self.scaler)
        ref_clip = inv_standardize(reference_data,self.scaler)
        np.savez(filename + ".npz", clips=anim_clip)
        n_clips = min(self.n_test, anim_clip.shape[0])
        for i in range(0,n_clips):
            filename_ = f'{filename}_{str(i)}.mp4'
            print('writing:' + filename_)
            parents = np.array([0,1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20]) - 1
            plot_animation_withRef(anim_clip[i,self.seqlen:,:],ref_clip[i,self.seqlen:,:], parents, filename_, fps=self.frame_rate, axis_scale=2.0)

    def save_animation(self, control_data, motion_data, filename):
        animation_data = np.concatenate((motion_data,control_data), axis=2)
        anim_clip = inv_standardize(animation_data, self.scaler)
        np.savez(filename + ".npz", clips=anim_clip)
        n_clips = min(self.n_test, anim_clip.shape[0])
        for i in range(0,n_clips):
            filename_ = f'{filename}_{str(i)}.mp4'
            print('writing:' + filename_)
            parents = np.array([0,1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20]) - 1
            plot_animation(anim_clip[i,self.seqlen:,:], parents, filename_, fps=self.frame_rate, axis_scale=2.0)
        

    def n_channels(self):
        return self.n_x_channels, self.n_cond_channels
        
    def get_train_dataset(self):
        return self.train_dataset
        
    def get_test_dataset(self):
        return self.test_dataset

    def get_validation_dataset(self):
        return self.validation_dataset
        
		