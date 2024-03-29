import numpy as np

import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import ArtistAnimation
from scipy.spatial.transform import Rotation as R

import sys
sys.path.append('BVH_Export')

from BVH import save, load
from Quaternions import Quaternions
from Animation import Animation
from InverseKinematics import JacobianInverseKinematics

def transforms_inv(ts):
    fts = ts.reshape(-1, 4, 4)
    fts = np.array(list(map(lambda x: np.linalg.inv(x), fts)))
    return fts.reshape(ts.shape)

def transforms_blank(shape):
    # type(shape): tuple
    ts = np.zeros(shape + (4, 4))
    ts[..., 0, 0] = 1.0; ts[..., 1, 1] = 1.0;
    ts[..., 2, 2] = 1.0; ts[..., 3, 3] = 1.0;
    return ts
    
def save_animation_BVH(clip, parents, filename=None ):
    rot = R.from_quat([0,0,0,1])
    translation = np.array([[0,0,0]])
    translations = np.zeros((clip.shape[0],3))
    
    joints, root_dx, root_dz, root_dr = clip[:,:-3], clip[:,-3], clip[:,-2], clip[:,-1]
    joints = joints.reshape((len(joints), -1, 3))
    for i in range(len(joints)):
        joints[i,:,:] = rot.apply(joints[i])
        joints[i,:,0] = joints[i,:,0] + translation[0,0]
        joints[i,:,2] = joints[i,:,2] + translation[0,2]
        rot = R.from_rotvec(np.array([0,-root_dr[i],0])) * rot
        translation = translation + rot.apply(np.array([root_dx[i], 0, root_dz[i]]))
        translations[i,:] = translation
    #
    
    rest, names, _ = load('visualization/modified_rest.bvh')
    names = np.array(names)
    names = names.tolist()
    
    targets = joints
    
    anim = rest.copy()
    anim.positions = anim.positions.repeat(len(targets), axis=0)
    anim.rotations.qs = anim.rotations.qs.repeat(len(targets), axis=0)

    anim.positions[:,0] = targets[:,0]
    anim.rotations[:,0] = anim.rotations[:,0]

    targetmap = {}
    for ti in range(targets.shape[1]):
        targetmap[ti] = targets[:,ti]
    ik = JacobianInverseKinematics(anim, targetmap, iterations=10, damping=10.0, silent=True)
    ik()
    save(filename + '.bvh', anim, names, frametime=1.0/30.0)

def plot_animation_withRef(clip, refclip, parents, filename=None, fps=30, axis_scale=50, elev=45, azim=45):
        
    rot = R.from_quat([0,0,0,1])
    translation = np.array([[0,0,0]])
    translations = np.zeros((clip.shape[0],3))
    
    joints, root_dx, root_dz, root_dr = clip[:,:-3], clip[:,-3], clip[:,-2], clip[:,-1]
    joints_ref = refclip[:,:-3]

    joints = joints.reshape((len(joints), -1, 3))
    joints_ref = joints_ref.reshape((len(joints_ref), -1, 3))
    for i in range(len(joints)):
        joints[i,:,:] = rot.apply(joints[i])
        joints[i,:,0] = joints[i,:,0] + translation[0,0]
        joints[i,:,2] = joints[i,:,2] + translation[0,2]
        
        joints_ref[i,:,:] = rot.apply(joints_ref[i])
        joints_ref[i,:,0] = joints_ref[i,:,0] + translation[0,0]
        joints_ref[i,:,2] = joints_ref[i,:,2] + translation[0,2]
        
        
        rot = R.from_rotvec(np.array([0,-root_dr[i],0])) * rot
        translation = translation + rot.apply(np.array([root_dx[i], 0, root_dz[i]]))
        translations[i,:] = translation
            
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-axis_scale, axis_scale)
    ax.set_zlim3d( 0, axis_scale)
    ax.set_ylim3d(-axis_scale, axis_scale)
    ax.grid(True)
    ax.set_axis_off()

    ax.view_init(elev=elev, azim=azim)

    xs = np.linspace(-200, 200, 50)
    ys = np.linspace(-200, 200, 50)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros(X.shape)
    
    wframe = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color='grey',lw=0.2)
    
    points = []
    lines = []
    acolors = list(sorted(colors.cnames.keys()))[::-1]
    tmp = np.zeros(translations.shape)    
    lines.append(plt.plot(translations[:,0],-translations[:,2], 
        lw=2, path_effects=[pe.Stroke(linewidth=1, foreground='black'), pe.Normal()])[0])
    lines.append([plt.plot([0,0], [0,0], [0,0], color='red', 
        lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])[0] for _ in range(joints.shape[1])])
    lines.append([plt.plot([0,0], [0,0], [0,0], color='green', 
        lw=2, path_effects=[pe.Stroke(linewidth=1, foreground='black'), pe.Normal()])[0] for _ in range(joints_ref.shape[1])])
    def animate(i):
        
        changed = []
        
        for j in range(len(parents)):
            if parents[j] != -1:
                lines[1][j].set_data(np.array([[joints[i,j,0], joints[i,parents[j],0]],[-joints[i,j,2],-joints[i,parents[j],2]]]))
                lines[1][j].set_3d_properties(np.array([ joints[i,j,1],joints[i,parents[j],1]]))
        
        for j in range(len(parents)):
            if parents[j] != -1:
                lines[2][j].set_data(np.array([[joints_ref[i,j,0], joints_ref[i,parents[j],0]],[-joints_ref[i,j,2],-joints_ref[i,parents[j],2]]]))
                lines[2][j].set_3d_properties(np.array([ joints_ref[i,j,1],joints_ref[i,parents[j],1]]))

        changed += lines
            
        return changed
        
    plt.tight_layout()
        
    ani = animation.FuncAnimation(fig, 
        animate, np.arange(joints.shape[0]), interval=1000/fps)

    if filename != None:
        ani.save(filename, fps=fps, bitrate=13934)
        ani.event_source.stop()
        del ani
        plt.close()    
    try:
        plt.show()
        plt.save()
    except AttributeError as e:
        pass

def plot_animation(clip, parents, filename=None, fps=30, axis_scale=50, elev=45, azim=45):
        
    rot = R.from_quat([0,0,0,1])
    translation = np.array([[0,0,0]])
    translations = np.zeros((clip.shape[0],3))
    
    joints, root_dx, root_dz, root_dr = clip[:,:-3], clip[:,-3], clip[:,-2], clip[:,-1]
    joints = joints.reshape((len(joints), -1, 3))
    for i in range(len(joints)):
        joints[i,:,:] = rot.apply(joints[i])
        joints[i,:,0] = joints[i,:,0] + translation[0,0]
        joints[i,:,2] = joints[i,:,2] + translation[0,2]
        rot = R.from_rotvec(np.array([0,-root_dr[i],0])) * rot
        translation = translation + rot.apply(np.array([root_dx[i], 0, root_dz[i]]))
        translations[i,:] = translation
            
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-axis_scale, axis_scale)
    ax.set_zlim3d( 0, axis_scale)
    ax.set_ylim3d(-axis_scale, axis_scale)
    ax.grid(True)
    ax.set_axis_off()

    ax.view_init(elev=elev, azim=azim)

    xs = np.linspace(-200, 200, 50)
    ys = np.linspace(-200, 200, 50)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros(X.shape)
    
    wframe = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color='grey',lw=0.2)
    
    points = []
    lines = []
    acolors = list(sorted(colors.cnames.keys()))[::-1]
    tmp = np.zeros(translations.shape)    
    lines.append(plt.plot(translations[:,0],-translations[:,2], 
        lw=2, path_effects=[pe.Stroke(linewidth=1, foreground='black'), pe.Normal()])[0])
    lines.append([plt.plot([0,0], [0,0], [0,0], color='red', 
        lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])[0] for _ in range(joints.shape[1])])
    def animate(i):
        
        changed = []
        
        for j in range(len(parents)):
            if parents[j] != -1:
                lines[1][j].set_data(np.array([[joints[i,j,0], joints[i,parents[j],0]],[-joints[i,j,2],-joints[i,parents[j],2]]]))
                lines[1][j].set_3d_properties(np.array([ joints[i,j,1],joints[i,parents[j],1]]))

        changed += lines
            
        return changed
        
    plt.tight_layout()
        
    ani = animation.FuncAnimation(fig, 
        animate, np.arange(joints.shape[0]), interval=1000/fps)

    if filename != None:
        ani.save(filename, fps=fps, bitrate=13934)
        ani.event_source.stop()
        del ani
        plt.close()    
    try:
        plt.show()
        plt.save()
    except AttributeError as e:
        pass
