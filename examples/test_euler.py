import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
from PIL import Image
import seaborn as sns
import argparse

sys.path.append('../src/')
from model_grain import FNO2d
from utilities3_euler import mean_angular_error, cosine_similarity
plt.rcParams.update({'font.size': 20})
S2Y = 3600 * 24 * 365.25
    
def main(which_euler, model_path, compare_fig_name, test_data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Determine if CUDA is available

    # Load test data
    euler_init, euler_true, strain_rate, temperature, pressure = read_hdf5_arrays_optimized(test_data_path, 'test_data')
    euler_known = euler_init.clone()

    # Load the model
    # mode1, mode2, width, step known. <== this MUST be the same as the training. See .out if unsure.
    model = FNO2d(12, 12, 16, 40, 'tanh', 'L2')
    model.load_state_dict(torch.load(model_path))
    print(f'loaded model: {model_path}')

    steps_to_pred = 160
    euler_pred_all = torch.zeros(euler_known.shape[0],euler_known.shape[1],euler_known.shape[2],steps_to_pred)
    
    # Simulation/Prediction loop
    with torch.no_grad():
        for i in range(steps_to_pred):
            print(f'predicting step {i}, strain: {strain_rate[i,0,0,0]}, temperature: {temperature[i,0,0,0]}, pressure: {pressure[i,0,0,0]}')
            euler_pred = model(euler_known, strain_rate, temperature, pressure)
            euler_known = torch.cat((euler_known[..., 1:], euler_pred), dim=-1)
            euler_pred_all[:,:,:,i] = euler_pred[:,:,:,0]
    
    # save results
    torch.save(euler_init, '/elle_grainsize_results/jcp/data/test/jcp_testresults_syn_init_'+which_euler+'_kde.pt')
    torch.save(euler_true, '/elle_grainsize_results/jcp/data/test/jcp_testresults_syn_true_'+which_euler+'_kde.pt')
    torch.save(euler_pred_all, '/elle_grainsize_results/jcp/data/test/jcp_testresults_syn_pred_'+which_euler+'_kde.pt')

    # Post-process and save results
    vmin, vmax, thresh, euler_pred, euler_true = process_euler_angles(which_euler, euler_pred, euler_true)
    euler_pred = euler_pred.detach().numpy()
    euler_true = euler_true.detach().numpy()

    # Calculate errors and plot results
    plot_and_save_errors(compare_fig_name, which_euler, euler_known, euler_true, euler_pred, vmin, vmax, thresh)

def read_hdf5_arrays_optimized(file_path, dataset_name):
    with h5py.File(file_path, 'r') as f:
        data = f[dataset_name]
        euler_known = np.array(data[:,:,:,0:40])
        euler_true = np.array(data[:,:,:,40:200])
        strain_rate = np.array(data[:,:,:,200:240])
        temperature = np.array(data[:,:,:,240:280])
        pressure = np.array(data[:,:,:,280:])
    print(f'loaded data: {file_path}')
    return torch.from_numpy(euler_known).float(), torch.from_numpy(euler_true).float(), \
           torch.from_numpy(strain_rate).float(), torch.from_numpy(temperature).float(), \
           torch.from_numpy(pressure).float()


def process_euler_angles(which_euler, euler_pred, euler_true):
    if which_euler in ['euler1', 'euler3']:
        euler_pred *= 180
        euler_true *= 180
        return -180, 180, 2.5, euler_pred, euler_true
    elif which_euler == 'euler2':
        euler_pred = euler_pred * 45 + 45
        euler_true = euler_true * 45 + 45
        return 0, 90, 0.1, euler_pred, euler_true
    return None

def plot_and_save_errors(compare_fig_name, which_euler, euler_known, euler_true, euler_pred, vmin, vmax, thresh):
    snap_show = {
        'euler1': [3, 5, 10],
        'euler2': [3,5,10],
        'euler3': [3,5,10]
    }.get(which_euler, [])

    difference = euler_pred[snap_show, :, :, 0] - euler_true[snap_show, :, :, 0]
    vmin_diff = difference.min()
    vmax_diff = difference.max()
    difference[np.abs(difference) < thresh] = np.nan
    print(f"filtering out difference value smaller than {thresh} to nan.")

    err_L2 = np.zeros(euler_pred.shape[0])
    err_mean_ang = np.zeros(euler_pred.shape[0])
    err_cos_simi = np.zeros(euler_pred.shape[0])
    print("Calculating L2 norm for each predicting time slice (randomized): ")
    for i in range(euler_pred.shape[0]):
        err_L2[i] = np.sqrt(np.sum((euler_pred[i,:,:,0] - euler_true[i,:,:,0])**2)/(euler_pred.shape[1]**2))
        err_mean_ang[i] = mean_angular_error(euler_pred[i,:,:,0],euler_true[i,:,:,0])
        err_cos_simi[i] = cosine_similarity(euler_pred[i,:,:,0],euler_true[i,:,:,0])
        print(f"err at sample {i}, err L2: {err_L2[i]:.4f}, mean angular diff: {err_mean_ang[i]:.4f}, cos similarity: {err_cos_simi[i]:.4f}")

    # np.savez('./../results/jcp_syn_'+which_euler+'_err.npz',err_L2=err_L2,
    #      err_mean_ang=err_mean_ang,err_cos_simi=err_cos_simi)
    save_figure(compare_fig_name, euler_true, euler_pred, difference, snap_show, vmin, vmax, vmin_diff, vmax_diff)

def save_figure(compare_fig_name, euler_true, euler_pred, difference, snap_show, vmin, vmax, vmin_diff, vmax_diff):
    fig = plt.figure(figsize=(18, 18), dpi=300)
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.05], wspace=0.15, hspace=0.15)
    for i, snap in enumerate(snap_show):
        ax_gt = fig.add_subplot(gs[i, 0])
        im_gt = ax_gt.imshow(euler_true[snap, :, :, 0], cmap='rainbow', vmin=vmin, vmax=vmax)
        ax_gt.set_title(f"Ground Truth, snapshot {snap}")
        ax_gt.axis('off')
        
        ax_pred = fig.add_subplot(gs[i, 1])
        im_pred = ax_pred.imshow(euler_pred[snap, :, :, 0], cmap='rainbow', vmin=vmin, vmax=vmax)
        ax_pred.set_title(f"Prediction, snapshot {snap}")
        ax_pred.axis('off')
        
        ax_diff = fig.add_subplot(gs[i, 2])
        im_diff = ax_diff.imshow(difference[i, :, :], cmap='rainbow', vmin=vmin_diff, vmax=vmax_diff)
        ax_diff.set_title(f"Difference, snapshot {snap}")
        ax_diff.axis('off')

    add_colorbars(fig, gs, im_gt, im_pred, im_diff)
    fig.savefig(compare_fig_name)

def add_colorbars(fig, gs, im_gt, im_pred, im_diff):
    cbar_ax1 = fig.add_subplot(gs[3, 0])
    fig.colorbar(im_gt, cax=cbar_ax1, orientation='horizontal').set_label('Ground Truth')
    cbar_ax2 = fig.add_subplot(gs[3, 1])
    fig.colorbar(im_pred, cax=cbar_ax2, orientation='horizontal').set_label('Prediction')
    cbar_ax3 = fig.add_subplot(gs[3, 2])
    fig.colorbar(im_diff, cax=cbar_ax3, orientation='horizontal').set_label('Difference')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process Euler angle data for predictions')
    parser.add_argument('--which_euler', type=str, required=True, help='Specify which euler angle data to process (euler1, euler2, euler3)')
    parser.add_argument('--model_path', type=str, default='./../model/jcp_syn_', help='Path to the trained model file')
    parser.add_argument('--epoch', type=str, default='100', help='Epoch of the trained model file')
    parser.add_argument('--test_data_path', type=str, default='/oak/stanford/groups/jsuckale/liuwj/elle_grainsize_results/jcp/data/test/', help='Path to the test data file')
    parser.add_argument('--compare_fig_name', type=str, default='./../results/jcp_', help='Save path for comparison fig')
    args = parser.parse_args()

    test_data_to_load = args.test_data_path + 'jcp_syn_'+ args.which_euler+'_test_data.h5'
    compare_fig_name = args.compare_fig_name + args.which_euler + '_2d_compare.png'
    model_to_load = args.model_path + args.which_euler + '_mode12_width16_tanh+L2_custom_N64_epoch'+args.epoch+'.pth'
    main(args.which_euler, model_to_load, compare_fig_name, test_data_to_load)
