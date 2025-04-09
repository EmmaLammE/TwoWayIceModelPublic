import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import seaborn as sns
import h5py
import argparse
from torch.utils.data import Dataset, DataLoader

sys.path.append('../src/')
from model_grain_kde import FNO1d

plt.rcParams.update({'font.size': 24})
S2Y = 3600*24*365.25

def main(data_name, data_path, model_path,compare_fig_name, grainsize_range):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Determine if CUDA is available

    # Load the model
    model = FNO1d(64, 32, 40, 'tanh', 'L2')
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Load and process the data
    # grain_known: [num of samples, kde reso, steps known]
    # grain_pred:  [num of samples, kde reso, steps pred]
    # grain_pred:  [num of samples, kde reso, 1]
    grain_init, grain_true, strain_rate, temperature, pressure = read_hdf5_arrays_optimized(data_name, 'test_data')
    grain_known = grain_init.clone()
    
    steps_to_pred = 160
    grain_pred_all = torch.zeros(grain_known.shape[0],grain_known.shape[1],steps_to_pred)
    control_params = torch.zeros(grain_known.shape[0],3,steps_to_pred)
    # Simulation/Prediction loop
    for i in range(steps_to_pred):
        grain_pred = model(grain_known, strain_rate, temperature, pressure) # strain rate shape [num samples, kde reso, step_known]
        grain_known = torch.cat((grain_known[..., 1:], grain_pred), dim=-1)
        grain_pred_all[:,:,i] = grain_pred[:,:,0]
        control_params[:,0,i] = strain_rate[:,0,0]
        control_params[:,1,i] = temperature[:,0,0]
        control_params[:,2,i] = pressure[:,0,0]
    for i in range(grain_known.shape[0]):
        err = torch.sum(torch.abs(grain_pred_all[i,:,:] - grain_true[i,:,:steps_to_pred]))
        print(f'err at sample {i} at all steps: {err}')

    # save results
    torch.save(grain_init, data_path + 'temp_test_syn_init_grain_kde.pt')
    torch.save(grain_true, data_path + 'temp_test_syn_true_grain_kde.pt')
    torch.save(grain_pred_all, data_path + 'temp_test_syn_pred_grain_kde.pt')
    torch.save(control_params, data_path + 'temp_test_syn_grain_control_params.pt')
    print(f'saved at {data_path}test_syn_init_grain_kde.pt')
    # Post-process and plot
    plot_results(grain_init, grain_true, grain_pred, compare_fig_name, grainsize_range)

def read_hdf5_arrays_optimized(file_path, dataset_name):
    with h5py.File(file_path, 'r') as f:
        data = f[dataset_name]
        grain_known = np.array(data[:,:,0:40])
        grain_true = np.array(data[:,:,40:200])
        strain_rate = np.array(data[:,:,200:240])
        temperature = np.array(data[:,:,240:280])
        pressure = np.array(data[:,:,280:])
    
    return torch.from_numpy(grain_known).float(), torch.from_numpy(grain_true).float(), \
           torch.from_numpy(strain_rate).float(), torch.from_numpy(temperature).float(), \
           torch.from_numpy(pressure).float()

def plot_results(grain_init, grain_true, grain_pred, compare_fig_name, grainsize_range):
    fig = plt.figure(figsize=(25, 18),dpi=300)
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1,1], wspace=0.15, hspace=0.15)
    for i, snap in enumerate([0, 20, -1, 1]):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(grainsize_range, grain_init[snap, :, 0].detach().numpy(), label='Initial')
        ax.plot(grainsize_range, grain_true[snap, :, -1].detach().numpy(), label='True')
        ax.plot(grainsize_range, grain_pred[snap, :, -1].detach().numpy(), label='Pred')
        ax.set_title(f"Sample {snap}")
        ax.legend()

    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    fig.savefig(compare_fig_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the grain KDE test simulation')
    parser.add_argument('--data_path', type=str, default='/elle_grainsize_results/jcp/data/test/', help='Path to test data file')
    parser.add_argument('--data_name', type=str, default='jcp_grain_kde_mode64_width32_tanh+L2_test_data.h5', help='Name of test data file')
    parser.add_argument('--model_path', type=str, default='./../model/jcp_grain_kde_mode64_width32_tanh+L2_custom_smooth10_N67_epoch201.pth', help='Path to trained model')
    parser.add_argument('--compare_fig_name', type=str, default='./../results/jcp_grain_kde_compare.png', help='Save path for comparison fig')
    parser.add_argument('--grainsize_range', type=str, help='Range of grain sizes, e.g., "3,4800,256"', default='3,4800,256')
    args = parser.parse_args()

    grainsize_range = np.linspace(*map(int, args.grainsize_range.split(',')))
    test_data_to_load = args.data_path + args.data_name
    main(test_data_to_load, args.data_path, args.model_path, args.compare_fig_name, grainsize_range)
