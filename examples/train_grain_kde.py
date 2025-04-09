import sys
sys.path.append('../src/')

import argparse
from timeit import default_timer

import torch
import torch.onnx
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import h5py
from torch.utils.data import Dataset
from scipy.stats import linregress
from model_grain_kde import FNO1d
from utilities3_grain import count_params, LpLoss
from utilities3_grain import smoothness_loss, log_normalize_np, reference_normalize_np

class H5Dataset(Dataset):
    """
    A custom PyTorch Dataset that reads samples from an HDF5 file on-the-fly.
    This avoids loading the entire dataset into memory at once.
    """

    def __init__(self, h5_file_path):
        super().__init__()
        self.h5_file_path = h5_file_path
        with h5py.File(self.h5_file_path, 'r') as f:
            # We assume the first dimension is 'N' (number of samples).
            # This requires that all relevant datasets have the same N.
            self.N = f['grain_known'].shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # We open the file *each time* we get a sample or a batch of samples.
        # This is simplest for concurrency
        with h5py.File(self.h5_file_path, 'r') as f:
            grain_kde_known   = f['grain_known'][idx]
            grain_kde_predict = f['grain_predict'][idx]
            strain_rate    = f['strain_rate'][idx]
            temperature    = f['temperature'][idx]
            pressure       = f['pressure'][idx]

        grain_kde_known   = torch.from_numpy(grain_kde_known).float()
        grain_kde_predict = torch.from_numpy(grain_kde_predict).float()
        strain_rate    = torch.from_numpy(strain_rate).float()
        temperature    = torch.from_numpy(temperature).float()
        pressure       = torch.from_numpy(pressure).float()

        return grain_kde_known, grain_kde_predict, strain_rate, temperature, pressure

def summarize_hdf5_dataset(h5_file_path, dataset_name, chunk_size=1024):
    """
    Prints shape, dtype, and min/max range of the dataset in `h5_file_path`.
    Loads data in chunks to avoid large memory usage.
    """
    with h5py.File(h5_file_path, 'r') as f:
        dset = f[dataset_name]
        shape = dset.shape
        dtype = dset.dtype
        print(f"{dataset_name} shape: {shape}, dtype: {dtype}")
        global_min = float('inf')
        global_max = float('-inf')
        N = int(dset.shape[0])
        indices = np.random.permutation(N)# first dimension is sample dimension
        # Read in small chunks along axis 0
        for i,index in enumerate(indices):
            if (i<N/500):
                chunk_data = dset[index,:,0]
                local_min = chunk_data.min()
                local_max = chunk_data.max()
                if local_min < global_min:
                    global_min = local_min
                if local_max > global_max:
                    global_max = local_max
        print(f"  data range for random {int(N/500)} samples: [{global_min}, {global_max}]\n")

def save_train_data_h5(save_file_name,grid_size,step_known,step_predict,train_index,grain_known,grain_predict,strain_rate,temperature,pressure):
    with h5py.File(save_file_name, 'w') as f:
        # 1) Create datasets with final shapes
        grain_known_train_dset = f.create_dataset(
            'grain_known', 
            shape=(len(train_index), grid_size,step_known),
            dtype=grain_known.dtype
        )
        grain_predict_train_dset = f.create_dataset(
            'grain_predict', 
            shape=(len(train_index), grid_size, step_predict),
            dtype=grain_predict.dtype
        )
        strain_rate_train_dset = f.create_dataset(
            'strain_rate', 
            shape=(len(train_index), grid_size, step_known),
            dtype=strain_rate.dtype
        )
        temperature_train_dset = f.create_dataset(
            'temperature', 
            shape=(len(train_index), grid_size, step_known),
            dtype=temperature.dtype
        )
        pressure_train_dset = f.create_dataset(
            'pressure', 
            shape=(len(train_index), grid_size, step_known),
            dtype=pressure.dtype
        )
        # 2) Copy in small batches to avoid big memory usage
        chunk_size = 1024  
        for start in range(0, len(train_index), chunk_size):
            end = start + chunk_size
            print(f'done transferring {end} samples')
            batch_idx = train_index[start:end]
            # Each of these advanced-index calls makes a copy of the slice,
            # but only of size [chunk_size, ...] so memory usage is limited.
            grain_known_train_dset[start:end]   = grain_known[batch_idx, :, :]
            grain_predict_train_dset[start:end] = grain_predict[batch_idx, :, :]
            strain_rate_train_dset[start:end]    = strain_rate[batch_idx, :, :]
            temperature_train_dset[start:end]    = temperature[batch_idx, :, :]
            pressure_train_dset[start:end]       = pressure[batch_idx, :, :]
    print(f"Saved train data to {save_file_name}")


def load_data(data_path,save_testdata_path,save_traindata_path,T_ref_bound, H_ref_bound,kde_ref_bound,S_ref_bound,num_input_params=4, num_timesteps=24, 
              grid_size=1000,step_known=1, step_predict=1, step_size = 24,
              batch_size = 2, shuffle=True):
    
    """ Load the data from the given paths and split into train, validation and test sets

    Parameters
    ----------
    step_known: how many steps are known
    step_predict: how many steps are predicted
    step_size: steps in between
    * * *     - - - - - - - - -     * * *
    ↑___↑    ↑________________↑     ↑___↑ 
    step         step size           step
    known                            pred
    """

    files = os.listdir(data_path)
    # Filter for .npz files
    grain_files = [x for x in files if x.endswith(".npz") and (x.startswith("grain_kde"))]
    grain_npz_count = len(grain_files)
    print("-----------------------------------\n-----------------------------------")
    print(f"Number of npz files to read: {grain_npz_count}")
    print("-----------------------------------")
    if (grain_npz_count==0):
        raise ValueError(f"Zero data file!! Check your path to your data folder!!")
    # 1) Gather all n's first
    all_n = []
    for grain_file in grain_files:
        grain_path = os.path.join(data_path, grain_file)
        shape = np.load(grain_path)['arr_0'].shape  # careful if large
        n = shape[-1] - step_size - step_known - step_predict + 1
        all_n.append(n)
    total_samples = sum(all_n)
    # 2) Pre-allocate big arrays
    grain_known    = np.empty((total_samples, grid_size, step_known), dtype=np.float32)
    grain_predict  = np.empty((total_samples, grid_size, step_predict), dtype=np.float32)
    strain_rate    = np.empty((total_samples, grid_size, step_known), dtype=np.float32)
    temperature    = np.empty((total_samples, grid_size, step_known), dtype=np.float32)
    pressure       = np.empty((total_samples, grid_size, step_known), dtype=np.float32)
    Ss_pool = np.full((grain_npz_count), np.nan, dtype=np.float32)
    Ts_pool = np.full((grain_npz_count), np.nan, dtype=np.float32)
    Hs_pool = np.full((grain_npz_count), np.nan, dtype=np.float32)

    # 3) Loop again to fill in data slices
    offset = 0
    for i, grain_file in enumerate(grain_files):
        grain_path = os.path.join(data_path, grain_file)
        grain_data = np.load(grain_path)['arr_0']
        print(f"Read file: {grain_file}, shape: {grain_data.shape}, num samples: {all_n[i]}, value range: ",grain_data[0,:,:].max(),grain_data[0,:,:].min())
        n_i = all_n[i]
        Ss_pool[i] = grain_data[1,0,0]
        Hs_pool[i] = grain_data[2,0,0]+ 900*9.8
        Ts_pool[i] = grain_data[3,0,0]
        # Create these small arrays or slice directly
        for j in range(n_i):
            # --- Grain Known ---
            known_slice = grain_data[0, :,j : j + step_known].astype(np.float32)
            known_slice = reference_normalize_np(known_slice, min_ref=kde_ref_bound[0], max_ref=kde_ref_bound[1])
            grain_known[offset + j] = known_slice
            # --- Grain Predict ---
            predict_slice = grain_data[0, :, (j + step_size + step_known) : (j + step_size + step_known + step_predict)]
            predict_slice = predict_slice.astype(np.float32)
            predict_slice = reference_normalize_np(predict_slice, min_ref=kde_ref_bound[0], max_ref=kde_ref_bound[1])
            grain_predict[offset + j] = predict_slice
            # --- Strain Rate (log normalize) ---
            sr_slice = grain_data[1, :, j : (j + step_known)].astype(np.float32)
            sr_slice = log_normalize_np(sr_slice, min_ref=S_ref_bound[0], max_ref=S_ref_bound[1])
            strain_rate[offset + j] = sr_slice
            # --- Pressure + offset + reference normalize ---
            pr_slice = grain_data[2, :, j : (j + step_known)].astype(np.float32)
            pr_slice += 900.0 * 9.8  # add offset
            pr_slice = reference_normalize_np(pr_slice, min_ref=H_ref_bound[0], max_ref=H_ref_bound[1])
            pressure[offset + j] = pr_slice
            # --- Temperature: invert it, 1/T ---
            temp_slice = grain_data[3, :,j : (j + step_known)].astype(np.float32)
            # Important: watch out for zeros or very small T that cause inf
            # If needed: temp_slice = np.where(temp_slice == 0, np.finfo(np.float32).tiny, temp_slice)
            temp_slice = 1.0 / temp_slice
            temperature[offset + j] = temp_slice
        offset += n_i
   
    print(f"-------- Done reading all data from {grain_npz_count} files ---------")
    print("----------------------------------------------------")
    
    # stop if the normalized data is outside of [-1,1]
    if(np.min(grain_known)<-1 or np.max(grain_known)>1):
        raise ValueError(f"grain_known is outside [-1,1]. min = {np.min(grain_known)} max = {np.max(grain_known)}.Check reference min and max used to normalize the data.")
    elif(np.min(grain_predict)<-1 or np.max(grain_predict)>1):
        raise ValueError(f"grain_predict is outside [-1,1]. min = {np.min(grain_predict)} max = {np.max(grain_predict)}.Check reference min and max used to normalize the data.")
    elif(np.min(strain_rate)<-1 or np.max(strain_rate)>1):
        raise ValueError(f"strain_rate is outside [-1,1]. min = {np.min(strain_rate)} max = {np.max(strain_rate)}.Check reference min and max used to normalize the data.")
    elif(np.min(pressure)<-1 or np.max(pressure)>1):
        raise ValueError(f"pressure is outside [-1,1]. min = {np.min(pressure)} max = {np.max(pressure)}.Check reference min and max used to normalize the data.")
    elif(np.min(temperature)<-1 or np.max(temperature)>1):
        raise ValueError(f"temperature is outside [-1,1]. min = {np.min(temperature)} max = {np.max(temperature)}.Check reference min and max used to normalize the data.")
    
    # save parameters to data pool
    torch.save(Ss_pool, '../data/jcp_syn_grain_Ss_pool.pt')
    torch.save(Ts_pool,'../data/jcp_syn_grain_Ts_pool.pt')
    torch.save(Hs_pool, '../data/jcp_syn_grain_Hs_pool.pt')
    print(f'Saved normalized strain rate, temperature, and pressure data pool.')
    
    # split the data into training, validation and test sets in the ratio 80:10:10 randomly
    random_index = np.random.permutation(strain_rate.shape[0])
    train_index = random_index[0:int(0.8 * strain_rate.shape[0])]
    valid_index = random_index[int(0.8 * strain_rate.shape[0]):int(0.9 * strain_rate.shape[0])]
    test_index  = random_index[int(0.9 * strain_rate.shape[0]):]
    
    # train data
    grain_known_train   = grain_known[train_index, :, :]
    grain_predict_train = grain_predict[train_index, :, :]
    strain_rate_train  = strain_rate[train_index, :, :]
    temperature_train = temperature[train_index, :, :]
    pressure_train = pressure[train_index, :, :]

    # validation data
    grain_known_valid = grain_known[valid_index, :, :]
    grain_predict_valid = grain_predict[valid_index, :, :]
    strain_rate_valid = strain_rate[valid_index, :, :]
    temperature_valid = temperature[valid_index, :, :]
    pressure_valid = pressure[valid_index, :, :]

    # test data
    grain_known_test = grain_known[test_index, :, :]
    grain_predict_test = grain_predict[test_index, :, :]
    strain_rate_test = strain_rate[test_index, :, :]
    temperature_test = temperature[test_index, :, :]
    pressure_test = pressure[test_index, :, :]
    
    print("-----------------------------------------------------------------")
    print("Size of the data       :")
    print(f"{'Shape of grain_known_train'       :<30} : {grain_known_train.shape}   data range: [{grain_known_train.min():.2f}, {grain_known_train.max():.2f}]")
    print(f"{'Shape of grain_predict_train'     :<30} : {grain_predict_train.shape}   data range: [{grain_predict_train.min():.2f}, {grain_predict_train.max():.2f}]")
    print(f"{'Shape of strain_rate_train'        :<30} : {strain_rate_train.shape}   data range: [{strain_rate_train.min():.2f}, {strain_rate_train.max():.2f}]")
    print(f"{'Shape of temperature_train'       :<30} : {temperature_train.shape}   data range: [{temperature_train.min():.2f}, {temperature_train.max():.2f}]")
    print(f"{'Shape of pressure_train'          :<30} : {pressure_train.shape}   data range: [{pressure_train.min():.2f}, {pressure_train.max():.2f}]")
    print(f"{'Shape of grain_known_valid'       :<30} : {grain_known_valid.shape}   data range: [{grain_known_valid.min():.2f}, {grain_known_valid.max():.2f}]")
    print(f"{'Shape of grain_predict_valid'     :<30} : {grain_predict_valid.shape}   data range: [{grain_predict_valid.min():.2f}, {grain_predict_valid.max():.2f}]")
    print(f"{'Shape of strain_rate_valid'        :<30} : {strain_rate_valid.shape}   data range: [{strain_rate_valid.min():.2f}, {strain_rate_valid.max():.2f}]")
    print(f"{'Shape of temperature_valid'       :<30} : {temperature_valid.shape}   data range: [{temperature_valid.min():.2f}, {temperature_valid.max():.2f}]")
    print(f"{'Shape of pressure_valid'          :<30} : {pressure_valid.shape}   data range: [{pressure_valid.min():.2f}, {pressure_valid.max():.2f}]")
    print(f"{'Shape of grain_known_test'        :<30} : {grain_known_test.shape}   data range: [{grain_known_test.min():.2f}, {grain_known_test.max():.2f}]")
    print(f"{'Shape of grain_predict_test'      :<30} : {grain_predict_test.shape}   data range: [{grain_predict_test.min():.2f}, {grain_predict_test.max():.2f}]")
    print(f"{'Shape of strain_rate_test'         :<30} : {strain_rate_test.shape}   data range: [{strain_rate_test.min():.2f}, {strain_rate_test.max():.2f}]")
    print(f"{'Shape of temperature_test'        :<30} : {temperature_test.shape}   data range: [{temperature_test.min():.2f}, {temperature_test.max():.2f}]")
    print(f"{'Shape of pressure_test'           :<30} : {pressure_test.shape}   data range: [{pressure_test.min():.2f}, {pressure_test.max():.2f}]")
    print("-----------------------------------------------------------------")

    save_train_data_h5(save_traindata_path+'jcp_syn_grain_kde_test_data.h5',
                       grid_size,step_known,step_predict,test_index,
                       grain_known,grain_predict,strain_rate,temperature,pressure)
    save_train_data_h5(save_traindata_path+'jcp_syn_grain_kde_train_data.h5',
                       grid_size,step_known,step_predict,train_index,
                       grain_known,grain_predict,strain_rate,temperature,pressure)
    save_train_data_h5(save_traindata_path+'jcp_syn_grain_kde_valid_data.h5',
                       grid_size,step_known,step_predict,valid_index,
                       grain_known,grain_predict,strain_rate,temperature,pressure)

    # can start from here if train, valid, test data have been saved before
    train_dataset = H5Dataset(save_traindata_path+'jcp_syn_grain_kde_train_data.h5')
    valid_dataset = H5Dataset(save_traindata_path+'jcp_syn_grain_kde_valid_data.h5')
    for name in ["grain_known", "grain_predict", "strain_rate", "temperature", "pressure"]:
        summarize_hdf5_dataset(save_traindata_path+'jcp_syn_grain_kde_train_data.h5', name, chunk_size=512)
    for name in ["grain_known", "grain_predict", "strain_rate", "temperature", "pressure"]:
        summarize_hdf5_dataset(save_traindata_path+'jcp_syn_grain_kde_valid_data.h5', name, chunk_size=512)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=2  # > 0 to parallelize data loading
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=2
    )
    return train_loader, valid_loader, len(grain_files)


def train(model, train_loader, valid_loader, optimizer, scheduler, 
          myloss, costFunc, epoch, device, T_in, T_end, step, batch_size,smoothness_weight,
          patience_window,patience_ratio,band_tolerance,scheduler_gamma):

    print(f"\nThe model has {count_params(model)} trainable parameters\n")
    print(f"Training the model on {device} for {epoch} epoch ...\n")

    train_loss = torch.zeros(epoch)
    test_loss  = torch.zeros(epoch)
    # check_dataloader(train_loader)
    loss_history = []
    slope,count = 0.0,0.0
    for ep in range(epoch):
        model.train()
        t1 = default_timer()
        train_l2_step_training = 0
        test_l2_step_testing = 0
        for xx, yy, C1, C2, C3 in train_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            C1 = C1.to(device)
            C2 = C2.to(device)
            C3 = C3.to(device)
            # Pre-allocate memory for pred and xx
            for t in range(0, (T_end - T_in),step):
                # read in the time step to predict
                y = yy[..., t:t + step]
                im_train = model(xx,C1,C2,C3) # it calls __call__() in PyTorch which calls forward()
                smooth_penalty = smoothness_loss(im_train[:,:,0])
                loss +=  myloss(im_train.reshape(batch_size, -1), y.reshape(batch_size, -1))
                loss += smoothness_weight * smooth_penalty
                xx = torch.cat((xx[..., step:], im_train), dim=-1)
                # Check predictions
                if torch.isnan(im_train).any() or torch.isinf(im_train).any():
                    print(f"NaN/Inf in predictions at epoch {epoch}, step {t}")
                # end of one time slice prediction
                # -------
                   
            # do zeroing gradient and backward for each batch
            train_l2_step_training += loss.item()
            optimizer.zero_grad()
            loss.backward()
            total_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            total_norm_after = torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None))
            optimizer.step()
            train_loss[ep] = train_l2_step_training
        # end of each training batch

        with torch.no_grad():
            for xx, yy, C1, C2, C3 in valid_loader:
                loss = 0
                xx = xx.to(device)
                yy = yy.to(device)
                C1 = C1.to(device)
                C2 = C2.to(device)
                C3 = C3.to(device)
                for t in range(0, (T_end - T_in),step):
                    y = yy[..., t:t + step]
                    im_test = model(xx,C1,C2,C3)
                    smooth_penalty = smoothness_loss(im_test[:,:,0])
                    loss += myloss(im_test.reshape(batch_size, -1), y.reshape(batch_size, -1))
                    loss += smoothness_weight * smooth_penalty
                    xx = torch.cat((xx[..., step:], im_test), dim=-1)

                test_l2_step_testing += loss.item()
                test_loss[ep] = test_l2_step_testing

        t2 = default_timer()
        # scheduler.step()
        loss_history.append(test_l2_step_testing)
        if len(loss_history) > patience_window:
            loss_history.pop(0)
        # Only check if we have a "full window" 
        if len(loss_history) == patience_window:
            slope, _, _, _, _ = linregress(torch.arange(patience_window), torch.Tensor(loss_history))
            if np.abs(slope)<0.1:
                count += 1
                if count >=5:
                    for param_group in optimizer.param_groups:
                        old_lr = param_group["lr"]
                        if old_lr<=1e-4:
                            continue
                        new_lr = old_lr * scheduler_gamma
                        param_group["lr"] = new_lr
                        count = 0
        print(f"current epoch {ep}, time {t2 -t1:.2f}, train loss {train_l2_step_training:.4f}, valid loss: {test_l2_step_testing:.4f}, max norm before: {total_norm_before:.3f}, after: {total_norm_after:.3f}, learning rate: {optimizer.param_groups[0]['lr']}")
        
    return train_loss, test_loss


def main(data_path, save_path,save_testdata_path,save_traindata_path,model_dimension, epochs_input,converge_name):
    """ Train the model using the given data
    """

    # define the hyperparameters
    learning_rate   = 0.0001
    scheduler_step  = 30
    scheduler_gamma = 0.1
    smoothness_weight = 0.1
    patience_ratio = 0.9
    patience_window = 30
    band_tolerance = 0.15
    batch_size      = 50
    mode1           = 64
    mode2           = 12
    width           = 32
    activation_func = 'tanh'
    loss_func       = 'L2'
    batch_average = True

    epochs = epochs_input
    num_input_params = 4
    num_timesteps = 25
    grid_size = model_dimension

    step_known = 40       # num of steps whose info is known, i.e. step 1-3 are given
    step_predict = 160    # num of steps whose info is to perdict, i.e. step 4-10 are to be predict given 
    step_size = 0

    # define non-dimensionalization reference, first is low values, seccond is high value
    T_ref_bound = np.array([-26.0,-1.0])
    H_ref_bound = np.array([1*900*9.8, 1002*900*9.8])
    kde_ref_bound = np.array([0.0, 0.183])
    S_ref_bound = np.array([0.99999e-12, 1.60001e-8])

    # define the device for training (only on one GPU if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train_loader, valid_loader, num_npz_files = load_data(data_path,save_testdata_path,save_traindata_path,T_ref_bound, H_ref_bound,kde_ref_bound,S_ref_bound,
                                                      num_input_params,num_timesteps, grid_size, step_known, step_predict, 
                                                      step_size,batch_size,True)

    # define the model
    print("-----------------------------------------------------------------")
    print("\nTraining using activation function: ",activation_func, ", loss function: ",loss_func, ", loss average over batch size?: ",batch_average)
    print("\n Recurring property: steps known",step_known, ", steps to predict: ",step_predict)
    print(f"  Model parameters: mode1: {mode1}, mode2: {mode2}, width: {width}, batch size: {batch_size}, base learning rate: {learning_rate}")
    model = FNO1d(mode1,  width,step_known,activation_func,loss_func).to(device)
    # print(model,'\n',model.parameters())
    
    # define the optimizer and the scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    # define the loss function
    myloss = LpLoss(size_average=batch_average)
    costFunc = torch.nn.MSELoss(reduction='sum')

    # train the model
    train_loss, valid_loss = train(model, train_loader, valid_loader, 
                                  optimizer, scheduler, myloss, costFunc, 
                                  epochs, device, step_known, step_known+step_predict, 
                                  1, batch_size,smoothness_weight,
                                  patience_window,patience_ratio,band_tolerance,scheduler_gamma)
    
    torch.save(model.state_dict(), save_path+'_smooth'+str(smoothness_weight)+'_N'+str(num_npz_files)+'_epoch'+str(epochs)+'.pth') #
    torch.save(train_loss, save_path + '_smooth'+str(smoothness_weight)+'_N'+str(num_npz_files)+'_epoch'+str(epochs)+'_train_loss.pt')
    torch.save(valid_loss, save_path + '_smooth'+str(smoothness_weight)+'_N'+str(num_npz_files)+'_epoch'+str(epochs)+'_valid_loss.pt') 
    print("Model saved to: ",save_path)
    
    # Plot the convergence curve
    fig, ax1 = plt.subplots(figsize=(10, 6),dpi=300)
    ax1.semilogy(valid_loss, 'b-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)
    # Create a second y-axis for the validation loss
    ax2 = ax1.twinx()
    ax2.semilogy(train_loss, 'r-', label='Training Loss')
    ax2.set_ylabel('Training Loss', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    fig.suptitle(f'Train and Valid Loss Convergence, N{num_npz_files}, {activation_func}, {loss_func}')
    fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.savefig(f'{converge_name}_epoch{epochs_input}_N{num_npz_files}_smooth{smoothness_weight}.png')
    plt.close() 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--data_path', type=str, default='/elle_grainsize_results/synthetic/data_for_FNO_training/train_data/', help='Path to data')
    parser.add_argument('--save_path', type=str, default='./../model/model', help='Path to saved model')
    parser.add_argument('--save_testdata_path', type=str, default='/elle_grainsize_results/jcp/data/test/', help='Path to saved test data')
    parser.add_argument('--save_traindata_path', type=str, default='/elle_grainsize_results/jcp/data/train_valid/', help='Path to saved train data')
    parser.add_argument('--model_dimension',type=int,default=256,help='Number of grid points in x or y. Currently only support square matrix.')
    parser.add_argument('--epochs',type=int,default=20,help='Num of epochs')
    parser.add_argument('--converge_name',type=str,default='jcp_grain_loss_convergence',help='Name for the convergence plot')
    args = parser.parse_args()

    main(args.data_path,args.save_path,args.save_testdata_path,args.save_traindata_path,args.model_dimension,args.epochs,args.converge_name)
