import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sqrt, exp
import time
import argparse

import sys
sys.path.append('../src/')
data_path = '/elle_grainsize_results/data_for_FNO_training/train_data/'

import torch
import statsmodels.api as sm
from model_grain import FNO2d
from model_grain_kde import FNO1d
from utilities3_grain import generate_unique_color
from utilities3_grain import reinitialize,reinitialize_test
from utilities3_grain import analyze_flynns
from utilities3_grain import log_normalize, reference_normalize
from utilities3_euler import clean_clustered_array_torch
from utilities3_euler import calculate_local_and_global_eigenvalues
from utilities3_euler import f1, f2, woodcock
from utilities3_euler import euler_to_orientation_tensor, get_orient_tensor2D

# Constants
S2Y = 3600 * 24 * 365.25
S2D = 3600.0 * 24
PI = np.pi
plt.rcParams.update({'font.size': 12})

################## Function Definitions ##################
def compute_rheol_Tau(η, η_gbs, η_disl, d, Tii2, T, P, ph, R,w13,fact):
    T_h = T + ph * P
    Tii = np.sqrt(Tii2)

    # Pre-compute common values
    m_gbs, n_gbs, n_disl = 1.4, 1.8, 4.0
    F_gbs = (2**((n_gbs - 1) / n_gbs) * 3**((n_gbs + 1) / (2 * n_gbs)))**-1
    F_disl = (2**((n_disl - 1) / n_disl) * 3**((n_disl + 1) / (2 * n_disl)))**-1
    
    # GBS parameters
    A1_gbs, Q1_gbs, Q2_gbs, Tstar_gbs = 3.9e-3 * 10**(-6 * n_gbs), 49e3, 192e3, 259
    
    # Dislocation parameters
    A1_disl, Q1_disl, Q2_disl, Tstar_disl = 4e4 * 10**(-6 * n_disl), 60e3, 180e3, 259

    # Vectorized operations
    Tstar_gbs_h = Tstar_gbs + ph * P
    Tstar_disl_h = Tstar_disl + ph * P
    
    Q_gbs_h = np.where(T_h <= Tstar_gbs_h, Q1_gbs, Q2_gbs)
    Q_disl_h = np.where(T_h <= Tstar_disl_h, Q1_disl, Q2_disl)

    As_gbs = A1_gbs * np.exp(-Q1_gbs / (R * Tstar_gbs_h))
    As_disl = A1_disl * np.exp(-Q1_disl / (R * Tstar_disl_h))

    B_gbs = F_gbs * As_gbs**-1 * np.exp((Q_gbs_h / R) * ((1 / T_h) - (1 / Tstar_gbs_h)))
    B_disl = F_disl * As_disl**-1 * np.exp((Q_disl_h / R) * ((1 / T_h) - (1 / Tstar_disl_h)))

    η_gbs[:] = fact**(n_gbs - 1) * B_gbs * Tii**(1 - n_gbs) * d**m_gbs
    η_disl[:] = fact**(n_disl - 1) * B_disl * Tii**(1 - n_disl)

    # Combined viscosity
    η[:] = 1.0 / (1.0 / η_gbs + 1.0 / η_disl)
    return η, m_gbs, n_gbs, n_disl

def saveResults2NPZ(Vxe, P, T, epsxz, etav, d, grain_kde, zv, 
                        euler1_all,euler2_all,euler3_all,
                        eigenvalues, w13,time_step, filepath, filename):
    data = {
        "Vxe": Vxe,
        "Pressure": P,
        "Temperature": T,
        "Strain_rate": epsxz,
        "viscosity": etav,
        "Grain_size": d,
        "Depth": zv,
        "Grain_kde":grain_kde,
        "eigenvalues":eigenvalues,
        "weaken_factor":w13,
        "Euler1_2d":euler1_all,
        "Euler2_2d":euler2_all,
        "Euler3_2d":euler3_all
    }
    np.savez(f"{filepath}{filename}", **data)

def plot_convergence(data, time_step,filepath):
    """Plot convergence for a single time step."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('err', color=color1)
    ax1.semilogy(data['err_values'], color=color1, label='err')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()  
    color2 = 'tab:orange'
    ax2.set_ylabel('errd', color=color2)
    ax2.semilogy(data['errd_values'], color=color2, label='errd')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title(f'Convergence for Time Step {time_step}')
    fig.tight_layout()  
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.savefig(f"{filepath}jcp_convergence_step_{time_step}.png")
    plt.close()
########################################################################


@torch.no_grad()
def ice_column1D(grain_model_epoch,grain_model_path,euler_model_epoch,euler_model_path,save_data_path,save_fig_path):
    # Constants
    S2Y = 3600 * 24 * 365.25
    S2D = 3600 * 24
    PI = np.pi

    # Parameters
    steady = False
    MULTISCALE = True
    fact = 1.0
    H = 1000.0
    alpha = 0.8
    alpha_rad = alpha * PI / 180
    rho = 900
    g = -9.8
    eta0 = 1e15
    T_surf = 247.15 # -26
    T_bed = 272.15 # 273.15 - 0
    T_K = 273.15
    R = 8.314
    Q0 = 6 * 1e4
    a0 = 8.75 * 1e-23
    kappa = 2.51
    cp = 2096.9
    aniso_strenght = 10
    npow = 3.0
    mpow = -(1.0 - 1.0 / npow) / 2.0
    P0 = 1e7 * 0
    ph = 7e-8
    dt = 1.2 * S2D
    nt = 2000
    c = 3.0
    eps_bg = 1e-12
    nz = 63
    rel = 1e-3
    tol = 1e-10
    iterMax = 1e5+1
    nout = iterMax - 1
    dz = H / nz
    damp = 1.0 - 0.1 / nz
    dtaudT  = 1 / (1/(dz**2 / kappa * rho * cp)/4.1 + 1/dt)
    
    # Move models to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # device should be cuda
    
    # ----------- FNO kde set up ----------#
    step_known      = 40
    kde_reso        = 256
    # IMPORTANT: check .out to see what reference max and min were used in training
    kde_max, kde_min = 0.183, 0
    H_max, H_min = 1001*9.8*900, 1*9.8*900
    S_max, S_min = 1.6e-8,1e-12
    # the model parameter MUST be the same as the training
    mode1           = 64
    width           = 32
    activation_func = 'tanh'
    loss_func       = 'L2'
    grain_model = FNO1d(mode1, width,step_known,activation_func,loss_func)
    grain_model.load_state_dict(torch.load(grain_model_path+str(grain_model_epoch)+'.pth'))
    grain_model = grain_model.to(device)
    
    grains_in = np.zeros((nz + 1,kde_reso,step_known))
    # load initial condition for small scale simulations. This is pre-saved using the scripts in grain_euler_process notebook in postprocess
    init_kde =  torch.load('/elle_grainsize_results/jcp/data/syn_init_kde.pt').to(device)
    init_strain =  torch.load('/elle_grainsize_results/jcp/data/syn_init_strain.pt').to(device)
    init_pressure =  torch.load('/elle_grainsize_results/jcp/data/syn_init_pressure.pt').to(device)
    init_temperature =  torch.load('/elle_grainsize_results/jcp/data/syn_init_temperature.pt').to(device)
    kde_in  = reference_normalize(init_kde[:,:,:step_known], kde_max, kde_min)
    strain_in = log_normalize(init_strain[:,:,:step_known],S_max, S_min)
    pressu_in = reference_normalize(init_pressure[:,:,:step_known],H_max, H_min)
    temper_in = 1/init_temperature[:,:,:step_known]
    
    # grain size range (this is pre-defined)
    area_per_pixel = 0.03*0.03/128/128
    min_grainsize,max_grainsize = 3, 800
    grainsize_range = torch.linspace(min_grainsize,max_grainsize, kde_reso).to(device)*area_per_pixel
    # Reshape grainsize_range to shape (1, 256, 1) for broadcasting
    grainsize_range = grainsize_range.view(1, -1, 1)
    max_d = torch.sqrt(grainsize_range.max()/PI)*2*0.6
    # calcualte initial grain size
    kde_temp = (kde_in+1)/2 * (kde_max - kde_min) + kde_min
    kde = kde_temp / kde_temp.sum(axis=1, keepdims=True) # kde shape [nz+1, kde reso, step known]
    d = torch.sqrt(torch.sum(grainsize_range * kde, dim=1)/PI)*2 # d shape [nz+1, step known]
    d = d[:,:step_known]
    d_history = d

    # ----------- FNO 2D set up ----------#
    grid_size       = 128
    mode1           = 12
    mode2           = 12
    width           = 16
    activation_func = 'tanh'
    loss_func       = 'L2'
    euler1_model = FNO2d(mode1, mode2, width,step_known,activation_func,loss_func)
    euler1_model.load_state_dict(torch.load(euler_model_path+str(euler_model_epoch)+'.pth'))
    euler2_model = FNO2d(mode1, mode2, width,step_known,activation_func,loss_func)
    euler2_model.load_state_dict(torch.load(euler_model_path.replace('euler1', 'euler2')+str(euler_model_epoch)+'.pth'))
    euler3_model = FNO2d(mode1, mode2, width,step_known,activation_func,loss_func)
    euler3_model.load_state_dict(torch.load(euler_model_path.replace('euler1', 'euler3')+str(euler_model_epoch)+'.pth'))
    euler1_model = euler1_model.to(device)
    euler2_model = euler2_model.to(device)
    euler3_model = euler3_model.to(device)
    
    euler1_in = np.zeros((nz + 1,grid_size,grid_size,step_known))
    euler2_in = np.zeros((nz + 1,grid_size,grid_size,step_known))
    euler3_in = np.zeros((nz + 1,grid_size,grid_size,step_known))
    init_euler1 = torch.load('/elle_grainsize_results/jcp/data/syn_init_euler_1.pt').to(device)
    init_euler2 = torch.load('/elle_grainsize_results/jcp/data/syn_init_euler_2.pt').to(device)
    init_euler3 = torch.load('/elle_grainsize_results/jcp/data/syn_init_euler_3.pt').to(device)
    init_strain2D = torch.load('/elle_grainsize_results/jcp/data/syn_init_strain2D.pt').to(device)
    init_pressure2D = torch.load('/elle_grainsize_results/jcp/data/syn_init_pressure2D.pt').to(device)
    init_temperature2D = torch.load('/elle_grainsize_results/jcp/data/syn_init_temperature2D.pt').to(device)
    euler1_in = init_euler1[:,:,:,:step_known]/180
    euler2_in = (init_euler2[:,:,:,:step_known]-45)/45
    euler3_in = init_euler3[:,:,:,:step_known]/180
    strain2D_in = log_normalize(init_strain2D[:,:,:,:step_known],S_max, S_min)
    pressu2D_in = reference_normalize(init_pressure2D[:,:,:,:step_known],H_max, H_min)
    temper2D_in = 1/init_temperature2D[:,:,:,:step_known]

    # repeat to make 64 grains in to 128 if needed
    grains_in = torch.repeat_interleave(kde_in, repeats=(nz+1)//64, dim=0)
    euler1_in = torch.repeat_interleave(euler1_in, repeats=(nz+1)//64, dim=0)
    euler2_in = torch.repeat_interleave(euler2_in, repeats=(nz+1)//64, dim=0)
    euler3_in = torch.repeat_interleave(euler3_in, repeats=(nz+1)//64, dim=0)
    w13 = np.ones(nz+1) # weakening factor
    eigenvalues=torch.ones((nz+1,3))
    # ----------- FNO set up done ----------#

    # Array initialization
    d_o = np.ones((nz+1,step_known))
    Vxe = np.zeros(nz + 2)
    epsxz = np.zeros(nz + 1)
    taoxz = np.zeros(nz + 1)
    Eii2 = np.zeros(nz + 1)
    Tii2 = np.ones(nz + 1)
    dVxdtau = np.zeros(nz)
    dtauVx = np.zeros(nz)
    Fx = np.zeros(nz)
    eta_gbs = np.zeros(nz + 1)
    eta_disl = np.zeros(nz + 1)
    err_values = np.empty((nt+1,int(iterMax)))
    errT_values = np.empty((nt+1,int(iterMax)))
    errd_values = np.empty(nt+1)

    # Initial conditions
    zc = np.linspace(-H + dz / 2, -dz / 2, nz)
    zv = np.linspace(-H, 0, nz + 1)
    zv2 = np.linspace(-H, 0, nz + 2)
    etav = eta0 * np.ones(nz + 1)
    etap = eta0 * np.ones(nz + 1)
    a = T_bed/200
    b =  30/H*np.log(T_surf/T_bed)
    c = np.log(1/a*(T_bed-T_surf)/(np.exp(-b*H)-1))
    T = a*np.exp(b*zv+c)+273.15-27.5 #np.linspace(T_pool, T_surf, nz + 1) # 1.36075*np.exp(-0.0050456148658740195*zv-0.6747875463067662)+273.15-24
    # T = np.linspace(T_bed,T_surf,nz+1)
    T0,Tend = T[0],T[-1]
    qzT = kappa * np.diff(np.diff(T) / dz)/dz
    dTdt = (qzT)/rho/cp
    P = rho * g * zv #9.8*900*180.88*np.ones(nz+1) # 
    
    # FNO data base
    Ss_pool = torch.load('../data/jcp_syn_grain_Ss_pool.pt')#.reshape(-1, 1).to(device) # Shape (64, 1)
    Hs_pool = torch.load('../data/jcp_syn_grain_Hs_pool.pt')#.reshape(-1, 1).to(device) # Shape (64, 1)
    Ts_pool = torch.load('../data/jcp_syn_grain_Ts_pool.pt')#.reshape(-1, 1).to(device) # Shape (64, 1)
    S_max_clamp, S_min_clamp = Ss_pool.max(), Ss_pool.min()
    H_max_clamp, H_min_clamp = Hs_pool.max(), Hs_pool.min()
    T_max_clamp, T_min_clamp = Ts_pool.max(), Ts_pool.min()               
    print(f'S pool range: [{S_min_clamp},{S_max_clamp}]')
    print(f'H pool range: [{H_min_clamp/900/9.8},{H_max_clamp/900/9.8}]')
    print(f'T pool range: [{T_min_clamp},{T_max_clamp}]')

    
    # Time loop
    if MULTISCALE:
        filename = f"jcp_ice1D_kde+euler_alpha{alpha}_reso{nz}_step0"
    else:
        filename = f"jcp_ice1D_ref_alpha{alpha}_reso{nz}_step0"
    saveResults2NPZ(Vxe, P, T, epsxz, etav, d.cpu().numpy(), grains_in.cpu().numpy(), zv, 
                            euler1_in.cpu().numpy(),
                            euler2_in.cpu().numpy(), 
                            euler3_in.cpu().numpy(), 
                            eigenvalues, w13,
                            0, save_data_path, filename)
    
    print(f'---------------------------------------------------------------------')
    print(f'---- FNO related ----')
    print(f'grain kde model: {grain_model_path}{grain_model_epoch}.pth')
    print(f'euler angle model: {euler_model_path}{euler_model_epoch}.pth and others')
    print(f"FNO inputs shapes: grains_in: {grains_in.shape}, euler: {euler1_in.shape}, strain_2D: {strain2D_in.shape}")
    print(f'starting weakening factor w13: {w13}')
    print(f"  euler 1 range: [{euler1_in.min()},{euler1_in.max()}]")
    print(f"  euler 2 range: [{euler2_in.min()},{euler2_in.max()}]")
    print(f"  euler 3 range: [{euler3_in.min()},{euler3_in.max()}]")
    print(f"  grain range  : [{grains_in.min()},{grains_in.max()}]")
    print(f"  max d to cap off: {max_d}")
    print(f'Data pool of size: {Ss_pool.shape}')
    print(f'---- flow model related ----')
    # print(f"starting T: {T-273.15}")
    print(f'num of time steps: {nt}. slope: {alpha}')
    print(f'ice thickness H: {H}. Num grid points: {nz}. dz: {dz}')
    print(f'temperature dt: {dtaudT}')
    print(f'Num of iter per step: {iterMax}. threshold: {tol}')
    print(f'---------------------------------------------------------------------')

    fig,ax = plt.subplots(3,4,figsize=(14, 10),dpi=300)
    start_time = time.time()
    for time_step in range(1, nt + 1):
        d_o, T_o, dTdt_o = d.clone(), T.copy(), dTdt.copy()
        iter = 1
        err = 2 * tol
        errd = 2* err
        while (err > tol) and iter < iterMax:
            epsxz = 0.5 * np.diff(Vxe) / dz + eps_bg
            etap, m_gbs, n_gbs, n_disl = compute_rheol_Tau(etap, eta_gbs, eta_disl, d[:,-1].cpu().numpy(), Tii2, T, P, ph, R, w13, fact)
            etav = rel * etap + (1 - rel) * etav
            taoxz = w13 * fact * etav * epsxz
            Tii2 = taoxz**2
            Fx = np.diff(taoxz) / dz - rho * g * np.sin(alpha_rad)
            dVxdtau = 0.5 * dVxdtau + 0.5 * (Fx + dVxdtau * damp)
            dtauVx = dz**2 / (0.5 * (etav[:-1] + etav[1:]))
            Vxe[1:-1] += dtauVx * dVxdtau
            # temperature equation
            qzT = kappa * np.diff(np.diff(T) / dz)/dz
            dTdt = (qzT + 2*taoxz[1:-1]*epsxz[1:-1])/rho/cp
            dTdtau = -(T[1:-1] - T_o[1:-1]) / dt + 0.5 * dTdt + 0.5 * dTdt_o
            T[1:-1] += dtaudT*dTdtau
            # BC
            Vxe[0], Vxe[-1] = -Vxe[1], Vxe[-2]
            T[0], T[-1] = T0,Tend

            err = np.linalg.norm(Fx) / nz
            err_values[time_step,iter] = err
            errT_values[time_step,iter] = np.linalg.norm(dTdtau)/nz
            if iter % nout == 0:
                iteration_time = time.time() - start_time
                print(f"time_step={time_step}, iter={iter}, err={err:.2e}, errT={errT_values[time_step,iter]:.2e}, "
                      f"velocity={Vxe[-1]*S2Y:.2e} m/yr,strain_rate={epsxz[1]*S2Y:.2e} 1/yr, viscosity={etav[0]:.2e} PaS, grain_size_range=[{d.min()*1e3:.1f}, {d.max()*1e3:.1f}] mm, "
                      f"time used={iteration_time/60:.3e} min")
                start_time = time.time()
            iter += 1
            # ------ end of one iteration ------ #

         # update FNO
        if (MULTISCALE):
            S_new = torch.tensor(epsxz.reshape(1, -1), device=device).float()
            H_new = torch.tensor(P.reshape(1, -1), device=device).float()
            T_new = torch.tensor(T.reshape(1, -1), device=device).float() - T_K
            S_new = torch.clamp(S_new, min=S_min_clamp, max=S_max_clamp).squeeze()
            H_new = torch.clamp(H_new, min=H_min_clamp, max=H_max_clamp).squeeze()
            T_new = torch.clamp(T_new, min=T_min_clamp, max=T_max_clamp).squeeze()
            S_new = log_normalize(S_new,S_max, S_min).to(device)  # Shape (64,1)
            H_new = reference_normalize(H_new,H_max, H_min).to(device)  # Shape (64,1)
            T_new = 1/T_new; # Shape (64,1))
            # set the parameters outside of training space to the max min of the training space
            strain_in = S_new[:, np.newaxis, np.newaxis].expand(nz+1, kde_reso, step_known)
            temper_in = T_new[:, np.newaxis, np.newaxis].expand(nz+1, kde_reso, step_known)
            pressu_in = H_new[:, np.newaxis, np.newaxis].expand(nz+1, kde_reso, step_known)
            grains_pred  = grain_model(grains_in, strain_in, temper_in,pressu_in)
            grains_in = torch.cat((grains_in[..., 1:], grains_pred), dim=-1)
            # de-normalize kde
            # IMPORTANT: check .out to see what reference max and min were used in training
            kde = (grains_pred+1)/2 * (kde_max - kde_min) + kde_min # de-normalization is correct
            kde = kde / kde.sum(axis=1, keepdims=True) # kde shape [nz+1, kde reso, step known]
            d =  torch.sqrt(torch.sum(grainsize_range * kde, dim=1)/PI)*2 # d shape [nz+1, step known]
            d[d>=max_d] = max_d
            d_history = torch.cat((d_history,d[:,0].unsqueeze(-1)), dim=-1)
                 
            strain2D_in = S_new[:, np.newaxis, np.newaxis, np.newaxis].expand(nz+1, grid_size,grid_size, step_known)
            temper2D_in = T_new[:, np.newaxis, np.newaxis, np.newaxis].expand(nz+1, grid_size,grid_size, step_known)
            pressu2D_in = H_new[:, np.newaxis, np.newaxis, np.newaxis].expand(nz+1, grid_size,grid_size, step_known)
            euler1_pred = euler1_model(euler1_in, strain2D_in, temper2D_in,pressu2D_in) # euler1_pred shape [nz+1,grid,grid,1]
            euler2_pred = euler2_model(euler2_in, strain2D_in, temper2D_in,pressu2D_in)
            euler3_pred = euler3_model(euler3_in, strain2D_in, temper2D_in,pressu2D_in)
            euler1_in = torch.cat((euler1_in[..., 1:], euler1_pred), dim=-1)
            euler2_in = torch.cat((euler2_in[..., 1:], euler2_pred), dim=-1)
            euler3_in = torch.cat((euler3_in[..., 1:], euler3_pred), dim=-1)
            
            for i in range(nz+1):
                angle1 = euler1_pred[i,:,:,0].flatten()
                angle2 = euler2_pred[i,:,:,0].flatten()
                angle3 = euler3_pred[i,:,:,0].flatten()
                eigenvalues[i,:], tensor = euler_to_orientation_tensor(angle1.to('cpu')*180, angle2.to('cpu')*48+48, angle3.to('cpu')*180)
                w13[i] = 0.5*tensor[2,2]
            print(f'grain size: {d[:,-1]*1e3}')
            print(f'w13: {w13}')
            print(f'T: {T-T_K}')
            errd = torch.abs(d.mean() - d_o.mean())/d_o.mean()
            errd_values[time_step]=errd
                
        if MULTISCALE:
            filename = f"jcp_ice1D_kde+euler_alpha{alpha}_reso{nz}_step{time_step}"
        else:
            filename = f"jcp_ice1D_ref_alpha{alpha}_reso{nz}_step{time_step}"
        
        saveResults2NPZ(Vxe, P, T, epsxz, etav, d.cpu().numpy(), grains_in.cpu().numpy(), zv, 
                            euler1_in.cpu().numpy(),
                            euler2_in.cpu().numpy(), 
                            euler3_in.cpu().numpy(), 
                        eigenvalues.cpu().numpy(),w13,
                        time_step, save_data_path, filename)
        print(f"\nFinished step {time_step}. Results saved to {save_data_path}{filename}\n")
        # ------ end of one time step ------ #
        # ---------------------------------- #

    # plot_convergence(convergence_data[-1], 1, save_fig_path)
    np.savez_compressed(save_data_path+'test_jcp_stokes1D_convergence.npy',err_values=err_values,errT_values=errT_values,errd_values=errd_values)
    print(f'convergence data saved as {save_data_path}test_jcp_stokes1D_convergence.npy')
    # plotting
    fig, axs = plt.subplots(2, 2, figsize=(15,20))
    print(f"d history shape: {d_history.shape}")
    axs[0,0].plot(d_history.cpu().numpy()*1e3,zv,linewidth=1.5)
    axs[0,0].set_xlabel("Grain size [mm]")
    axs[0,0].set_ylabel("Depth [m]")
    axs[0,1].plot(Vxe[1:]*S2Y,zv,linewidth=1.5)
    axs[0,1].set_xlabel("Velocity [m/yr]")
    axs[0,1].set_ylabel("Depth [m]")
    axs[1,0].plot(epsxz*S2Y,zv,linewidth=1.5)
    axs[1,0].set_xlabel("strain rate [1/yr]")
    axs[1,0].set_ylabel("Depth [m]")
    axs[1,1].plot(etav/1e6,zv,linewidth=1.5) # etav/1e6
    axs[1,1].set_xlabel("Viscosity [M Pa]")
    axs[1,1].set_ylabel("Depth [m]")
    plt.tight_layout()
    plt.savefig(f"./../results/figures/jcp_syn_1D_results.png")
    print(f"Figure ./../results/figures/jcp_syn_1D_results.png saved.")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stokes 1D model for synthetic kde')
    parser.add_argument('--grain_model_epoch', type=int, default=1000, help='number of epoch')
    parser.add_argument('--grain_model_path', type=str, default='./../model/jcp_grain_kde_mode64_width32_tanh+L2_custom_smooth10_N67_epoch', help='Path to trained grain kde model')
    parser.add_argument('--euler_model_epoch', type=int, default=500, help='number of epoch')
    parser.add_argument('--euler_model_path', type=str, default='./../model/jcp_syn_euler1_mode12_width16_tanh+L2_custom_N64_epoch', help='Path to trained euler model')
    parser.add_argument('--save_fig_path', type=str, default='./../results/figures/', help='Path to saved test data')
    parser.add_argument('--save_data_path', type=str, default='/elle_grainsize_results/ElleFNO_results/syn_1D/', help='Path to saved train data')
    parser.add_argument('--init_grain_data',type=str,default=20,help='Num of epochs')
    parser.add_argument('--init_euler_data',type=str,default='test_grain_loss_convergence',help='Name for the convergence plot')
    args = parser.parse_args()

    ice_column1D(args.grain_model_epoch,args.grain_model_path,args.euler_model_epoch,args.euler_model_path,args.save_data_path,args.save_fig_path)