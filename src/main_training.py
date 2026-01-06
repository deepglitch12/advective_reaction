from torch.utils.data import DataLoader
from datasets import PipeDatasetFromFiles
from pinn_model import HeatPINN
import torch as tt
import torch.nn as nn
import torch.nn.functional as ff
import numpy as np
from time import time
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
 
def sample_Xr(nr, T_max=500.0, T_min=273.0, v_max=1.0, v_min=0.0, alpha_max=3e-3, alpha_min=1e-6, L=1000, tf=3600):
    t = tt.rand(nr, 1, device=device) * (tf - 0.1) + 0.1
    x = tt.rand(nr, 1, device=device) * (L - 0.1) + 0.1

    # random parameters (same ranges as dataset)
    T_in = tt.rand(nr, 1, device=device) * (T_max - T_min) + T_min
    v = tt.rand(nr, 1, device=device) * (v_max - v_min) + v_min
    alpha = tt.rand(nr, 1, device=device) * (alpha_max - alpha_min) + alpha_min

    return tt.cat([t, x, T_in, v, alpha], dim=1)

def sample_Xe(ne, Tinf=293.0, T_max=500.0, T_min=273.0, v_max=1.0, v_min=0.0, alpha_max=3e-3, alpha_min=1e-6, L=1000, tf=3600):

    # Initial condition: t = 0
    t0 = tt.zeros(ne // 2, 1, device=device)
    x0 = tt.rand(ne // 2, 1, device=device) * (L - 0.01) + 0.01

    T_in0 = tt.rand(ne // 2, 1, device=device) * (T_max - T_min) + T_min
    v0 = tt.rand(ne // 2, 1, device=device) * (v_max - v_min) + v_min
    alpha0 = tt.rand(ne // 2, 1, device=device) * (alpha_max - alpha_min) + alpha_min

    X_init = tt.cat([t0, x0, T_in0, v0, alpha0], dim=1)

    # Boundary condition: x = 0
    tb = tt.rand(ne // 2, 1, device=device) * tf
    xb = tt.zeros(ne // 2, 1, device=device)

    T_inb = tt.rand(ne // 2, 1, device=device) * (T_max - T_min) + T_min
    vb = tt.rand(ne // 2, 1, device=device) * (v_max - v_min) + v_min
    alphab = tt.rand(ne // 2, 1, device=device) * (alpha_max - alpha_min) + alpha_min

    X_bc = tt.cat([tb, xb, T_inb, vb, alphab], dim=1)

    return tt.vstack([X_init, X_bc])


def normalize_inputs(X_in, T_max=500.0, T_min=273.0, v_max=1.0, v_min=0.0, alpha_max=3e-3, alpha_min=1e-6, L=1000, tf=3600):

    # Define Lower and Upper bounds for all 5 columns
    lb = tt.tensor([0.0, 0.0, T_min, v_min, alpha_min], device=device)
    ub = tt.tensor([tf,  L,   T_max, v_max, alpha_max], device=device)
    
    # Formula: 2 * (x - lb) / (ub - lb) - 1
    X_norm = 2.0 * (X_in - lb) / (ub - lb) - 1.0
    return X_norm

def pde_sobolev_loss(Xr_org, Tinf = 293.0, T_max=500.0, T_min=273.0, L=1000, T=3600):

    A_norm = np.array(
        [
            2.0/(T - 0.0),
            2.0/(L - 0.0),
        ]
    )
    A = tt.tensor(A_norm, device=device, dtype=tt.float32).reshape(-1,1)

    Xr = normalize_inputs(Xr_org)
    Xr.requires_grad_(True)

    T_out = pINN(Xr)

    mult = (T_max - T_min)/2.0

    dT_dX = tt.autograd.grad(T_out, Xr, tt.ones_like(T_out), create_graph=True)[0]
    dT_dt = dT_dX[:,0] * A[0] * mult
    dT_dx = dT_dX[:,1] * A[1] * mult

    T_NN = mult*T_out + (T_max + T_min)/2.0

    F = dT_dt + Xr_org[:,3]*dT_dx + Xr_org[:,4]*(T_NN - Tinf)

    sobo_loss = (
        (dT_dt / (T_max - T_min))**2 +
        (L * dT_dx / (T_max - T_min))**2
    ).mean()


    return (F**2).mean(), sobo_loss

def const_loss(Xe_org, Tinf = 293.0, T_max=500.0, T_min=273.0):

    Xe = normalize_inputs(Xe_org)
    T_out = pINN(Xe)

    # De-normalize temperature
    T_NN = 0.5 * (T_max - T_min) * T_out + 0.5 * (T_max + T_min)

    # Masks
    init_mask = Xe[:, 0] == -1
    bc_mask   = Xe[:, 1] == -1

    # Initial condition loss
    init_loss = (T_NN[init_mask] - Tinf) ** 2
    init_loss = init_loss.mean() if init_loss.numel() > 0 else 0.0

    # Boundary condition loss
    bc_loss = (T_NN[bc_mask] - Xe_org[bc_mask, 2]) ** 2
    bc_loss = bc_loss.mean() if bc_loss.numel() > 0 else 0.0

    return init_loss + bc_loss

def observation_loss(Xo_org, To_true, T_max=500.0, T_min=273.0):

    Xo = normalize_inputs(Xo_org)
    T_pred = pINN(Xo)

    T_pred = ((T_max - T_min) / 2.0) * T_pred + (T_max + T_min) / 2.0
    return ff.mse_loss(T_pred, To_true)

def step():

    global data_iter

    pINN.train()

    optimizer.zero_grad()

    try:
        Xo, To = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        Xo, To = next(data_iter)

    
    obs_loss = observation_loss(Xo, To)

    Xr = sample_Xr(nr)
    pde_loss, sobo_loss = pde_sobolev_loss(Xr)

    Xe = sample_Xe(ne)
    bc_loss = const_loss(Xe)

    lambda_obs = 10.0
    lambda_pde = 1.0
    lambda_bc  = 5.0
    lambda_t   = 0.0

    total_loss = (
        lambda_obs * obs_loss +
        lambda_pde * pde_loss +
        lambda_bc * bc_loss   +
        lambda_t * sobo_loss
    )

    total_loss.backward()

    # tt.nn.utils.clip_grad_norm_(pINN.parameters(), max_norm=1.0)
    
    optimizer.step()

    log["total_loss"].append(total_loss.item())
    log["pde_loss"].append(pde_loss.item())
    log["bc_loss"].append(bc_loss.item())
    log["obs_loss"].append(obs_loss.item())
    log["sobo"].append(sobo_loss.item())

if __name__ == '__main__':

    device = 'cpu'
    use_existing_model = 1

    if use_existing_model == 1:
        pINN = HeatPINN().to(device)
        pINN.load_state_dict(tt.load('out/best_model.pt', map_location=device))
        print("Model Loaded Sucessfully!!")
        learning_rate = 1e-3
    else:
        learning_rate = 1e-2
        pINN = HeatPINN().to(device)

    lambda_ridge = 1e-5

    optimizer = tt.optim.Adam(pINN.parameters(), lr = learning_rate, weight_decay=lambda_ridge) 

    path_in_dir_script = Path(__file__).parent  #foldwe where the main script is
    path_out_dir = path_in_dir_script / "../src"
    path_out_dir.mkdir(exist_ok=True)

    # Logging dict
    log = dict(
        epoch="",
        total_loss = [],
        pde_loss = [],
        bc_loss = [],
        obs_loss = [],
        sobo = [],
        saved_weights=[],
        device=device,
    )

    n  = 5000
    ne = 4000
    nr = 5000

    dataset = PipeDatasetFromFiles(
    dataset_dir="./data/datasets",
    device=device
    )

    loader = DataLoader(
        dataset,
        batch_size=n,
        shuffle=True,
    )

    data_iter = iter(loader)

    epochs = 20000
    save_dir = "out"
    os.makedirs(save_dir, exist_ok=True)

    best_loss = float('inf')
    if use_existing_model==1:
        best_model_path = f"{save_dir}/best_model_updated.pt"
    else:
        best_model_path = f"{save_dir}/best_model.pt"

    for epoch in range(epochs):

        log["epoch"] = epoch

        step()

        current_loss = log['total_loss'][-1]
        if current_loss < best_loss:
            best_loss = current_loss
            best = pINN.state_dict()
            tt.save(best, best_model_path)

        # Save log
        with open(f"{save_dir}/training_log.json", "w") as log_file:
            json.dump(log, log_file, indent=4)

        if epoch == 1 or epoch % 200 == 0 or epoch == epochs - 1:
            print(
                    f"Epoch {epoch}/{epochs} | "
                    f"Total loss={log['total_loss'][-1]:.3e} | "
                    f"PDELoss={log['pde_loss'][-1]:.3e} | "
                    f"BCLoss={log['bc_loss'][-1]:.3e} |"
                    f"PointLoss={log['obs_loss'][-1]:.3e}|"
                )
        
            plt.figure(figsize=(10, 6))
            plt.semilogy(log["total_loss"], label="Train total loss")
            plt.semilogy(log["pde_loss"], label="PDE Loss")
            plt.semilogy(log["bc_loss"], label="BC Loss")
            plt.semilogy(log["obs_loss"], label="Point Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid()
            if use_existing_model==1:
                plt.savefig(f"{save_dir}/loss_curve_new.png", dpi=200)
            else:                  
                plt.savefig(f"{save_dir}/loss_curve.png", dpi=200)
            plt.close()

    if use_existing_model==1:
        final_path = f"{save_dir}/final_model_updated.pt"
    else:
        final_path = f"{save_dir}/final_model.pt"
    tt.save(pINN.state_dict(), final_path)
    print(f"Saved final model to {final_path}")