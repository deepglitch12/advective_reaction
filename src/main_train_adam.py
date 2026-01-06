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

# --- Configuration for Scales ---
# We define these globally or pass them to ensure consistency across functions
TF_SCALE = 3600.0
L_SCALE = 1000.0
T_MAX = 500.0
T_MIN = 273.0
DT_SCALE = T_MAX - T_MIN

def sample_Xr(nr, T_max=500.0, T_min=273.0, v_max=1.0, v_min=0.0, alpha_max=3e-3, alpha_min=1e-6, T_amb_max = 350.0, L=1000, tf=3600):
    t = tt.rand(nr, 1, device=device) * (tf - 0.1) + 0.1
    x = tt.rand(nr, 1, device=device) * (L - 0.1) + 0.1

    # random parameters (same ranges as dataset)
    T_in = tt.rand(nr, 1, device=device) * (T_max - T_min) + T_min
    v = tt.rand(nr, 1, device=device) * (v_max - v_min) + v_min
    alpha = tt.rand(nr, 1, device=device) * (alpha_max - alpha_min) + alpha_min
    T_amb = tt.rand(nr, 1, device=device) * (T_amb_max - T_min) + T_min

    return tt.cat([t, x, T_in, v, alpha, T_amb], dim=1)

def sample_Xe(ne, T_max=500.0, T_min=273.0, v_max=1.0, v_min=0.0, alpha_max=3e-3, alpha_min=1e-6, T_amb_max = 350.0, L=1000, tf=3600):
    # Initial condition: t = 0
    t0 = tt.zeros(ne // 2, 1, device=device)
    x0 = tt.rand(ne // 2, 1, device=device) * (L - 0.01) + 0.01

    T_in0 = tt.rand(ne // 2, 1, device=device) * (T_max - T_min) + T_min
    v0 = tt.rand(ne // 2, 1, device=device) * (v_max - v_min) + v_min
    alpha0 = tt.rand(ne // 2, 1, device=device) * (alpha_max - alpha_min) + alpha_min
    T_amb0 = tt.rand(ne // 2, 1, device=device) * (T_amb_max - T_min) + T_min

    X_init = tt.cat([t0, x0, T_in0, v0, alpha0, T_amb0], dim=1)

    # Boundary condition: x = 0
    tb = tt.rand(ne // 2, 1, device=device) * tf
    xb = tt.zeros(ne // 2, 1, device=device)

    T_inb = tt.rand(ne // 2, 1, device=device) * (T_max - T_min) + T_min
    vb = tt.rand(ne // 2, 1, device=device) * (v_max - v_min) + v_min
    alphab = tt.rand(ne // 2, 1, device=device) * (alpha_max - alpha_min) + alpha_min
    T_ambb = tt.rand(ne // 2, 1, device=device) * (T_amb_max - T_min) + T_min

    X_bc = tt.cat([tb, xb, T_inb, vb, alphab, T_ambb], dim=1)

    return tt.vstack([X_init, X_bc])


def normalize_inputs(X_in, T_max=500.0, T_min=273.0, v_max=1.0, v_min=0.0, alpha_max=3e-3, alpha_min=1e-6, T_amb_max = 350.0, L=1000, tf=3600):
    # Define Lower and Upper bounds for all 5 columns
    lb = tt.tensor([0.0, 0.0, T_min, v_min, alpha_min, T_min], device=device)
    ub = tt.tensor([tf,  L,   T_max, v_max, alpha_max, T_amb_max], device=device)
    
    # Formula: 2 * (x - lb) / (ub - lb) - 1
    X_norm = 2.0 * (X_in - lb) / (ub - lb) - 1.0
    return X_norm

def pde_sobolev_loss(Xr_org, T_max=500.0, T_min=273.0, T_amb_max = 350.0, L=1000, T=3600):
    # 1. Setup Scaling Factors
    # We use these to map derivatives from Normalized space -> Physical space
    scale_t = 2.0 / T
    scale_x = 2.0 / L
    scale_Temp = (T_max - T_min) / 2.0
    
    # Characteristic Rate for Non-dimensionalization (K/s)
    # This brings the PDE residual magnitude to ~1.0
    characteristic_rate = (T_max - T_min) / T 

    # 2. Forward Pass
    Xr = normalize_inputs(Xr_org)
    Xr.requires_grad_(True)
    T_out_norm = pINN(Xr) # Output is in range [-1, 1]

    # 3. Compute Gradients w.r.t Normalized Inputs (Chain Rule later)
    # d(T_norm) / d(Input_norm)
    grads = tt.autograd.grad(T_out_norm, Xr, tt.ones_like(T_out_norm), create_graph=True)[0]
    dTn_dtn = grads[:, 0:1]
    dTn_dxn = grads[:, 1:2]

    # 4. Convert Derivatives to Physical Units
    # dT/dt = (dT/dTn * dTn/dtn * dtn/dt)
    dT_dt = dTn_dtn * scale_t * scale_Temp
    dT_dx = dTn_dxn * scale_x * scale_Temp
    
    # Reconstruct Physical Temperature for the alpha term
    T_NN = scale_Temp * T_out_norm + (T_max + T_min) / 2.0

    # 5. Compute Physical PDE Residual (Units: K/s)
    F_physical = dT_dt + Xr_org[:, 3:4] * dT_dx + Xr_org[:, 4:5] * (T_NN - Xr_org[:, 5:6])

    # 6. Normalize the Residual (Dimensionless)
    # We divide by the characteristic heating rate to make it unitless and O(1)
    F_norm = F_physical / characteristic_rate

    # 7. Sobolev (Derivative of Residual) - Computed on Normalized Residual
    # We want the gradient of the residual w.r.t inputs to be small (smoothness)
    grad_F = tt.autograd.grad(F_norm, Xr, tt.ones_like(F_norm), create_graph=True)[0]
    
    # Simple Sobolev: penalize gradients of the normalized residual w.r.t normalized time/space
    sobo_loss = (grad_F[:, 0:1]**2 + grad_F[:, 1:2]**2).mean()

    return (F_norm**2).mean(), sobo_loss

def const_loss(Xe_org, T_max=500.0, T_min=273.0, T_amb_max = 350.0):
    Xe = normalize_inputs(Xe_org)
    T_pred_norm = pINN(Xe) # Range [-1, 1]

    # Normalize the Targets to [-1, 1] as well
    # Instead of converting pred to physical, we convert target to normalized
    # This keeps the loss magnitude consistent (around 1.0)
    
    # Create mask
    init_mask = Xe[:, 0] == -1
    bc_mask   = Xe[:, 1] == -1
    
    # Target for Init: (Tinf - Mid) / Scale
    # T_norm = (T_physical - Mid) / Scale
    mid_T = (T_max + T_min) / 2.0
    scale_T = (T_max - T_min) / 2.0
    
    T_target_init_norm =  Xe[:, 5:6]
    
    # Target for BC: (T_in - Mid) / Scale
    # Xe_org[:, 2] is the physical T_in
    T_target_bc_norm = Xe[:, 2]

    # Initial condition loss (Dimensionless)
    if init_mask.any():
        init_loss = (T_pred_norm[init_mask].flatten() - T_target_init_norm) ** 2
        init_loss = init_loss.mean()
    else:
        init_loss = 0.0

    # Boundary condition loss (Dimensionless)
    if bc_mask.any():
        bc_loss = (T_pred_norm[bc_mask].flatten() - T_target_bc_norm[bc_mask]) ** 2
        bc_loss = bc_loss.mean()
    else:
        bc_loss = 0.0

    return init_loss , bc_loss

def observation_loss(Xo_org, To_true, T_max=500.0, T_min=273.0):
    Xo = normalize_inputs(Xo_org)
    T_pred_norm = pINN(Xo)

    # Normalize the Ground Truth to [-1, 1]
    mid_T = (T_max + T_min) / 2.0
    scale_T = (T_max - T_min) / 2.0
    
    To_true_norm = (To_true - mid_T) / scale_T

    # MSE on normalized values
    return ff.mse_loss(T_pred_norm, To_true_norm)

def step():
    global data_iter

    pINN.train()
    optimizer.zero_grad()

    try:
        Xo, To = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        Xo, To = next(data_iter)
    
    # Move data to device
    Xo = Xo.to(device)
    To = To.to(device)

    # All losses now return dimensionless values approx O(1)
    obs_loss = observation_loss(Xo, To)

    Xr = sample_Xr(nr)
    pde_loss, sobo_loss = pde_sobolev_loss(Xr)

    Xe = sample_Xe(ne)
    init_loss, bc_loss = const_loss(Xe)

    # Weights can now be closer to 1.0 since units are balanced
    lambda_obs  = 10.0
    lambda_pde  = 1.0   
    lambda_bc   = 50.0
    lambda_init = 50.0
    lambda_t    = 1e-3  # Sobolev regularization weight

    total_loss = (
        lambda_obs * obs_loss   +
        lambda_pde * pde_loss   +
        lambda_bc * bc_loss     +
        lambda_init * init_loss +
        lambda_t * sobo_loss
    )

    total_loss.backward()

    # Gradient clipping helps stability
    tt.nn.utils.clip_grad_norm_(pINN.parameters(), max_norm=1.0)
    
    optimizer.step()

    log["total_loss"].append(total_loss.item())
    log["pde_loss"].append(pde_loss.item())
    log["bc_loss"].append(bc_loss.item())
    log["init_loss"].append(init_loss.item())
    log["obs_loss"].append(obs_loss.item())
    log["sobo"].append(sobo_loss.item())

if __name__ == '__main__':

    device = 'cpu' # Change to 'cuda' if available
    use_existing_model = 0

    # Define model
    pINN = HeatPINN().to(device)

    if use_existing_model == 1:
        if os.path.exists('out/best_model.pt'):
            pINN.load_state_dict(tt.load('out/best_model.pt', map_location=device))
            print("Model Loaded Successfully!!")
            learning_rate = 1e-2 # Lower LR for fine-tuning
        else:
            print("Model not found, starting fresh.")
            learning_rate = 1e-2
    else:
        learning_rate = 1e-3
    
    lambda_ridge = 1e-4
    optimizer = tt.optim.Adam(pINN.parameters(), lr=learning_rate, weight_decay=lambda_ridge)

    path_in_dir_script = Path(__file__).parent
    path_out_dir = path_in_dir_script / "../src"
    path_out_dir.mkdir(exist_ok=True)

    # Logging dict
    log = dict(
        epoch="",
        total_loss = [],
        pde_loss = [],
        bc_loss = [],
        obs_loss = [],
        init_loss = [],
        sobo = [],
        saved_weights=[],
        device=device,
    )

    n  = 5000
    ne = 5000
    nr = 5000

    # Ensure dataset exists or handle gracefully
    try:
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
    except Exception as e:
        print(f"Warning: Could not load dataset ({e}). Ensure data path is correct.")
        # Create a dummy loader for testing if dataset is missing
        dummy_data = [(tt.rand(n, 6), tt.rand(n, 1)) for _ in range(10)]
        loader = DataLoader(dummy_data, batch_size=n)
        data_iter = iter(loader)

    epochs = 20000
    save_dir = "out"
    os.makedirs(save_dir, exist_ok=True)

    best_loss = float('inf')
    best_model_path = f"{save_dir}/best_model.pt"

    for epoch in range(epochs):
        log["epoch"] = epoch
        step()

        current_loss = log['total_loss'][-1]
        
        # Save best model
        if current_loss < best_loss:
            best_loss = current_loss
            best = pINN.state_dict()
            tt.save(best, best_model_path)

        # Periodic logging
        if epoch == 0 or epoch % 200 == 0 or epoch == epochs - 1:
            # Save log to JSON
            with open(f"{save_dir}/training_log.json", "w") as log_file:
                json.dump(log, log_file, indent=4)

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Total={log['total_loss'][-1]:.2e} | "
                f"PDE={log['pde_loss'][-1]:.2e} | "
                f"BC={log['bc_loss'][-1]:.2e} | "
                f"Obs={log['obs_loss'][-1]:.2e} | "
                f"IC={log['init_loss'][-1]:.2e} | "
                f"Sobo={log['sobo'][-1]:.2e}"
            )
        
            plt.figure(figsize=(10, 6))
            plt.semilogy(log["total_loss"], label="Total Loss")
            plt.semilogy(log["pde_loss"], label="PDE Loss (Norm)")
            plt.semilogy(log["bc_loss"], label="BC Loss (Norm)")
            plt.semilogy(log["init_loss"], label="IC Loss (Norm)")
            plt.semilogy(log["obs_loss"], label="Obs Loss (Norm)")
            plt.xlabel("Epoch")
            plt.ylabel("Loss (Dimensionless)")
            plt.legend()
            plt.grid(True, which="both", ls="-")
            plt.savefig(f"{save_dir}/loss_curve_new.png" if use_existing_model==1 
                        else f"{save_dir}/loss_curve.png", dpi=200)
            plt.close()

    final_path = f"{save_dir}/final_model.pt"
    tt.save(pINN.state_dict(), final_path)
    print(f"Saved final model to {final_path}")