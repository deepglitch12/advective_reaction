import torch as tt
import torch.nn.functional as ff
from torch.utils.data import DataLoader
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from pinn_model import HeatPINN
from datasets import PipeDatasetFromFiles
from main_train_adam import sample_Xe, sample_Xr, observation_loss, pde_sobolev_loss, const_loss



def train_lbfgs():
    
    ne = 5000
    nr = 5000

    dataset = PipeDatasetFromFiles(
        dataset_dir="./data/datasets",
        device=device
    )

    loader = DataLoader(
        dataset,
        batch_size=len(dataset),  # FULL batch
        shuffle=False
    )

    Xo, To = next(iter(loader))
    Xo, To = Xo.to(device), To.to(device)

    # -----------------------------
    # Optimizer: LBFGS
    # -----------------------------
    optimizer = tt.optim.LBFGS(
        pINN.parameters(),
        lr=1.0,
        max_iter=20,
        max_eval=25,
        history_size=50,
        line_search_fn="strong_wolfe"
    )

    # -----------------------------
    # Loss weights
    # -----------------------------
    lambda_obs  = 10.0
    lambda_pde  = 1.0
    lambda_bc   = 50.0
    lambda_init = 50.0
    lambda_t    = 1e-3

    # -----------------------------
    # Logging
    # -----------------------------
    log = dict(
        total_loss=[],
        pde_loss=[],
        bc_loss=[],
        init_loss=[],
        obs_loss=[],
        sobo=[]
    )

    save_dir = "out"
    os.makedirs(save_dir, exist_ok=True)

    best_loss = float("inf")

    # -----------------------------
    # Training loop
    # -----------------------------
    epochs = 500  # LBFGS converges fast

    for epoch in range(epochs):

        def closure():
            optimizer.zero_grad()

            # Observation loss
            obs_loss = observation_loss(Xo, To)

            # Physics samples
            Xr = sample_Xr(nr)
            pde_loss, sobo_loss = pde_sobolev_loss(Xr)

            Xe = sample_Xe(ne)
            init_loss, bc_loss = const_loss(Xe)

            total_loss = (
                lambda_obs  * obs_loss +
                lambda_pde  * pde_loss +
                lambda_bc   * bc_loss +
                lambda_init * init_loss +
                lambda_t    * sobo_loss
            )

            total_loss.backward()
            return total_loss

        loss = optimizer.step(closure)

        # -------- Logging (recompute once, no grad) --------
        with tt.no_grad():
            obs_loss = observation_loss(Xo, To)
            Xr = sample_Xr(nr)
            pde_loss, sobo_loss = pde_sobolev_loss(Xr)
            Xe = sample_Xe(ne)
            init_loss, bc_loss = const_loss(Xe)

            total_loss = (
                lambda_obs  * obs_loss +
                lambda_pde  * pde_loss +
                lambda_bc   * bc_loss +
                lambda_init * init_loss +
                lambda_t    * sobo_loss
            )

        log["total_loss"].append(total_loss.item())
        log["pde_loss"].append(pde_loss.item())
        log["bc_loss"].append(bc_loss.item())
        log["init_loss"].append(init_loss.item())
        log["obs_loss"].append(obs_loss.item())
        log["sobo"].append(sobo_loss.item())

        # -------- Save best --------
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            tt.save(pINN.state_dict(), f"{save_dir}/best_model_lbfgs.pt")

        # -------- Print --------
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(
                f"[LBFGS] Epoch {epoch:4d} | "
                f"Total={total_loss:.2e} | "
                f"PDE={pde_loss:.2e} | "
                f"BC={bc_loss:.2e} | "
                f"IC={init_loss:.2e} | "
                f"Obs={obs_loss:.2e} | "
                f"Sobo={sobo_loss:.2e}"
            )

    # -----------------------------
    # Save final
    # -----------------------------
    tt.save(pINN.state_dict(), f"{save_dir}/final_model_lbfgs.pt")

    with open(f"{save_dir}/training_log_lbfgs.json", "w") as f:
        json.dump(log, f, indent=4)

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(10, 6))
    plt.semilogy(log["total_loss"], label="Total")
    plt.semilogy(log["pde_loss"], label="PDE")
    plt.semilogy(log["bc_loss"], label="BC")
    plt.semilogy(log["init_loss"], label="IC")
    plt.semilogy(log["obs_loss"], label="Obs")
    plt.legend()
    plt.grid(True, which="both")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{save_dir}/loss_curve_lbfgs.png", dpi=200)
    plt.close()


if __name__ == "__main__":

    device = "cpu"  # or "cuda"

    pINN = HeatPINN().to(device)


    pINN.load_state_dict(tt.load('out/final_model.pt', map_location=device))
    print("Model Loaded Successfully!!")


    train_lbfgs()
