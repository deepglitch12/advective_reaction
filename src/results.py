import numpy as np
import matplotlib.pyplot as plt
import torch as tt
from one_d_pipe_dynamics import pipe_rk4
from scipy.interpolate import interp1d
from pinn_model import HeatPINN
import os

# ================== NORMALIZATION ==================
def normalize_inputs(
    X_in,
    T_max=500.0,
    T_min=273.0,
    v_max=1.0,
    v_min=0.0,
    alpha_max=3e-3,
    alpha_min=1e-6,
    T_amb_max=350.0,
    L=1000.0,
    tf=3600.0,
):
    lb = tt.tensor([0.0, 0.0, T_min, v_min, alpha_min, T_min], device=device)
    ub = tt.tensor([tf, L, T_max, v_max, alpha_max, T_amb_max], device=device)
    return 2.0 * (X_in - lb) / (ub - lb) - 1.0


# ================== PARAMETERS ==================
L = 1000.0
N = 1000
u = 0.8

rho = 1000.0
cp = 4180.0
D = 0.1
h = 80.0

Tinf = 273.0
t0 = 0.0
tf = 3600.0
dt = 1.0

device = "cpu"

A = np.pi * D**2 / 4
P = np.pi * D
alpha = h * P / (rho * cp * A)

time_vec = np.arange(t0, tf + dt, dt)

# ================== LOAD PINN ==================
pINN = HeatPINN().to(device)
pINN.load_state_dict(tt.load("out/decent_model.pt", map_location=device))
pINN.eval()
print("PINN model loaded.")

# ================== PINN PREDICTION ==================
def pinn_predict(t_phys, x_phys, T_in, v, alpha, T_amb):
    x = tt.tensor(x_phys, device=device, dtype=tt.float32).reshape(-1, 1)
    t = tt.full_like(x, float(t_phys))
    Ts = tt.full_like(x, float(T_in))
    us = tt.full_like(x, float(v))
    al = tt.full_like(x, float(alpha))
    Ta = tt.full_like(x, float(T_amb))

    X = tt.cat([t, x, Ts, us, al, Ta], dim=1)

    with tt.no_grad():
        T = ((500.0 - 273.0) / 2.0) * pINN(normalize_inputs(X)) \
            + (500.0 + 273.0) / 2.0

    return T.cpu().numpy().flatten()


# ================== SETTINGS ==================
T_heat_list = [350.0, 400.0, 475.0]
snapshot_times = [600, 1800, 3600]

base_out_dir = "out"
os.makedirs(base_out_dir, exist_ok=True)

# ================== MAIN LOOP ==================
for T_heat in T_heat_list:

    print(f"\nRunning case: T_heat = {T_heat} K")
    case_dir = os.path.join(base_out_dir, f"T_heat_{int(T_heat)}")
    os.makedirs(case_dir, exist_ok=True)

    # -------- Inlet ramp --------
    ramp_val = np.array([
        0.0 if t < 0 else
        t / 1200.0 if t < 1200.0 else
        1.0
        for t in time_vec
    ])

    Tin_vec = (T_heat - Tinf) * ramp_val + Tinf

    T_in = interp1d(
        time_vec,
        Tin_vec,
        kind="previous",
        fill_value="extrapolate",
        bounds_error=False,
    )

    # -------- RK4 Simulation --------
    x, t_hist, T_hist = pipe_rk4(
        L, N, u, rho, cp, D, h, Tinf,
        T_in, t0, tf, dt
    )

    # ================== SNAPSHOT PROFILES ==================
    for t_snap in snapshot_times:
        k = int(t_snap / dt)

        T_rk = T_hist[k, 1:]
        T_pinn = pinn_predict(
            t_snap, x,
            T_in=T_in(t_snap),
            v=u,
            alpha=alpha,
            T_amb=Tinf
        )[1:]

        plt.figure()
        plt.plot(x[1:], T_rk, "r-", linewidth=2, label="RK4")
        plt.plot(x[1:], T_pinn, "b--", linewidth=2, label="PINN")
        plt.xlabel("x [m]")
        plt.ylabel("Temperature [K]")
        plt.title(f"T_heat = {T_heat} K, t = {t_snap} s")
        plt.grid(True)
        plt.legend()

        plt.savefig(
            os.path.join(case_dir, f"profile_t{t_snap}.png"),
            dpi=200
        )
        plt.close()

    # ================== ERROR VS TIME ==================
    error_l2 = []

    for k in range(len(t_hist)):
        T_rk = T_hist[k, 1:]
        T_pinn = pinn_predict(
            t_hist[k], x,
            T_in=T_in(t_hist[k]),
            v=u,
            alpha=alpha,
            T_amb=Tinf
        )[1:]

        diff = T_rk - T_pinn
        error_l2.append(np.linalg.norm(diff) / np.sqrt(len(diff)))

    error_l2 = np.array(error_l2)

    plt.figure()
    plt.plot(t_hist, error_l2)
    plt.xlabel("Time [s]")
    plt.ylabel("RMS Error [K]")
    plt.title(f"RMS Error vs Time (T_heat = {T_heat} K)")
    plt.grid(True)
    plt.savefig(os.path.join(case_dir, "rms_error_vs_time.png"), dpi=200)
    plt.close()

    # ================== SPACE–TIME ERROR MAP ==================
    Err_map = np.zeros((len(t_hist), len(x) - 1))

    for k in range(len(t_hist)):
        T_rk = T_hist[k, 1:]
        T_pinn = pinn_predict(
            t_hist[k], x,
            T_in=T_in(t_hist[k]),
            v=u,
            alpha=alpha,
            T_amb=Tinf
        )[1:]
        Err_map[k, :] = np.abs(T_rk - T_pinn)

    plt.figure(figsize=(8, 4))
    plt.imshow(
        Err_map,
        aspect="auto",
        origin="lower",
        extent=[x[1], x[-1], t_hist[0], t_hist[-1]]
    )
    plt.colorbar(label="|Error| [K]")
    plt.xlabel("x [m]")
    plt.ylabel("Time [s]")
    plt.title(f"Absolute Error Map (T_heat = {T_heat} K)")
    plt.savefig(os.path.join(case_dir, "absolute_error_map.png"), dpi=200)
    plt.close()

    # ================== OUTLET TEMPERATURE ==================
    T_out_rk = T_hist[:, -1]
    T_out_pinn = np.array([
        pinn_predict(
            t_hist[k], x,
            T_in=T_in(t_hist[k]),
            v=u,
            alpha=alpha,
            T_amb=Tinf
        )[-1]
        for k in range(len(t_hist))
    ])

    plt.figure()
    plt.plot(t_hist, T_out_rk, "r-", label="RK4")
    plt.plot(t_hist, T_out_pinn, "b--", label="PINN")
    plt.xlabel("Time [s]")
    plt.ylabel("Outlet Temperature [K]")
    plt.title(f"Outlet Temperature (T_heat = {T_heat} K)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(case_dir, "outlet_temperature.png"), dpi=200)
    plt.close()

print("\nAll simulations and plots saved successfully.")
