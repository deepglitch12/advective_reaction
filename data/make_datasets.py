import numpy as np
from one_d_pipe_dynamics import pipe_rk4
from scipy.interpolate import interp1d
import os

# ======================
# Fixed parameters
# ======================
L = 1000.0
N = 1000
rho = 1000.0
cp = 4180.0

t0 = 0.0
tf = 3600.0
dt = 1.0

t_start_ramp = 0
t_ramp_peak  = 1200
t_ramp_dur   = 2400

time_vec = np.arange(t0, tf + dt, dt)

ramp_val = np.array([
    0.0 if t < t_start_ramp else
    (t - t_start_ramp) / (t_ramp_peak - t_start_ramp) if t < t_ramp_peak else
    1.0
    for t in time_vec
])


# ======================
# Parameter grids
# ======================
T_in_values = [290.0, 350.0, 500.0]
v_values = [0.1, 0.4, 0.9]
T_amb_values = [273.0]

alpha_params = [
    {"h": 20.0,  "D": 0.2},
    {"h": 80.0, "D": 0.1},
    {"h": 120.0, "D": 0.05},
]

# ======================
# Output folder
# ======================
dataset_dir = "./data/datasets"
os.makedirs(dataset_dir, exist_ok=True)


dataset_id = 1

# ======================
# Main loop
# ======================
for T_heat in T_in_values:
    for T_amb in T_amb_values:
        # inlet temperature profile
        Tin_vec = (T_heat - T_amb)*ramp_val + T_amb

        T_in = interp1d(
            time_vec,
            Tin_vec,
            kind="previous",
            fill_value="extrapolate",
            bounds_error=False,
        )

        for u in v_values:
            for ap in alpha_params:

                h = ap["h"]
                D = ap["D"]

                print(
                    f"Running dataset {dataset_id}: "
                    f"T_in={T_heat}, v={u}, h={h}, D={D}, T_amb={T_amb}"
                )

                x, t_hist, T_hist = pipe_rk4(
                    L, N, u, rho, cp, D, h, T_amb,
                    T_in, t0, tf, dt
                )

                # ======================
                # Save dataset
                # ======================
                filename = os.path.join(dataset_dir, f"dataset_{dataset_id}.npz")


                np.savez(
                    filename,
                    x=x,
                    t=t_hist,
                    T=T_hist,
                    T_in=Tin_vec,
                    v=u,
                    h=h,
                    D=D,
                    rho=rho,
                    cp=cp,
                    T_amb=T_amb,
                    dt=dt,
                )

                dataset_id += 1
