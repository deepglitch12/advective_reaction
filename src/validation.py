import numpy as np
import matplotlib.pyplot as plt
import torch as tt
from one_d_pipe_dynamics import pipe_rk4
from scipy.interpolate import interp1d
from pinn_model import HeatPINN

def normalize_inputs(X_in, T_max=500.0, T_min=273.0, v_max=1.0, v_min=0.0, alpha_max=3e-3, alpha_min=1e-6, T_amb_max = 350.0, L=1000, tf=3600):
    # Define Lower and Upper bounds for all 5 columns
    lb = tt.tensor([0.0, 0.0, T_min, v_min, alpha_min, T_min], device=device)
    ub = tt.tensor([tf,  L,   T_max, v_max, alpha_max, T_amb_max], device=device)
    
    # Formula: 2 * (x - lb) / (ub - lb) - 1
    X_norm = 2.0 * (X_in - lb) / (ub - lb) - 1.0
    return X_norm

# ------------------ PARAMETERS ------------------
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


# ------------------ INLET PROFILE ------------------
T_base = Tinf
T_heat = 475.0
t_start_pulse = 0
t_duration = 3600

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
Tin_vec = (T_heat - Tinf)*ramp_val + Tinf


T_in = interp1d(
    time_vec,
    Tin_vec,
    kind="previous",
    fill_value="extrapolate",
    bounds_error=False,
)

# ------------------ RK4 SOLUTION ------------------
x, t_hist, T_hist = pipe_rk4(
    L, N, u, rho, cp, D, h, Tinf,
    T_in, t0, tf, dt
)

# ------------------ PINN NORMALIZATION ------------------
tf_norm = tf/2.0
x_norm = L/2.0

device = 'cpu'
pINN = HeatPINN().to(device)
pINN.load_state_dict(tt.load('out/best_model.pt', map_location=device))
print("Model Loaded Sucessfully!!")

def pinn_predict(t_phys, x_phys, T_in, v, alpha, T_amb, T_max=500.0, T_min=273.0):

    x = tt.tensor(x_phys, device=device, dtype=tt.float32).reshape(-1, 1)

    t  = tt.full_like(x, float(t_phys), dtype=tt.float32)
    Ts = tt.full_like(x, float(T_in),   dtype=tt.float32)
    us = tt.full_like(x, float(v),      dtype=tt.float32)
    al = tt.full_like(x, float(alpha),  dtype=tt.float32)
    Ta = tt.full_like(x, float(T_amb),  dtype=tt.float32)

    # normalize input order: (N,5)
    Xstack = tt.cat([t, x, Ts, us, al, Ta], dim=1)

    with tt.no_grad():
        T_pred = ((T_max - T_min) / 2.0) * pINN(normalize_inputs(Xstack)) \
                 + (T_max + T_min) / 2.0

    return T_pred.cpu().numpy().flatten()


# ------------------ ANIMATION ------------------
plt.figure()
ax = plt.gca()
ax.set_xlabel("x [m]")
ax.set_ylabel("Temperature [K]")
# ax.set_ylim(T_amb - 50, T_heat + 50)
ax.set_xlim(x[0], x[-1])
ax.grid(True)

step_size = 10  # seconds

for k in range(0, len(t_hist), int(step_size / dt)):
    ax.cla()
    ax.grid(True)

    # RK4 (ground truth)
    ax.plot(
        x[1:], T_hist[k, 1:],
        "r-", linewidth=2, label="RK4"
    )

    # PINN prediction
    T_pinn = pinn_predict(t_hist[k], x, T_in=T_in(t_hist[k]), v=u, alpha=alpha, T_amb=Tinf)
    ax.plot(
        x[1:], T_pinn[1:],
        "b--", linewidth=2, label="PINN"
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("Temperature [K]")
    ax.set_ylim( Tinf, T_heat + 50)
    ax.set_xlim(x[0], x[-1])
    ax.set_title(f"Time = {t_hist[k]:.1f} s")
    ax.legend()

    plt.pause(0.01)

plt.show()
