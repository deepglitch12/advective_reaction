import numpy as np
import matplotlib.pyplot as plt
from one_d_pipe_dynamics import pipe_rk4
from scipy.interpolate import interp1d

L = 1000.0
N = 1000
u = 0.4

rho = 1000.0
cp = 4180.0
D = 0.1
h = 50.0

T_amb = 290.0
t0 = 0.0
tf = 3600.0
dt = 1.0

T_base = T_amb        # K
T_heat = 500.0        # pulse temperature
t_start_pulse = 1
t_duration = 2000

# time grid for inlet definition (1-second resolution)
time_vec = np.arange(t0, tf + 1)

Tin_vec = T_base * np.ones_like(time_vec)
Tin_vec[t_start_pulse : t_start_pulse + t_duration] = T_heat

# MATLAB: interp1(...,'previous','extrap')
T_in = interp1d(
    time_vec,
    Tin_vec,
    kind="previous",
    fill_value="extrapolate",
    bounds_error=False,
)

x, t_hist, T_hist = pipe_rk4(
    L, N, u, rho, cp, D, h, T_amb,
    T_in, t0, tf, dt
)

plt.figure()
ax = plt.gca()
ax.grid(True)

ax.set_xlabel("x [m]")
ax.set_ylabel("Temperature [K]")
ax.set_ylim(T_amb - 50, T_heat + 50)
ax.set_xlim(x[0], x[-1])

step_size = 10  # seconds

for k in range(0, len(t_hist), int(step_size / dt)):
    ax.cla()
    ax.grid(True)

    ax.plot(x[1:], T_hist[k, 1:], "r-", linewidth=2)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("Temperature [K]")
    ax.set_ylim(T_amb - 50, T_heat + 50)
    ax.set_xlim(x[0], x[-1])

    ax.set_title(f"Time = {t_hist[k]:.1f} s")
    plt.pause(0.01)

plt.show()

