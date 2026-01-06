import numpy as np


def pipe_rk4(L, N, v, rho, cp, D, h, T_amb, T_in, t0, tf, dt):
    """
    Solves 1D plug-flow heat equation in a pipe with heat loss using RK4.

    PDE:
        dT/dt + v dT/dx = -alpha (T - T_amb)

    Parameters
    ----------
    L : float
        Pipe length [m]
    N : int
        Number of spatial nodes
    v : float
        Mean fluid velocity [m/s]
    rho : float
        Fluid density [kg/m^3]
    cp : float
        Specific heat [J/kg-K]
    D : float
        Pipe diameter [m]
    h : float
        Convective heat transfer coefficient [W/m^2-K]
    T_amb : float
        Ambient temperature [K]
    T_in : callable
        Inlet temperature function, T_in(t)
    t0, tf : float
        Start and end time [s]
    dt : float
        Time step [s]

    Returns
    -------
    x : ndarray, shape (N,)
        Spatial grid
    t_hist : ndarray, shape (Nt,)
        Time grid
    T_hist : ndarray, shape (Nt, N)
        Temperature history
    """

    # geometry
    A = np.pi * D**2 / 4
    P = np.pi * D
    alpha = h * P / (rho * cp * A)

    dx = L / (N - 1)
    x = np.linspace(0.0, L, N)

    # initial condition
    T = T_amb * np.ones(N)

    # time grid
    t_hist = np.arange(t0, tf + dt, dt)
    Nt = len(t_hist)

    T_hist = np.zeros((Nt, N))
    T_hist[0, :] = T.copy()

    # parameter bundle (MATLAB-style)
    par = (dx, N, T_amb, alpha, v)

    # time stepping
    for k in range(Nt - 1):
        t = t_hist[k]

        k1 = one_d_pipe(T, t, T_in, par)
        k2 = one_d_pipe(T + 0.5 * dt * k1, t + 0.5 * dt, T_in, par)
        k3 = one_d_pipe(T + 0.5 * dt * k2, t + 0.5 * dt, T_in, par)
        k4 = one_d_pipe(T + dt * k3, t + dt, T_in, par)

        T = T + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        T_hist[k + 1, :] = T.copy()

    return x, t_hist, T_hist


def one_d_pipe(T, t, T_in, par):
    """
    RHS for the 1D pipe temperature ODE system
    """

    dx, N, T_amb, alpha, v = par

    dTdt = np.zeros_like(T)

    # Enforce inlet temperature
    T = T.copy()
    T[0] = T_in(t)

    # upwind spatial derivative
    dTdx = np.diff(T) / dx

    dTdt[1:] = -v * dTdx - alpha * (T[1:] - T_amb)

    # inlet node held fixed
    dTdt[0] = 0.0

    return dTdt
