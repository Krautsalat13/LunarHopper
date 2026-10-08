#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Planar lunar-hopper descent guidance, navigation and control study.

This script simulates a 2D lunar hopper (translation + pitch) landing on the
Moon and analyses how well it lands, both with perfect state knowledge and when
the state is reconstructed from noisy sensors.

Methods
-------
* Dynamics: rigid-body equations for a single gimbaled thruster, integrated
  with a fixed-step RK4 propagator.
* Control: a cascade of three single-input PD loops (altitude, lateral,
  attitude). The lateral and attitude gains are scheduled on the current
  thrust so each loop keeps a fixed damped second-order response.
* Navigation: an Extended Kalman Filter over the translational state
  (x, z, vx, vz) fed by an accelerometer (every step) and an altimeter
  (intermittent). With attitude taken as known the model is linear, so the
  EKF here coincides with a linear Kalman filter.
* Analysis: Monte Carlo dispersion over initial conditions, a perfect-state
  versus EKF comparison over the same dispersions, and a NEES/NIS statistical
  consistency check of the filter.

Key parameters
---------------
m = 150 kg, g = 1.625 m/s^2, release altitude z0 = 10 m, step dt = 0.01 s.
Monte Carlo dispersion: x0 in [-5, 5] m, vx0 in [-2, 2] m/s, vz0 in [-3, 0] m/s,
theta0 in [-10, 10] deg, omega0 in [-3, 3] deg/s.

Outputs
-------
Figures are written to ``Plots/`` as both PNG (for the README) and PDF (for the
LaTeX reports). Monte Carlo result tables are written to ``Data/`` as CSV.

Running it
----------
    python Hopper.py

Regenerates every figure and data file in one pass. Runs are seeded, so the
results are reproducible. LaTeX text rendering is used if a system LaTeX
install is found on PATH, otherwise matplotlib's built-in mathtext is used.
"""
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")            # non-interactive backend: run headless, no windows
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "Plots"   # figures (PNG for the README, PDF for the reports)
DATA_DIR = BASE_DIR / "Data"     # Monte Carlo result tables (CSV)
PLOTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


def save_fig(fig, name):
    """Save a figure to Plots/ as both PNG (README) and PDF (LaTeX reports)."""
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")

# Fall back to matplotlib's built-in mathtext (Computer Modern) if LatTex installation missing.
_HAS_LATEX = shutil.which("latex") is not None

plt.rcParams.update({
    "text.usetex": _HAS_LATEX,
    "text.latex.preamble": r'\usepackage[dvipsnames]{xcolor}',
    "font.family": "serif",          
    "mathtext.fontset": "cm",        
    "font.size": 14,                
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.7
})


def Color(color_name):
    colors = {
        "RWTH": (0, 84, 159),
        "Schwarz": (0, 0, 0),
        "Petrol": (0, 97, 101),
        "Türkis": (0, 152, 161),
        "Grün": (87, 171, 39),
        "Maigrün": (189, 205, 0),
        "Gelb": (255, 237, 0),
        "Orange": (246, 168, 0),
        "Rot": (204, 7, 30),
        "Magenta": (227, 0, 102),
        "Bordeaux": (161, 16, 53),
        "Violett": (97, 33, 88),
        "Lila": (122, 111, 172),
        "RWTHlight": (142, 186, 229),
        "Bordeauxlight": (205,139,135)
    }
    if color_name not in colors:
        raise ValueError(f"Color '{color_name}' not found.")
    r, g, b = colors[color_name]
    return (r / 255, g / 255, b / 255)

# Constants
m = 150.0          # mass [kg]
g = 1.625          # moon gravity [m/s2]
h = 2.0            # height [m]
r = 0.25           # radius [m]
l = h/2            # gimbal-to-CoM distance [m] (CoM at geometric center)
I = (1/12)*m*(3*r**2 + h**2)  # moment of inertia about CoM [kgm2]

#y' = f(t,y(t))
def dynamics(state, T, delta):
    x, z, vx, vz, theta, omega = state
    ax = (T/m) * np.sin(theta + delta)
    az = (T/m) * np.cos(theta + delta) - g
    alpha = (l * T / I) * np.sin(delta)
    return np.array([vx, vz, ax, az, omega, alpha])

def RK4(state, T, delta, dt):
    k1 = dynamics(state, T, delta)
    k2 = dynamics(state + 0.5 * dt * k1, T, delta)
    k3 = dynamics(state + 0.5 * dt * k2, T, delta)
    k4 = dynamics(state + dt * k3, T, delta)
    return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)



# initial state = [x, z, vx, vz, theta, omega]
initial_state = np.array([0, 10, 0, 0, 0, 0])


# Verification via edge cases

def test_dynamics():
    # Free Fall
    state = initial_state
    dt = 0.01
    t = 0
    while state[1] > 0:
        state = RK4(state, 0, 0, dt)
        t += dt

    print(f"Impact at t = {t:.3f} s (expected: 3.508s)")

    # Gravity Canceling
    state = initial_state
    dt = 0.01
    t = 5
    while t > 0:
        state = RK4(state, m*g, 0, dt)
        t -= dt

    print(f"Height after 5s: {state[1]:.3f} m (expected: 10m)")

    # Gimbal set
    state = initial_state
    dt = 0.01
    t = 1
    while t > 0:
        state = RK4(state, m*g, np.radians(5), dt)
        t -= dt

    print(f"state: {state}")

# Real Dynamics
# PD control
omega_n_z = 0.5
xi_z = 1
K_p_z = m* omega_n_z**2
K_d_z = 2 * m * xi_z * omega_n_z

omega_n_delta = 5*omega_n_z
xi_delta = 0.7

omega_n_x = 0.5
xi_x = 0.7

def C(T):
    return l*T/I


# ------------------------------------------------------------------
# Sensors + Extended Kalman Filter
# ------------------------------------------------------------------
# The EKF estimates only translation, [x, z, vx, vz]. Attitude (theta, omega)
# is assumed known and passed straight to the attitude loop.
#
# Sensors:
#   IMU        : accelerometer read every step, measures the specific force
#                (thrust acceleration, gravity is not sensed) plus white noise.
#                The filter propagates on this reading (inertial dead-reckoning).
#   Altimeter  : a downward-pointing sensor read intermittently, measures the
#                altitude z plus white noise, and bounds the vertical drift.
#
# The altimeter observes altitude directly, so horizontal position and velocity
# are corrected only through IMU dead-reckoning, as with a real IMU+altimeter
# suite. The filter is written with the EKF predict/update structure (Jacobians
# F and H); with attitude known the translational model happens to be linear, so
# here it coincides with a linear Kalman filter.

SIGMA_A    = 0.01          # accelerometer noise std [m/s^2]
SIGMA_Z    = 0.15          # altimeter noise std [m]
ALT_PERIOD = 20            # altimeter fix every 20 steps (0.2 s at dt=0.01)
# Initial estimate variance [x, z, vx, vz]. Horizontal position/velocity are
# not aided by the altimeter, so any initial velocity error integrates into an
# unobservable position drift over the descent; the initial velocity estimate
# is therefore taken to be good (navigation-grade), a few cm/s.
P0_DIAG    = [0.09, 0.09, 0.0009, 0.0009]   # std: 0.3 m, 0.3 m, 0.03 m/s, 0.03 m/s


def ekf_predict(xhat, P, a_meas, dt):
    # Propagate the estimate with the measured specific force a_meas = [ax, az].
    # Gravity is added back here since the accelerometer does not sense it.
    F = np.eye(4)
    F[0, 2] = dt
    F[1, 3] = dt

    xhat = xhat.copy()
    xhat[0] += xhat[2] * dt
    xhat[1] += xhat[3] * dt
    xhat[2] += a_meas[0] * dt
    xhat[3] += (a_meas[1] - g) * dt

    # Process noise: white acceleration held over the step (DWNA model).
    gvec = np.array([[0.5 * dt * dt], [dt]])
    Qax = SIGMA_A**2 * (gvec @ gvec.T)      # 2x2 block for one axis
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] += Qax        # x, vx
    Q[np.ix_([1, 3], [1, 3])] += Qax        # z, vz

    P = F @ P @ F.T + Q
    return xhat, P


def ekf_update_altimeter(xhat, P, z_meas):
    # Altimeter measurement h(x) = z.
    H = np.array([[0.0, 1.0, 0.0, 0.0]])
    y = z_meas - xhat[1]                     # innovation
    S = H @ P @ H.T + SIGMA_Z**2
    K = P @ H.T / S[0, 0]                     # 4x1
    xhat = xhat + K[:, 0] * y
    P = (np.eye(4) - K @ H) @ P
    return xhat, P


def simulate(state, t = 20, dt = 0.01, use_ekf = False, rng = None, record = False):
    height = []
    time = []
    Thrust = []
    delta_angles = []
    theta_angles = []
    horizontal_disposition = []
    below_floor = []

    # Optional truth/estimate history for the estimation diagnostic.
    # P and the altimeter innovations are also logged, for the NEES/NIS
    # consistency check in ekf_consistency().
    hist = {'true_x': [], 'true_z': [], 'true_vx': [], 'true_vz': [],
            'est_x': [], 'est_z': [], 'est_vx': [], 'est_vz': [],
            'P': [], 'nu': [], 'nis_t': []}

    state = np.array(state, dtype=float)

    if use_ekf:
        if rng is None:
            rng = np.random.default_rng()
        P = np.diag(P0_DIAG).astype(float)
        err = rng.normal(0, np.sqrt(P0_DIAG))       # initial estimate error
        xhat = state[:4].copy() + err
    else:
        xhat = None

    engine_off = False
    touchdown_idx = None
    touchdown_state = None
    for i in range(int(t/dt)):
        height.append(state[1])
        time.append(i*dt)
        below_floor.append(state[1] < 0)

        # Control state: fly on the EKF estimate for translation when use_ekf
        # is on, otherwise on truth. Attitude is always taken from truth.
        if use_ekf:
            cs = np.array([xhat[0], xhat[1], xhat[2], xhat[3], state[4], state[5]])
        else:
            cs = state

        if record:
            hist['true_x'].append(state[0]); hist['true_z'].append(state[1])
            hist['true_vx'].append(state[2]); hist['true_vz'].append(state[3])
            if use_ekf:
                hist['est_x'].append(xhat[0]); hist['est_z'].append(xhat[1])
                hist['est_vx'].append(xhat[2]); hist['est_vz'].append(xhat[3])
                # Covariance carried by the estimate that flies this step,
                # aligned with the est_* values above for the NEES.
                hist['P'].append(P.copy())

        # Engine cutoff triggers on what the loop believes its altitude is,
        # touchdown is a physical event and is detected on the true altitude.
        if not engine_off and cs[1] <= 0.10 and i > 0:
            engine_off = True

        if touchdown_idx is None and state[1] <= 0 and i > 0:
            touchdown_idx = i
            touchdown_state = state.copy()

        if engine_off:
            T = 0
            delta = 0
        else:
            T = m*g + K_p_z * (0 - cs[1]) + K_d_z * (0 - cs[3])
            T = np.clip(T, 0, np.inf)
            if T < 1e-3:
                K_p_delta = 0
                K_d_delta = 0
                theta_cmd = 0.0
            else:
                K_p_delta = omega_n_delta**2 / C(T)
                K_d_delta = 2 * xi_delta * omega_n_delta / C(T)
                K_p_x = m*omega_n_x**2/T
                K_d_x = 2 * m* xi_x * omega_n_x / T
                theta_cmd = -K_p_x * cs[0] - K_d_x * cs[2]
                theta_cmd = np.clip(theta_cmd, np.radians(-20), np.radians(20))
            delta = -K_p_delta* (cs[4]-theta_cmd) - K_d_delta*cs[5]
            delta = np.clip(delta, np.radians(-10), np.radians(10))

        Thrust.append(T)
        delta_angles.append(delta)
        theta_angles.append(state[4])
        horizontal_disposition.append(state[0])

        theta_pre = state[4]
        state = RK4(state, T, delta, dt)

        if use_ekf:
            # IMU reading: true specific force over the step plus white noise.
            a_true = np.array([(T/m) * np.sin(theta_pre + delta),
                               (T/m) * np.cos(theta_pre + delta)])
            a_meas = a_true + rng.normal(0, SIGMA_A, 2)
            xhat, P = ekf_predict(xhat, P, a_meas, dt)

            # Intermittent altimeter fix on the true (post-step) altitude.
            if (i + 1) % ALT_PERIOD == 0:
                z_meas = state[1] + rng.normal(0, SIGMA_Z)
                if record:
                    # Innovation and its covariance before the correction.
                    # H = [0,1,0,0], so H P H^T is just P[1,1]. The signed
                    # normalized innovation nu = y/sqrt(S) should be standard
                    # normal; its square is the NIS (chi-square, 1 DOF).
                    S_pre = P[1, 1] + SIGMA_Z**2
                    hist['nu'].append((z_meas - xhat[1]) / np.sqrt(S_pre))
                    hist['nis_t'].append(i + 1)
                xhat, P = ekf_update_altimeter(xhat, P, z_meas)

    result = (time, height, Thrust, delta_angles, theta_angles,
              horizontal_disposition, state, touchdown_idx, touchdown_state,
              below_floor)
    if record:
        return result + (hist,)
    return result

    
def plot(initial_state, name):
    time, height, Thrust, delta_angles, theta_angles, horizontal_disposition, state, td_idx, td_state, below_floor = simulate(initial_state)

    td_t = time[td_idx] if td_idx is not None else None

    fig, axs = plt.subplots(
        2, 2,
        figsize=(12, 5)
    )

    fig.suptitle(
        rf"Initial State: x={initial_state[0]}, z={initial_state[1]}, "
        rf"$v_x$={initial_state[2]}, $v_z$={initial_state[3]}, "
        rf"$\vartheta$={np.rad2deg(initial_state[4])}, "
        rf"$\omega$={np.rad2deg(initial_state[5])}"
    )
    plt.subplots_adjust(top=0.9)

    # -------------------------------------------------
    # Height
    # -------------------------------------------------
    axs[0][0].plot(time, np.clip(height, 0, None), label='Height over Surface', color=Color("RWTH"))
    axs[0][0].plot(time, np.zeros_like(time), color='black', linestyle='--', label='Surface')

    if td_t is not None:
        axs[0][0].axvline(td_t, color=Color("Grün"), ls=':', lw=1)

    axs[0][0].set_ylabel("Height (m)")
    axs[0][0].set_title("Height")
    #axs[0][0].legend(loc = "upper right")
    axs[0][0].grid()

    # -------------------------------------------------
    # Thrust
    # -------------------------------------------------
    axs[1][0].plot(time, Thrust, label='Thrust', color=Color("RWTH"))
    axs[1][0].plot(
        time,
        m * g * np.ones_like(time),
        label='Hover Thrust',
        color=Color("Bordeaux"),
        ls="--"
    )

    if td_t is not None:
        axs[1][0].axvline(td_t, color=Color("Grün"), ls=':', lw=1)

    axs[1][0].set_ylabel("Thrust (N)")
    axs[1][0].set_title("Thrust")
    #axs[1][0].legend(loc = "upper right")
    axs[1][0].grid()

    # -------------------------------------------------
    # Angles
    # -------------------------------------------------
    axs[0][1].plot(time, np.rad2deg(delta_angles), label=r'$\delta$', color=Color("RWTH"))
    axs[0][1].plot(time, np.rad2deg(theta_angles), label=r'$\vartheta$', color=Color("Bordeaux"))

    if td_t is not None:
        axs[0][1].axvline(td_t, color=Color("Grün"), ls=':', lw=1)

    axs[0][1].set_ylabel("Angle (°)")
    axs[0][1].set_title("Angles")
    axs[0][1].legend(loc = "upper right")
    axs[0][1].grid()

    # -------------------------------------------------
    # Horizontal displacement
    # -------------------------------------------------
    axs[1][1].plot(
        time,
        horizontal_disposition,
        label='Horizontal Disposition x',
        color=Color("RWTH")
    )
    axs[1][1].plot(time, np.zeros_like(time), color="black", linestyle='--')

    if td_t is not None:
        axs[1][1].axvline(td_t, color=Color("Grün"), ls=':', lw=1)

    axs[1][1].set_ylabel("x (m)")
    axs[1][1].set_xlabel("Time (s)")   # only bottom subplot gets x-label
    axs[1][1].set_title("Horizontal Disposition")
    #axs[1][1].legend(loc = "upper right")
    axs[1][1].grid()

    # -------------------------------------------------
    # Touchdown information
    # -------------------------------------------------
    THRESH = {
        'x':     1.0,             # m
        'vx':    1.0,             # m/s
        'vz':    2.0,             # m/s
        'theta': np.radians(10),  # rad
        'omega': np.radians(10),  # rad/s
    }

    if td_state is not None:
        x_td, z_td, vx_td, vz_td, theta_td, omega_td = td_state

        fig.text(
            0.1, 0.005,
            rf"Touchdown ($t$={td_t:.2f}s):",
            ha='left',
            fontsize=14,
            fontweight='bold',
            color='black'
        )

        items = [
            (rf"$x$={x_td:+.3f}m", abs(x_td) < THRESH['x']),
            (rf"$v_x$={vx_td:+.3f}m/s", abs(vx_td) < THRESH['vx']),
            (rf"$v_z$={vz_td:+.3f}m/s", abs(vz_td) < THRESH['vz']),
            (
                rf"$\vartheta$={np.rad2deg(theta_td):+.2f}$^\circ$",
                abs(theta_td) < THRESH['theta']
            ),
            (
                rf"$\omega$={np.rad2deg(omega_td):+.2f}$^\circ$/s",
                abs(omega_td) < THRESH['omega']
            ),
        ]

        x_pos = 0.3
        for text, ok in items:
            color = (0.0, 0.55, 0.0) if ok else (0.85, 0.05, 0.05)

            fig.text(
                x_pos, 0.005,
                text,
                ha='left',
                fontsize=14,
                color=color
            )

            x_pos += 0.15

    else:
        fig.text(
            0.5, 0.005,
            r"No touchdown within simulation time",
            ha='center',
            fontsize=14,
            fontweight='bold',
            color=(0.85, 0.05, 0.05)
        )

    # Layout
    fig.subplots_adjust(bottom=0.08, hspace=0.5)
    plt.tight_layout()
    save_fig(fig, name)
    plt.close(fig)

    return 0

# Example: working landing (lateral offset + initial tilt, corrected in time) and failing one (offset too large to null out the resulting tilt before touchdown).
EXAMPLES = [
    ("hopper_work", np.array([2, 10, 0, 0, np.radians(5), 0])),   # working case
    ("hopper_fail", np.array([20, 10, 0, 0, 0, 0])),              # failing case
]


def monte_carlo(N=200, seed=42, use_ekf=False, sensor_seed=7):
    rng = np.random.default_rng(seed)                 # initial conditions
    sensor_rng = np.random.default_rng(sensor_seed)   # sensor noise (EKF only)

    # Sample initial conditions over plausible operating envelope
    x0     = rng.uniform(-5, 5, N)
    vx0    = rng.uniform(-2, 2, N)
    vz0    = rng.uniform(-3, 0, N)
    theta0 = rng.uniform(-np.radians(10), np.radians(10), N)
    omega0 = rng.uniform(-np.radians(3), np.radians(3), N)

    # Storage
    trajectories = []   # list of (x_traj, z_traj) up to touchdown
    touchdowns = []     # list of dicts or None

    for i in range(N):
        state0 = np.array([x0[i], 10.0, vx0[i], vz0[i], theta0[i], omega0[i]])
        time, height, _, _, _, x_traj, _, td_idx, td_state, _ = simulate(
            state0, use_ekf=use_ekf, rng=sensor_rng)
        
        end = td_idx if td_idx is not None else len(time)
        trajectories.append((np.array(x_traj[:end]), np.array(height[:end])))
        
        if td_state is None:
            touchdowns.append(None)
        else:
            touchdowns.append({
                'x': td_state[0], 'vx': td_state[2], 'vz': td_state[3],
                'theta': td_state[4], 'omega': td_state[5],
            })
    
    return trajectories, touchdowns


def classify_landing(td):
    if td is None:
        return 'fail'
    soft = (abs(td['vx']) < 0.5 and abs(td['vz']) < 1.0 and
            abs(np.rad2deg(td['theta'])) < 5 and abs(td['x']) < 0.5)
    acceptable = (abs(td['vx']) < 1 and abs(td['vz']) < 2.0 and
                  abs(np.rad2deg(td['theta'])) < 10 and abs(td['x']) < 1.0)
    if soft: return 'soft'
    if acceptable: return 'acceptable'
    return 'fail'


def write_touchdowns_csv(touchdowns, filename):
    # Per-run touchdown table. Runs that never touch down are written with empty
    # state fields and outcome 'fail'.
    path = DATA_DIR / filename
    with open(path, "w") as f:
        f.write("run,x,vx,vz,theta_deg,omega_degs,outcome\n")
        for i, td in enumerate(touchdowns):
            outcome = classify_landing(td)
            if td is None:
                f.write(f"{i},,,,,,{outcome}\n")
            else:
                f.write(f"{i},{td['x']:.6f},{td['vx']:.6f},{td['vz']:.6f},"
                        f"{np.rad2deg(td['theta']):.6f},"
                        f"{np.rad2deg(td['omega']):.6f},{outcome}\n")
    return path


def write_summary_csv(td_truth, td_ekf, nees_mean=None, nis_mean=None,
                      filename="summary.csv"):
    # Headline outcome shares for both control modes, plus the filter
    # consistency means. Shares are in percent of N.
    rt, re = landing_rates(td_truth), landing_rates(td_ekf)
    Nt = rt['N']

    def pct(n):
        return 100.0 * n / Nt

    path = DATA_DIR / filename
    with open(path, "w") as f:
        f.write("metric,perfect_state,ekf_estimate\n")
        f.write(f"N,{Nt},{re['N']}\n")
        f.write(f"soft_pct,{pct(rt['soft']):.3f},{pct(re['soft']):.3f}\n")
        f.write("within_tolerance_pct,"
                f"{pct(rt['soft']+rt['acceptable']):.3f},"
                f"{pct(re['soft']+re['acceptable']):.3f}\n")
        f.write(f"failure_pct,{pct(rt['fail']):.3f},{pct(re['fail']):.3f}\n")
        if nees_mean is not None:
            f.write(f"nees_mean,{nees_mean:.3f},\n")
        if nis_mean is not None:
            f.write(f"nis_mean,{nis_mean:.3f},\n")
    return path


def plot_monte_carlo(trajectories, touchdowns):
    N = len(touchdowns)

    classes = [classify_landing(td) for td in touchdowns]
    n_soft = classes.count('soft')
    n_acc = classes.count('acceptable')
    n_fail = classes.count('fail')
    
    color_map = {'soft': Color("Grün"), 'acceptable': Color("Orange"), 'fail': Color("Rot")}
    
    # ----- Figure 1: trajectory fan -----
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    for (x_traj, z_traj), cls in zip(trajectories, classes):
        ax1.plot(x_traj, z_traj, color=color_map[cls], alpha=0.25, linewidth=0.6)
    ax1.scatter([0], [0], marker='X', s=200, color='black', zorder=10, label='Target')
    ax1.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax1.set_xlabel(r'$x$ (m)')
    ax1.set_ylabel(r'$z$ (m)')
    ax1.set_title(rf'Monte Carlo trajectories ($N={N}$)')
    ax1.grid(True, alpha=0.3)
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=Color("Grün"), lw=2, label=f'Soft ({n_soft})'),
        Line2D([0], [0], color=Color("Orange"), lw=2, label=f'Acceptable ({n_acc})'),
        Line2D([0], [0], color=Color("Rot"), lw=2, label=f'Fail ({n_fail})'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    save_fig(fig1, "mc_trajectories")
    plt.close(fig1)

    # ----- Figure 2: theta vs vx scatter at touchdown -----
    import matplotlib.patches as patches
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    # Threshold boxes
    ax2.add_patch(patches.Rectangle((-5, -0.5), 10, 1.0,
                  facecolor=Color("Grün"), alpha=0.15, label='Soft threshold'))
    ax2.add_patch(patches.Rectangle((-10, -1.0), 20, 2.0,
                  fill=False, edgecolor=Color("Orange"), linewidth=1.5, 
                  linestyle='--', label='Acceptable threshold'))
    
    # Scatter
    for td, cls in zip(touchdowns, classes):
        if td is None: continue
        ax2.scatter(np.rad2deg(td['theta']), td['vx'], 
                    c=[color_map[cls]], s=25, alpha=0.7, edgecolors='none')
    
    ax2.set_xlabel(r'Touchdown $\vartheta$ ($^\circ$)')
    ax2.set_ylabel(r'Touchdown $v_x$ (m/s)')
    ax2.set_title(rf'Touchdown distribution ($N={N}$)')
    ax2.set_xlim(-20, 20)
    ax2.set_ylim(-2, 2)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axvline(0, color='black', linewidth=0.5)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    plt.tight_layout()
    save_fig(fig2, "mc_touchdown")
    plt.close(fig2)

    # ----- Print statistics -----
    print(f"\n=== Monte Carlo Results (N={N}) ===")
    print(f"  Soft landings:       {n_soft}/{N} ({100*n_soft/N:.1f}%)")
    print(f"  Acceptable landings: {n_soft + n_acc}/{N} ({100*(n_soft+n_acc)/N:.1f}%)")
    print(f"  Failures:            {n_fail}/{N} ({100*n_fail/N:.1f}%)")
    
    # Percentile statistics for landed cases
    valid = [td for td in touchdowns if td is not None]
    if valid:
        vx_abs = np.abs([td['vx'] for td in valid])
        vz_abs = np.abs([td['vz'] for td in valid])
        theta_abs_deg = np.abs([np.rad2deg(td['theta']) for td in valid])
        x_abs = np.abs([td['x'] for td in valid])
        
        print(f"\n  95th percentile of touchdown values:")
        print(f"    |v_x|   = {np.percentile(vx_abs, 95):.3f} m/s")
        print(f"    |v_z|   = {np.percentile(vz_abs, 95):.3f} m/s")
        print(f"    |theta| = {np.percentile(theta_abs_deg, 95):.2f} deg")
        print(f"    |x|     = {np.percentile(x_abs, 95):.3f} m")


def landing_rates(touchdowns):
    classes = [classify_landing(td) for td in touchdowns]
    N = len(classes)
    n_soft = classes.count('soft')
    n_acc = classes.count('acceptable')
    n_fail = classes.count('fail')
    return {'N': N, 'soft': n_soft, 'acceptable': n_acc, 'fail': n_fail}


def compare_truth_vs_ekf(N=1000, seed=42, sensor_seed=7):
    # Same initial conditions for both modes; only the state fed to the
    # controller differs (truth vs EKF estimate).
    _, td_truth = monte_carlo(N=N, seed=seed, use_ekf=False)
    _, td_ekf   = monte_carlo(N=N, seed=seed, use_ekf=True, sensor_seed=sensor_seed)

    rt = landing_rates(td_truth)
    re = landing_rates(td_ekf)

    def pct(n):
        return 100 * n / N

    print(f"\n=== Closed-loop landing: perfect state vs EKF estimate (N={N}) ===")
    print(f"{'':22}{'perfect state':>15}{'EKF estimate':>15}")
    print(f"{'Soft landings':22}{pct(rt['soft']):>14.1f}%{pct(re['soft']):>14.1f}%")
    print(f"{'Acceptable (or soft)':22}"
          f"{pct(rt['soft']+rt['acceptable']):>14.1f}%"
          f"{pct(re['soft']+re['acceptable']):>14.1f}%")
    print(f"{'Failures':22}{pct(rt['fail']):>14.1f}%{pct(re['fail']):>14.1f}%")
    print(f"\nSoft-landing rate change with EKF: "
          f"{pct(re['soft']) - pct(rt['soft']):+.1f} percentage points")
    return td_truth, td_ekf


def plot_estimation_error(state0, seed=1):
    # Single closed-loop run flown on the EKF estimate; compares the estimate
    # against truth for the four estimated states.
    out = simulate(state0, use_ekf=True, rng=np.random.default_rng(seed), record=True)
    time = out[0]
    td_idx = out[7]
    hist = out[10]

    end = td_idx if td_idx is not None else len(time)
    t = np.array(time[:end])

    fields = [('x', 'true_x', 'est_x', 'x (m)'),
              ('z', 'true_z', 'est_z', 'z (m)'),
              ('vx', 'true_vx', 'est_vx', r'$v_x$ (m/s)'),
              ('vz', 'true_vz', 'est_vz', r'$v_z$ (m/s)')]

    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    for ax, (_, tk, ek, ylabel) in zip(axs.flat, fields):
        ax.plot(t, np.array(hist[tk][:end]), color=Color("RWTH"), label='truth')
        ax.plot(t, np.array(hist[ek][:end]), color=Color("Bordeaux"),
                ls='--', label='EKF estimate')
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
    fig.suptitle("EKF estimate vs truth (closed loop on estimate)")
    plt.tight_layout()
    save_fig(fig, "ekf_estimate_vs_truth")
    plt.close(fig)


def plot_compare_bars(td_truth, td_ekf):
    # Grouped bar chart of landing outcomes: perfect state (solid) vs EKF
    # estimate (hatched). Regenerates the perfect-vs-EKF figure used in the report.
    rt = landing_rates(td_truth)
    re = landing_rates(td_ekf)
    Nt = rt['N']

    # Three outcome shares in percent: soft, within tolerance (soft or
    # acceptable), and failure.
    labels = ['Soft', 'Within\ntolerance', 'Failure']
    perfect = [100*rt['soft']/Nt,
               100*(rt['soft'] + rt['acceptable'])/Nt,
               100*rt['fail']/Nt]
    ekf = [100*re['soft']/Nt,
           100*(re['soft'] + re['acceptable'])/Nt,
           100*re['fail']/Nt]

    # Percent sign is a comment character under usetex, so escape it there.
    pct = r'\%' if _HAS_LATEX else '%'

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, perfect, w, color=Color("RWTH"), label='Perfect state')
    b2 = ax.bar(x + w/2, ekf, w, color=Color("RWTHlight"),
                hatch='//', edgecolor=Color("RWTH"), label='EKF estimate')

    # Percentage label on top of every bar.
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.1f}{pct}",
                        (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=11,
                        xytext=(0, 2), textcoords='offset points')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f'Share of landings ({pct})')
    ax.set_ylim(0, 105)
    ax.set_title(rf'Landing outcomes: perfect state vs EKF ($N={Nt}$)')
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "mc_perfect_vs_ekf")
    plt.close(fig)


def ekf_consistency(N=300, seed=2024, plot_result=True):
    # Statistical validation of the EKF via NEES and NIS. For a consistent
    # filter the 4-state NEES  e^T P^-1 e  is chi-square with 4 DOF (mean 4),
    # and the scalar altimeter NIS is chi-square with 1 DOF (mean 1). This
    # checks the estimate against the *covariance the filter reports*, which is
    # a stronger statement than the estimate merely tracking the truth.
    import math

    rng_ic = np.random.default_rng(seed)
    nees_all = []      # per-step 4-state NEES, pooled over the ensemble
    nu_all = []        # per-update normalized altimeter innovation, pooled
    traj_nees = []     # per-trajectory mean NEES (one independent sample per run)
    traj_nis = []      # per-trajectory mean NIS

    for k in range(N):
        # One dispersed initial condition from the Monte Carlo envelope.
        s0 = np.array([rng_ic.uniform(-5, 5), 10.0,
                       rng_ic.uniform(-2, 2), rng_ic.uniform(-3, 0),
                       rng_ic.uniform(-np.radians(10), np.radians(10)),
                       rng_ic.uniform(-np.radians(3), np.radians(3))])
        out = simulate(s0, use_ekf=True,
                       rng=np.random.default_rng(10_000 + k), record=True)
        td_idx, hist = out[7], out[10]
        end = td_idx if td_idx is not None else len(out[0])

        # NEES over the descent, decimated every 10 steps to limit the serial
        # correlation between neighbouring samples.
        tn = []
        for j in range(0, end, 10):
            e = np.array([hist['true_x'][j]  - hist['est_x'][j],
                          hist['true_z'][j]  - hist['est_z'][j],
                          hist['true_vx'][j] - hist['est_vx'][j],
                          hist['true_vz'][j] - hist['est_vz'][j]])
            val = float(e @ np.linalg.solve(hist['P'][j], e))
            nees_all.append(val)
            tn.append(val)
        if tn:
            traj_nees.append(np.mean(tn))

        # Altimeter innovations, only for fixes taken before touchdown. The
        # NIS is the square of the normalized innovation.
        tnu = [v for v, ti in zip(hist['nu'], hist['nis_t']) if ti < end]
        nu_all.extend(tnu)
        if tnu:
            traj_nis.append(np.mean(np.square(tnu)))

    nees_all = np.array(nees_all)
    nu_all = np.array(nu_all)
    tnf = np.array(traj_nees)
    tni = np.array(traj_nis)

    # 95% band for the mean of M independent chi-square(k) samples (CLT):
    # a chi-square(k) has mean k and variance 2k, so the standard error of the
    # mean over M runs is sqrt(2k/M). Per-trajectory means are the independent
    # samples (pooled per-step samples are serially correlated).
    def band(k, M):
        se = math.sqrt(2 * k / M)
        return k - 1.96 * se, k + 1.96 * se

    lo4, hi4 = band(4, len(tnf))
    lo1, hi1 = band(1, len(tni))
    ok4 = lo4 <= tnf.mean() <= hi4
    ok1 = lo1 <= tni.mean() <= hi1

    print(f"\n=== EKF consistency (N={N} dispersed closed-loop descents) ===")
    print(f"  4-state NEES: mean {tnf.mean():.3f}  (ideal 4.0, 95% band "
          f"[{lo4:.3f}, {hi4:.3f}])  -> {'consistent' if ok4 else 'CHECK'}")
    print(f"  Altimeter NIS: mean {tni.mean():.3f}  (ideal 1.0, 95% band "
          f"[{lo1:.3f}, {hi1:.3f}])  -> {'consistent' if ok1 else 'CHECK'}")

    if plot_result:
        # Chi-square(k) density in closed form, to overlay on the histograms.
        def chi2_pdf(x, k):
            return x**(k/2 - 1) * np.exp(-x/2) / (2**(k/2) * math.gamma(k/2))

        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

        # NEES histogram against chi-square(4).
        axs[0].hist(nees_all, bins=60, density=True, color=Color("RWTHlight"),
                    edgecolor='none')
        xx = np.linspace(1e-3, np.percentile(nees_all, 99.5), 400)
        axs[0].plot(xx, chi2_pdf(xx, 4), color=Color("Bordeaux"), lw=2,
                    label=r'$\chi^2_4$')
        axs[0].axvline(nees_all.mean(), color=Color("RWTH"), ls='--',
                       label=f'mean {nees_all.mean():.2f}')
        axs[0].set_xlabel('4-state NEES')
        axs[0].set_ylabel('density')
        axs[0].set_title('State consistency (NEES)')
        axs[0].legend(loc='upper right')
        axs[0].grid(True, alpha=0.3)

        # Normalized altimeter innovation against the standard normal. This is
        # the readable form of the scalar NIS check: a consistent filter makes
        # nu = y/sqrt(S) standard normal (unit variance, zero mean).
        axs[1].hist(nu_all, bins=60, density=True, color=Color("RWTHlight"),
                    edgecolor='none')
        xx = np.linspace(-4, 4, 400)
        axs[1].plot(xx, np.exp(-xx**2 / 2) / np.sqrt(2 * np.pi),
                    color=Color("Bordeaux"), lw=2, label=r'$\mathcal{N}(0,1)$')
        axs[1].set_xlabel('normalized altimeter innovation')
        axs[1].set_ylabel('density')
        axs[1].set_title(rf'Innovation consistency (NIS mean {np.mean(np.square(nu_all)):.2f})')
        axs[1].legend(loc='upper right')
        axs[1].grid(True, alpha=0.3)

        fig.suptitle('EKF consistency: empirical vs theoretical chi-square')
        plt.tight_layout()
        save_fig(fig, "ekf_consistency")
        plt.close(fig)

    return {'nees_mean': float(tnf.mean()), 'nis_mean': float(tni.mean())}


def main():
    # Propagator sanity checks against the analytic limits.
    test_dynamics()

    # Two demonstration descents: a working landing and a failing one.
    for name, scenario in EXAMPLES:
        plot(scenario, name)

    # Baseline Monte Carlo flown on perfect state.
    trajectories, touchdowns = monte_carlo(N=1000)
    plot_monte_carlo(trajectories, touchdowns)

    # EKF: single-run estimate-vs-truth diagnostic, then the full comparison.
    plot_estimation_error(np.array([3.0, 10.0, 0.0, 0.0, np.radians(3), 0.0]))
    td_truth, td_ekf = compare_truth_vs_ekf(N=1000)
    plot_compare_bars(td_truth, td_ekf)
    write_touchdowns_csv(td_truth, "mc_touchdowns_perfect.csv")
    write_touchdowns_csv(td_ekf, "mc_touchdowns_ekf.csv")

    # Statistical validation of the filter (NEES / NIS).
    cons = ekf_consistency()

    write_summary_csv(td_truth, td_ekf,
                      nees_mean=cons['nees_mean'], nis_mean=cons['nis_mean'])


if __name__ == "__main__":
    main()