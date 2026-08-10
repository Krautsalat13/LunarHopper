#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:36:52 2026

@author: tamilarasan
"""
import shutil
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

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

def simulate(state, t = 20, dt = 0.01):
    height = []
    time = []
    Thrust = []
    delta_angles = []
    theta_angles = []
    horizontal_disposition = []
    below_floor = []
    
    engine_off = False        
    touchdown_idx = None
    touchdown_state = None
    for i in range(int(t/dt)):
        height.append(state[1])
        time.append(i*dt)
        below_floor.append(state[1] < 0)
        
 
        if not engine_off and state[1] <= 0.10 and i > 0:
            engine_off = True
        

        if touchdown_idx is None and state[1] <= 0 and i > 0:
            touchdown_idx = i
            touchdown_state = state.copy()
        
        if engine_off:
            T = 0
            delta = 0
        else:
            T = m*g + K_p_z * (0 - state[1]) + K_d_z * (0 - state[3])
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
                theta_cmd = -K_p_x * state[0] - K_d_x * state[2]
                theta_cmd = np.clip(theta_cmd, np.radians(-20), np.radians(20))
            delta = -K_p_delta* (state[4]-theta_cmd) - K_d_delta*state[5]
            delta = np.clip(delta, np.radians(-10), np.radians(10))
        
        Thrust.append(T)
        delta_angles.append(delta)
        theta_angles.append(state[4])
        horizontal_disposition.append(state[0])
        
        state = RK4(state, T, delta, dt)
    return time, height, Thrust, delta_angles, theta_angles, horizontal_disposition, state, touchdown_idx, touchdown_state, below_floor

    
def plot(initial_state):
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
    x0, z0, vx0, vz0, theta0, omega0 = initial_state
    case_name = (
        f"hopper_case_x{x0:g}_z{z0:g}_vx{vx0:g}_vz{vz0:g}"
        f"_theta{np.rad2deg(theta0):g}deg_omega{np.rad2deg(omega0):g}degs.png"
    )
    plt.savefig(OUTPUT_DIR / case_name, dpi=300, bbox_inches='tight')

    plt.show()

    return 0

# Example: working landing (lateral offset + initial tilt, corrected in time) and failing one (offset too large to null out the resulting tilt before touchdown).
EXAMPLES = [
    np.array([2, 10, 0, 0, np.radians(5), 0]),   # working case
    np.array([20, 10, 0, 0, 0, 0]),              # failing case
]

if __name__ == "__main__":
    for scenario in EXAMPLES:
        plot(scenario)


def monte_carlo(N=200, seed=42):
    rng = np.random.default_rng(seed)
    
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
        time, height, _, _, _, x_traj, _, td_idx, td_state, _ = simulate(state0)
        
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


def plot_monte_carlo(trajectories, touchdowns):
    N = len(touchdowns)
    
    # Classify each landing
    def classify(td):
        if td is None:
            return 'fail'
        soft = (abs(td['vx']) < 0.5 and abs(td['vz']) < 1.0 and 
                abs(np.rad2deg(td['theta'])) < 5 and abs(td['x']) < 0.5)
        acceptable = (abs(td['vx']) < 1 and abs(td['vz']) < 2.0 and 
                      abs(np.rad2deg(td['theta'])) < 10 and abs(td['x']) < 1.0)
        if soft: return 'soft'
        if acceptable: return 'acceptable'
        return 'fail'
    
    classes = [classify(td) for td in touchdowns]
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
    plt.savefig(OUTPUT_DIR / f"monte_carlo_trajectories_N{N}.png", dpi=300, bbox_inches='tight')
    plt.show()

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
    plt.savefig(OUTPUT_DIR / f"monte_carlo_touchdown_N{N}.png", dpi=300, bbox_inches='tight')
    plt.show()

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


if __name__ == "__main__":
    trajectories, touchdowns = monte_carlo(N=1000)

    plot_monte_carlo(trajectories, touchdowns)