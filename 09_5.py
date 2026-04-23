import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time

# 5) First-order system form of the wave equation:
# d_t [u, v]^T = [0  I; D2  0] [u, v]^T, where v = d_t u and D2 approximates d_xx.
# Boundary conditions: u(t, -2) = u(t, 2) = 0 (interior unknowns only).
# Initial data: u(0, x) = exp(-100 x^2), v(0, x) = 0.
#
# This script compares:
# 1) Explicit Euler
# 2) Implicit Euler
# 3) Arbitrary ODE solver (solve_ivp / RK45)
# 4) Leapfrog-equivalent (velocity-Verlet / Störmer-Verlet)

T = 1.0
x_left = -2.0
x_right = 2.0
Nx = 200
exponential_factor = 100.0

# Use a safe step for leapfrog-like methods and ODE reference.
N_steps = 400
delta = T / N_steps
hx = (x_right - x_left) / (Nx + 1)
mu = delta / hx

# Deliberately larger time step to test explicit-Euler stability.
N_steps_explicit = 200
delta_explicit = T / N_steps_explicit
mu_explicit = delta_explicit / hx

print(f"{Nx=}")
print(f"{N_steps=}, {delta=:.6e}, {mu=:.6f}")
print(f"{N_steps_explicit=}, {delta_explicit=:.6e}, {mu_explicit=:.6f}")

space_grid = np.linspace(x_left + hx, x_right - hx, Nx)

# Spatial operator for d_xx: (1/hx^2) * tridiag(1,-2,1)
D2 = (1.0 / (hx * hx)) * sp.diags(
    [1.0, -2.0, 1.0], offsets=[-1, 0, 1], shape=(Nx, Nx), format="csr"
)
I = sp.eye(Nx, format="csr")
Z = sp.csr_matrix((Nx, Nx))
A = sp.bmat([[Z, I], [D2, Z]], format="csr")

u0 = np.exp(-exponential_factor * space_grid * space_grid)
v0 = np.zeros_like(u0)
y0 = np.concatenate([u0, v0])


def split_state(y):
    return y[:Nx], y[Nx:]


def state_norm(y):
    u, v = split_state(y)
    # physical energy-like norm
    return np.sqrt(np.linalg.norm(v) ** 2 + np.linalg.norm((1.0 / hx) * np.diff(np.r_[0.0, u, 0.0])) ** 2)


def odd_periodic_extension(y):
    L = x_right - x_left
    y_folded = ((y + L) % (2.0 * L)) - L
    x_abs = np.abs(y_folded) + x_left
    values = np.exp(-exponential_factor * x_abs * x_abs)
    return np.where(y_folded >= 0.0, values, -values)


def exact_u(t):
    y_grid = space_grid - x_left
    return 0.5 * (odd_periodic_extension(y_grid - t) + odd_periodic_extension(y_grid + t))


def explicit_euler():
    y = y0.copy()
    norms = [state_norm(y)]
    start = time.time()
    for _ in range(N_steps_explicit):
        y = y + delta_explicit * (A @ y)
        norms.append(state_norm(y))
    end = time.time()
    return y[:Nx], np.array(norms), end - start


def implicit_euler():
    y = y0.copy()
    norms = [state_norm(y)]
    M = sp.eye(2 * Nx, format="csr") - delta * A
    solve_M = spla.factorized(M.tocsc())
    start = time.time()
    for _ in range(N_steps):
        y = solve_M(y)
        norms.append(state_norm(y))
    end = time.time()
    return y[:Nx], np.array(norms), end - start


def arbitrary_ode_solver():
    def rhs(_, y):
        return A @ y

    start = time.time()
    sol = solve_ivp(
        rhs,
        t_span=(0.0, T),
        y0=y0,
        method="RK45",
        t_eval=np.linspace(0.0, T, N_steps + 1),
        rtol=1e-7,
        atol=1e-9,
    )
    end = time.time()
    norms = np.array([state_norm(sol.y[:, i]) for i in range(sol.y.shape[1])])
    return sol.y[:Nx, -1], norms, end - start


def leapfrog_equivalent_verlet():
    # Equivalent two-step leapfrog written as a first-order symplectic update.
    u = u0.copy()
    v = v0.copy()
    norms = [np.sqrt(np.linalg.norm(v) ** 2 + np.linalg.norm((1.0 / hx) * np.diff(np.r_[0.0, u, 0.0])) ** 2)]
    start = time.time()
    for _ in range(N_steps):
        v_half = v + 0.5 * delta * (D2 @ u)
        u = u + delta * v_half
        v = v_half + 0.5 * delta * (D2 @ u)
        norms.append(
            np.sqrt(
                np.linalg.norm(v) ** 2
                + np.linalg.norm((1.0 / hx) * np.diff(np.r_[0.0, u, 0.0])) ** 2
            )
        )
    end = time.time()
    return u, np.array(norms), end - start


u_exact_T = exact_u(T)

u_exp, norms_exp, time_exp = explicit_euler()
u_imp, norms_imp, time_imp = implicit_euler()
u_rk, norms_rk, time_rk = arbitrary_ode_solver()
u_verlet, norms_verlet, time_verlet = leapfrog_equivalent_verlet()

err_exp = np.linalg.norm(u_exp - u_exact_T)
err_imp = np.linalg.norm(u_imp - u_exact_T)
err_rk = np.linalg.norm(u_rk - u_exact_T)
err_verlet = np.linalg.norm(u_verlet - u_exact_T)

print("\nFinal errors ||u(T)-u_exact(T)||:")
print(f"Explicit Euler: {err_exp:.6e}")
print(f"Implicit Euler: {err_imp:.6e}")
print(f"RK45 (solve_ivp): {err_rk:.6e}")
print(f"Verlet (leapfrog-equivalent): {err_verlet:.6e}")

print("\nEnergy-like norm growth factors (final / initial):")
print(f"Explicit Euler: {norms_exp[-1] / norms_exp[0]:.6e}")
print(f"Implicit Euler: {norms_imp[-1] / norms_imp[0]:.6e}")
print(f"RK45 (solve_ivp): {norms_rk[-1] / norms_rk[0]:.6e}")
print(f"Verlet (leapfrog-equivalent): {norms_verlet[-1] / norms_verlet[0]:.6e}")

print("\nElapsed times:")
print(f"Explicit Euler: {time_exp:.4f} s")
print(f"Implicit Euler: {time_imp:.4f} s")
print(f"RK45 (solve_ivp): {time_rk:.4f} s")
print(f"Verlet (leapfrog-equivalent): {time_verlet:.4f} s")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

axes[0].plot(space_grid, u_exact_T, "k--", label="Exact")
axes[0].plot(space_grid, u_exp, label="Explicit Euler")
axes[0].plot(space_grid, u_imp, label="Implicit Euler")
axes[0].plot(space_grid, u_rk, label="RK45")
axes[0].plot(space_grid, u_verlet, label="Verlet (leapfrog-eq.)")
axes[0].set_title("Final profile u(T, x)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("u")
axes[0].legend()

t_exp = np.linspace(0.0, T, N_steps_explicit + 1)
t_std = np.linspace(0.0, T, N_steps + 1)
axes[1].plot(t_exp, norms_exp / norms_exp[0], label="Explicit Euler")
axes[1].plot(t_std, norms_imp / norms_imp[0], label="Implicit Euler")
axes[1].plot(t_std, norms_rk / norms_rk[0], label="RK45")
axes[1].plot(t_std, norms_verlet / norms_verlet[0], label="Verlet (leapfrog-eq.)")
axes[1].set_title("Stability check (relative norm)")
axes[1].set_xlabel("t")
axes[1].set_ylabel("norm(t) / norm(0)")
axes[1].legend()

plt.tight_layout()
plt.show()

# Separate view without Explicit Euler, to inspect non-blowup methods clearly.
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4.5))

axes2[0].plot(space_grid, u_exact_T, "k--", label="Exact")
axes2[0].plot(space_grid, u_imp, label="Implicit Euler")
axes2[0].plot(space_grid, u_rk, label="RK45")
axes2[0].plot(space_grid, u_verlet, label="Verlet (leapfrog-eq.)")
axes2[0].set_title("Final profile u(T, x) without Explicit Euler")
axes2[0].set_xlabel("x")
axes2[0].set_ylabel("u")
axes2[0].legend()

axes2[1].plot(t_std, norms_imp / norms_imp[0], label="Implicit Euler")
axes2[1].plot(t_std, norms_rk / norms_rk[0], label="RK45")
axes2[1].plot(t_std, norms_verlet / norms_verlet[0], label="Verlet (leapfrog-eq.)")
axes2[1].set_title("Stability check without Explicit Euler")
axes2[1].set_xlabel("t")
axes2[1].set_ylabel("norm(t) / norm(0)")
axes2[1].legend()

plt.tight_layout()
plt.show()
