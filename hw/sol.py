import numpy as np
import matplotlib.pyplot as plt
import time

# solution of the 1D wave equation
# on the interval (0,pi/2) over time (0,1)
# d_tt u(t,x) =  d_xx u(t,x) + 2
# dt u(u, x) = 0
# u(u, x) = cos x
# dx u(t, 0) = 0
# u(t, pi/2) = t^2

# analytic solution: u(t, x) = t^2 + cos t cos x

# time discretization
T = 1
N_time_steps = 28
delta = T / N_time_steps

# space discretization
Nx = 50
interval_end = np.pi / 2
hx = interval_end / (Nx + 1)

# wave CFL number for u_tt = u_xx
lam = delta / hx
print(f"{lam=}")


def rhs_force(t):
    return 2


def right_dirichlet(t):
    return t * t


def laplacian_with_mixed_bc(u):
    lap = np.zeros_like(u)

    # interior points: d_xx u(t, x) = 1/h^2 (u[i+1] - 2u[i] + u[i-1])
    lap[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / (hx * hx)

    # left boundary: d_x u(t,0) = 0
    lap[0] = 2 * (u[1] - u[0]) / (hx * hx)
    return lap


# full grid including boundary points x=0 and x=pi/2
space_disc = np.linspace(0, interval_end, Nx + 2)
time_disc = np.linspace(0, T, N_time_steps + 1)

# initialize with u(0,x)=cos(x), u_t(0,x)=0
u_0 = np.cos(space_disc)
u_0[-1] = right_dirichlet(time_disc[0])
u_prev = u_0.copy()

# second-order start value
t0 = time_disc[0]
u = u_0 + 0.5 * delta * delta * (laplacian_with_mixed_bc(u_0) + rhs_force(t0))
u[-1] = right_dirichlet(time_disc[1])

start = time.time()
for n in range(1, N_time_steps):
    t_n = time_disc[n]
    lap = laplacian_with_mixed_bc(u)
    u_next = 2 * u - u_prev + delta * delta * (lap + rhs_force(t_n))
    u_next[-1] = right_dirichlet(time_disc[n + 1])

    u, u_prev = u_next, u
end = time.time()
print(f"Elapsed time: {end - start:.4f} seconds")

u_analytic = T * T + np.cos(T) * np.cos(space_disc)
print(f"max_abs_error = {np.max(np.abs(u - u_analytic)):.4e}")

plt.plot(space_disc, u, label="Numerical solution at T=1")
plt.plot(space_disc, u_analytic, "--", label="Analytic solution at T=1")
plt.legend()
plt.title("Wave equation with centered second-order scheme")
plt.xlabel("x")
plt.ylabel("u(T,x)")
plt.show()
