import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time

# solution of the 1D advection problem
# on the interval (0,3) over time (0,2)
# d_t u + 2 d_x u = 0.1 d_xx u
# u(t, 0) = e^(-0.1 t) sin(-2t)
# u(t, 3) = e^(-0.1 t) sin (3 - 2t)
# u(0, x) = sin x
# analytic solution: u(t, x) = e^(-0.1 t) sin(x - 2t)

T = 2
b = 3
N_time_steps = 500
delta = T / N_time_steps
Nx = 100
hx = b / (Nx + 1)
mu = 2 * delta / hx
rx = 0.1 * delta / (hx * hx)
print(f"{Nx=}")
print(f"{N_time_steps=}")
print(f"{mu=}")
print(f"{rx=}")
print(f"{mu + 2 * rx=}")

# basic matrices
D_upw = sp.diags([-1, 1], offsets=[-1, 0], shape=(Nx, Nx), dtype=None)  # / hx
# D_c is not used in the code, but translated for completeness
D_c = sp.diags([-1, 1], offsets=[-1, 1], shape=(Nx, Nx), dtype=None)  # / hx
D_c_2 = sp.diags([1, -2, 1], offsets=[-1, 0, 1], shape=(Nx, Nx), dtype=None)  # / hx
I = sp.eye(Nx)

space_grid = np.linspace(hx, b - hx, Nx)
init_cond = np.sin(space_grid)
# plt.plot(init_cond, label="initial condition")
# plt.legend()
# plt.show()

# time step: explicit Euler
# space step: upwind + central

# initialize
u = init_cond.copy()

start = time.time()
for i in range(N_time_steps):
    t_n = i * delta

    # boundary handling
    # advection
    g_a = np.zeros_like(u)
    g_a[0] = mu * np.exp(-0.1 * t_n) * np.sin(-2 * t_n)

    # diffusion
    g_d = np.zeros_like(u)
    g_d[0] = rx * np.exp(-0.1 * t_n) * np.sin(-2 * t_n)
    g_d[-1] = rx * np.exp(-0.1 * t_n) * np.sin(3 - 2 * t_n)

    u = u - mu * D_upw @ u + rx * D_c_2 @ u + g_a + g_d

end = time.time()
print(f"Elapsed time: {end - start:.4f} seconds")

u_exact = np.exp(-0.1 * T) * np.sin(space_grid - 2 * T)
err = np.linalg.norm(u - u_exact)
print(f"{err=}")

plt.plot(u, label="Numerical solution")
u_exact = np.exp(-0.1 * T) * np.sin(space_grid - 2 * T)
plt.plot(
    u_exact,
    label="Analytic solution",
    linestyle="--",
)
plt.legend()
plt.title("Explicit Euler + upwind")
plt.show()
