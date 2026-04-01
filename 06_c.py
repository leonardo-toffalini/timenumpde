import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time

###### 8/3. a

# solution of the 1D wave problem
# on the interval [-pi/2, pi/2]
# d_tt u = 4 d_xx u
# with u(0, x) = 0, d_t u(0, x) = 2 cos(x),
# u(t, -pi/2) = u(t, pi/2) = 0
# analytic solution: u(t, x) = sin(2t) cos(x)

T = 8
a = -np.pi / 2
b = np.pi / 2
N_time_steps = 500
delta = T / N_time_steps
Nx = 50
hx = (b - a) / (Nx + 1)
r_2 = 4 * delta * delta / (hx * hx)
print(f"{Nx=}")
print(f"{N_time_steps=}")
print(f"{r_2=}")

# second derivative stencil (without 1/hx^2 factor)
D_c_2 = sp.diags([1, -2, 1], offsets=[-1, 0, 1], shape=(Nx, Nx), dtype=None)

space_grid = np.linspace(a + hx, b - hx, Nx)

# initialize
u_0 = 0 * space_grid
u_prev = u_0.copy()
u = u_prev + delta * 2 * np.cos(space_grid)

start = time.time()
for i in range(2, N_time_steps + 1):
    tn = i * delta
    u_next = 2 * u - u_prev + r_2 * D_c_2 @ u

    u, u_prev = u_next, u
    u_exact = np.sin(2 * tn) * np.cos(space_grid)


end = time.time()
print(f"Elapsed time: {end - start:.4f} seconds")

u_exact = np.sin(2 * T) * np.cos(space_grid)
err = np.linalg.norm(u - u_exact)
print(f"{err=}")

plt.plot(u, label="Numerical solution")
plt.plot(
    u_exact,
    label="Analytic solution",
    linestyle="--",
)
plt.legend()
plt.title("Leapfrog for wave equation")
plt.show()
