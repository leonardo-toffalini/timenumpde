import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time

# (a2) Forward Euler in time + backward (upwind) difference:
# u^{n+1} = u^n - mu * D_- u^n

T = 1.0
x_left = -2.0
x_right = 2.0
Nx = 200
N_time_steps = 80
exponential_factor = 100.0

delta = T / N_time_steps
hx = (x_right - x_left) / (Nx + 1)
mu = delta / hx

print(f"{Nx=}")
print(f"{N_time_steps=}")
print(f"{mu=}")

space_grid = np.linspace(x_left + hx, x_right - hx, Nx)

D_minus = sp.diags([-1.0, 1.0], offsets=[-1, 0], shape=(Nx, Nx), format="csr")

u = np.exp(-exponential_factor * space_grid * space_grid)
initial_peak_x = space_grid[np.argmax(u)]
u_left = 0.0

start = time.time()
for _ in range(N_time_steps):
    g_minus = np.zeros(Nx)
    g_minus[0] = -u_left
    u = u - mu * (D_minus @ u + g_minus)
end = time.time()
print(f"Elapsed time: {end - start:.4f} seconds")

x_back = space_grid - T
u_exact = np.where(x_back >= x_left, np.exp(-exponential_factor * x_back * x_back), 0.0)
err = np.linalg.norm(u - u_exact)
print(f"{err=}")
final_peak_x = space_grid[np.argmax(u)]
prop_speed = (final_peak_x - initial_peak_x) / T
print(f"Estimated propagation speed: {prop_speed:.6f}")

plt.plot(space_grid, u, label="Numerical solution")
plt.plot(space_grid, u_exact, "--", label="Exact transport profile")
plt.title("(a2) Forward Euler + upwind difference")
plt.xlabel("x")
plt.ylabel("u(T, x)")
plt.legend()
plt.show()
