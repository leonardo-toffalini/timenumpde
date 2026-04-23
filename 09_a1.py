import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time

# (a1) Leapfrog in time + centered difference in space:
# u^{n+1} = u^{n-1} - mu * D0 u^n
# Problem data on x in [-2, 2], t in [0, 8]:
# u(0, x) = exp(-100 x^2), u(t, -2) = u(t, 2) = 0

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

# D0 without division by hx:
# (D0 u)_j = (u_{j+1} - u_{j-1}) / 2
D0 = sp.diags([-0.5, 0.0, 0.5], offsets=[-1, 0, 1], shape=(Nx, Nx), format="csr")

u0 = np.exp(-exponential_factor * space_grid * space_grid)
initial_peak_x = space_grid[np.argmax(u0)]
u_prev = u0.copy()

# Bootstrap first step with FTBS (consistent inflow from left boundary).
u_left = 0.0
D_minus = sp.diags([-1.0, 1.0], offsets=[-1, 0], shape=(Nx, Nx), format="csr")
g_minus = np.zeros(Nx)
g_minus[0] = -u_left
u = u_prev - mu * (D_minus @ u_prev + g_minus)

start = time.time()
for _ in range(1, N_time_steps):
    g0 = np.zeros(Nx)  # both boundaries are zero
    u_next = u_prev - mu * (D0 @ u + g0)
    u_prev, u = u, u_next
end = time.time()
print(f"Elapsed time: {end - start:.4f} seconds")

# Exact transport profile with zero inflow at x = -2.
x_back = space_grid - T
u_exact = np.where(x_back >= x_left, np.exp(-exponential_factor * x_back * x_back), 0.0)
err = np.linalg.norm(u - u_exact)
print(f"{err=}")
final_peak_x = space_grid[np.argmax(u)]
prop_speed = (final_peak_x - initial_peak_x) / T
print(f"Estimated propagation speed: {prop_speed:.6f}")

plt.plot(space_grid, u, label="Numerical solution")
plt.plot(space_grid, u_exact, "--", label="Exact transport profile")
plt.title("(a1) Leapfrog + centered difference")
plt.xlabel("x")
plt.ylabel("u(T, x)")
plt.legend()
plt.show()
