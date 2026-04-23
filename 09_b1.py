import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time

# (b1) Two-step wave scheme:
# u^{n+1} = -u^{n-1} + 2u^n + mu^2 * D0^2 u^n
# for u_tt = u_xx with
# u(0, x) = exp(-100 x^2), u_t(0, x) = 0, u(t, -2) = u(t, 2) = 0.

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
L = x_right - x_left

# D0^2 represented with the second-difference stencil (without 1/hx^2):
D2 = sp.diags([1.0, -2.0, 1.0], offsets=[-1, 0, 1], shape=(Nx, Nx), format="csr")

u0 = np.exp(-exponential_factor * space_grid * space_grid)
u_prev = u0.copy()
initial_peak_x = space_grid[np.argmax(u0)]

# Start step from u_t(0, x) = 0 using Taylor expansion.
u = u0 + 0.5 * (mu * mu) * (D2 @ u0)

start = time.time()
for _ in range(1, N_time_steps):
    u_next = -u_prev + 2.0 * u + (mu * mu) * (D2 @ u)
    u_prev, u = u, u_next
end = time.time()
print(f"Elapsed time: {end - start:.4f} seconds")

def odd_periodic_extension(y):
    y_folded = ((y + L) % (2.0 * L)) - L
    x_abs = np.abs(y_folded) + x_left
    values = np.exp(-exponential_factor * x_abs * x_abs)
    return np.where(y_folded >= 0.0, values, -values)

y_grid = space_grid - x_left
u_exact = 0.5 * (
    odd_periodic_extension(y_grid - T) + odd_periodic_extension(y_grid + T)
)
err = np.linalg.norm(u - u_exact)
print(f"{err=}")
print(f"max|u(T)| = {np.max(np.abs(u)):.6e}")

# Use the right-moving half-wave to estimate speed from displacement.
right_mask = space_grid >= 0.0
final_peak_x_right = space_grid[right_mask][np.argmax(np.abs(u[right_mask]))]
prop_speed_est = (final_peak_x_right - initial_peak_x) / T
print(f"Estimated propagation speed: {prop_speed_est:.6f}")

plt.plot(space_grid, u, label="Numerical profile u(T, x)")
plt.plot(space_grid, u_exact, "--", label="Exact solution u(T, x)")
plt.title("(b1) Two-step wave scheme")
plt.xlabel("x")
plt.ylabel("u")
plt.legend()
plt.show()
