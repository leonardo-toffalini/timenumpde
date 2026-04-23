import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time

# (a3) Implicit centered:
# u^{n+1} = u^n - mu * D0 u^{n+1}
# (I + mu D0) u^{n+1} = u^n

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

D0 = sp.diags([-0.5, 0.0, 0.5], offsets=[-1, 0, 1], shape=(Nx, Nx), format="csr")
I = sp.eye(Nx, format="csr")
A = I + mu * D0
solve_A = spla.factorized(A.tocsc())

u = np.exp(-exponential_factor * space_grid * space_grid)
initial_peak_x = space_grid[np.argmax(u)]

start = time.time()
for _ in range(N_time_steps):
    # boundary contributions are zero since both Dirichlet boundaries are zero
    u = solve_A(u)
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
plt.title("(a3) Implicit centered difference")
plt.xlabel("x")
plt.ylabel("u(T, x)")
plt.legend()
plt.show()
