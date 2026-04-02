import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time

# solution of the 1D wave equation
# on the interval [0, pi/2] over time [0, 1]
# d_tt u(t, x) = d_xx u(t, x) + 2
# d_t u(0, x) = 0, u(0, x) = cos(x)
# d_x u(t, 0) = 0, u(t, pi/2) = t^2
# analytic solution: u(t, x) = t^2 + cos(t) cos(x)

T = 1
a = 0
b = np.pi / 2
N_time_steps = 28
delta = T / N_time_steps
Nx = 50
hx = (b - a) / (Nx + 1)
r_2 = delta * delta / (hx * hx)
print(f"{Nx=}")
print(f"{N_time_steps=}")
print(f"{r_2=}")

cfl = delta / hx
if cfl > 1:
    print(
        f"\n=============== CFL condition is not satisfied: cfl={cfl:.6e} (should be <= 1) ===============\n"
    )

# unknowns are x_0, ..., x_Nx (right Dirichlet node x=b excluded)
space_grid = np.linspace(a, b - hx, Nx + 1)

# second derivative stencil on unknowns (without 1/hx^2 factor)
D_c_2 = sp.diags(
    [1, -2, 1], offsets=[-1, 0, 1], shape=(Nx + 1, Nx + 1), dtype=None, format="lil"
)

# !! very important: handles d_x u(t, 0) = 0 (see explanation in the writeup)
D_c_2[0, 1] = 2
D_c_2 = D_c_2.tocsr()

force_vec = 2 * np.ones(Nx + 1)
bc_vec = np.zeros(Nx + 1)
bc_vec[-1] = 1

u_0 = np.cos(space_grid)
u_prev = u_0.copy()

t0 = 0

# second order Taylor approximation of u(t, x) around t=0 (see writeup)
g = t0 * t0
lap_u0 = (1 / (hx * hx)) * (D_c_2 @ u_prev + bc_vec * g)
u = u_prev + 0.5 * delta * delta * (lap_u0 + force_vec)

start = time.time()
for i in range(1, N_time_steps):
    tn = i * delta
    g = tn * tn
    lap_u = (1 / (hx * hx)) * (D_c_2 @ u + bc_vec * g)
    u_next = 2 * u - u_prev + delta * delta * (lap_u + force_vec)
    u, u_prev = u_next, u
end = time.time()
print(f"Elapsed time: {end - start:.4f} seconds")

u_num = np.append(u, T * T)
space_grid_full = np.append(space_grid, b)
u_exact = T * T + np.cos(T) * np.cos(space_grid_full)
err = np.linalg.norm(u_num - u_exact, ord=2) * np.sqrt(hx)
print(f"{err=}")

plt.plot(space_grid_full, u_num, label="Numerical solution")
plt.plot(space_grid_full, u_exact, label="Analytic solution", linestyle="--")
plt.legend()
plt.title("Leapfrog for wave equation")
plt.xlabel("x")
plt.ylabel("u(T, x)")
plt.show()
