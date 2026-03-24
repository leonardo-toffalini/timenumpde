import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time

# solution of the 1D advection problem
# on the interval (0,3) over time (0,2)
# d_t u(t,x) +  2 d_x u(t,x) = 0
# u(0,x) = sin x;
# u(t,0) = sin -2t;
# analytic solution: sin(x-2t)

T = 2
N_time_steps = 1000
delta = T / N_time_steps
Nx = 50
hx = 3 / (Nx + 1)
mu = 2 * delta / hx
r = 0.1 * delta / (hx * hx)
print(f"{mu=}")
print(f"{r=}")
print(f"{mu + 2 * r = }")

# basic matrices
D_upw = sp.diags([-1, 1], offsets=[-1, 0], shape=(Nx, Nx)) / hx
D_c = sp.diags([-1, 1], offsets=[-1, 1], shape=(Nx, Nx))  # / hx
D_c_2 = sp.diags([1, -2, 1], offsets=[-1, 0, 1], shape=(Nx, Nx))  # / hx
I = sp.eye(Nx)

space_disc = np.linspace(hx, 3 - hx, Nx)
init_cond = np.sin(space_disc)
plt.plot(init_cond)
plt.show()

# time step: explicit Euler
# values on the left-hand side
time_disc = np.linspace(0, T - delta, N_time_steps)
space_disc = np.linspace(hx, 3 - hx, Nx)
u_left = np.exp(-0.1 * time_disc) * np.sin(-2 * time_disc)
u_right = np.exp(-0.1 * time_disc) * np.sin(3 - 2 * time_disc)

# initialize
u = init_cond.copy()

start = time.time()
for i in range(N_time_steps - 1):
    u_prev = u  # save prev value for rhs bc handling

    # main step
    u = u - mu * D_upw @ u + r * D_c_2 @ u

    # boundary condition
    u[0] = u[0] + mu * u_left[i] + r * u_left[i]
    u[-1] = u[-1] + r * u_right[i]

    # corrected rhs bc with upwind disc
    u[-1] = u_prev[-1] - mu * (u_prev[-1] - u_prev[-2])

end = time.time()
print(f"Elapsed time: {end - start:.4f} seconds")

plt.plot(u, label="Numerical solution")
plt.plot(
    np.exp(-0.1 * T) * np.sin(space_disc - 2 * T),
    label="Analytic solution",
)
plt.legend()
plt.title("Lax–Wendroff without rhs bc but corrected with upwind disc")
plt.show()
