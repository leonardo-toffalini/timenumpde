import numpy as np
import scipy.sparse as sp
import scipy
import matplotlib.pyplot as plt
import time

# solution of the 1D advection problem
# on the interval (0,3) over time (0,2)
# d_t u(t,x) +  2 d_x u(t,x) = 0
# u(0,x) = sin x;
# u(t,0) = sin -2t;
# analytic solution: sin(x-2t)

T = 2
N_time_steps = 250
delta = T / N_time_steps
Nx = 50
hx = 3 / (Nx + 1)
mu = 2 * delta / hx
print(f"{mu=}")

# basic matrices
D_upw = sp.diags([-1, 1], offsets=[-1, 0], shape=(Nx, Nx)) / hx
D_c = sp.diags([-1, 1], offsets=[-1, 1], shape=(Nx, Nx))  # / hx
I = sp.eye(Nx)

init_cond = np.sin(np.linspace(hx, 3 - hx, Nx))
plt.plot(init_cond)
plt.show()

# time step: explicit Euler
# values on the left-hand side
u_left = np.sin(-2 * np.linspace(0, T - delta, N_time_steps))
u_right = np.sin(3 - 2 * np.linspace(0, T - delta, N_time_steps))

# main step
# u_next = u  - mu/4 D_0 (u + u_next)

# initialize
u = init_cond.copy()

start = time.time()
for i in range(N_time_steps - 1):
    # main step
    u = sp.linalg.spsolve(I + mu / 4 * D_c, (I - mu / 4 * D_c) @ u)

    # boundary condition
    u[0] = u[0] + mu / 4 * (u_left[i + 1] + u_left[i])
    u[-1] = u[-1] - mu / 4 * (u_right[i + 1] + u_right[i])

end = time.time()
print(f"Elapsed time: {end - start:.4f} seconds")

plt.plot(u, label="Numerical solution")
plt.plot(np.sin(np.linspace(hx, 3 - hx, Nx) - 2 * 2), label="Analytic solution")
plt.legend()
plt.title("Crank-Nicolson + central")
plt.show()
