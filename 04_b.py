import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve
import time

# Problem 3: Peaceman--Rachford scheme for
# u_t = u_xx + u_yy on (0, pi) x (0, pi) with homogeneous Neumann boundary
# conditions and initial condition u(0, x, y) = cos(x) cos(y).

T3 = 1.0
N_time_steps_3 = 100
delta_3 = T3 / N_time_steps_3

Nx3 = 40  # number of subintervals in x, including boundaries gives Nx3 + 1 points
Ny3 = 40  # number of subintervals in y
hx3 = np.pi / Nx3
hy3 = np.pi / Ny3

n_x3 = Nx3 + 1  # number of grid points in x (including boundaries)
n_y3 = Ny3 + 1  # number of grid points in y

# 1D second-derivative matrices with homogeneous Neumann BCs
D2x_N = diags([1, -2, 1], [-1, 0, 1], shape=(n_x3, n_x3), dtype=float) / (hx3 * hx3)
D2x_N = D2x_N.tolil()
D2x_N[0, 0] = -1.0 / (hx3 * hx3)
D2x_N[0, 1] = 1.0 / (hx3 * hx3)
D2x_N[-1, -2] = 1.0 / (hx3 * hx3)
D2x_N[-1, -1] = -1.0 / (hx3 * hx3)
D2x_N = D2x_N.tocsr()

D2y_N = diags([1, -2, 1], [-1, 0, 1], shape=(n_y3, n_y3), dtype=float) / (hy3 * hy3)
D2y_N = D2y_N.tolil()
D2y_N[0, 0] = -1.0 / (hy3 * hy3)
D2y_N[0, 1] = 1.0 / (hy3 * hy3)
D2y_N[-1, -2] = 1.0 / (hy3 * hy3)
D2y_N[-1, -1] = -1.0 / (hy3 * hy3)
D2y_N = D2y_N.tocsr()

Ix3 = eye(n_x3)
Iy3 = eye(n_y3)

D2x_comp_3 = kron(Iy3, D2x_N)
D2y_comp_3 = kron(D2y_N, Ix3)
full_lap_3 = D2x_comp_3 + D2y_comp_3
I3 = eye(n_x3 * n_y3)

# Initial condition u(0, x, y) = cos(x) cos(y)
x3 = np.linspace(0.0, np.pi, n_x3)
y3 = np.linspace(0.0, np.pi, n_y3)
cosx3 = np.cos(x3)
cosy3 = np.cos(y3)
init_cond_3 = np.kron(cosy3, cosx3)

# Peaceman--Rachford ADI scheme
u3 = init_cond_3.copy()
Ap3 = I3 + delta_3 * D2x_comp_3 / 2.0
Am3 = I3 - delta_3 * D2x_comp_3 / 2.0
Bp3 = I3 + delta_3 * D2y_comp_3 / 2.0
Bm3 = I3 - delta_3 * D2y_comp_3 / 2.0

start = time.time()
for _ in range(N_time_steps_3):
    u_half_3 = spsolve(Am3, Bp3 @ u3)
    u3 = spsolve(Bm3, Ap3 @ u_half_3)
print("Problem 3 - Peaceman--Rachford with Neumann BCs")
print(time.time() - start)

# Exact solution at time T3: u(T3, x, y) = exp(-2 T3) cos(x) cos(y)
u_real_3 = np.exp(-2.0 * T3) * init_cond_3
factor_3 = np.sqrt(hx3 * hy3)
error_PR_3 = factor_3 * np.linalg.norm(u3 - u_real_3)

print("L2 error (PR, Neumann) =", error_PR_3)

