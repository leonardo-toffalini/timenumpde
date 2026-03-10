import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve, expm_multiply
import time

T = 1
N_time_steps = 100
N_time_steps_ee = 20000
N_time_steps_ie = 1000
delta = 1 / N_time_steps
delta_ie = 1 / N_time_steps_ie
delta_ee = 1 / N_time_steps_ee
Nx = 40
Ny = 50
hx = np.pi / (Nx + 1)
hy = 2 * np.pi / (Ny + 1)

D2x = diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx)) / (hx * hx)
D2y = diags([1, -2, 1], [-1, 0, 1], shape=(Ny, Ny)) / (hy * hy)

D2x_comp = kron(eye(Ny), D2x)
D2y_comp = kron(D2y, eye(Nx))

# spy equivalent in python is matplotlib's spy, but here we skip plotting

full_lap = D2x_comp + D2y_comp
I = eye(Nx * Ny)

sin_base_x = np.sin(np.linspace(hx, np.pi - hx, Nx))
sin_base_y = np.sin(np.linspace(hy, 2 * np.pi - hy, Ny))
init_cond = np.kron(sin_base_y, sin_base_x)
_ = init_cond[299]  # MATLAB 3000th element is index 2999 in Python

# Crank--Nicolson scheme
u = init_cond.copy()
A = I - delta * full_lap / 2
B = I + delta * full_lap / 2

start = time.time()
for _ in range(N_time_steps):
    u = spsolve(A, B @ u)
print("Crank--Nicolson")
print(time.time() - start)
u_CN = u

# explicit Euler
u = init_cond.copy()
start = time.time()
for _ in range(N_time_steps_ee):
    u = u + delta_ee * full_lap @ u
print("explicit Euler")
print(time.time() - start)
u_ee = u

# implicit Euler
u = init_cond.copy()
MM = I - delta_ie * full_lap
start = time.time()
for _ in range(N_time_steps_ie):
    u = spsolve(MM, u)
print("implicit Euler")
print(time.time() - start)
u_ie = u

# Djakonov scheme
u = init_cond.copy()
Am = I - delta * D2x_comp / 2
Ap = I + delta * D2x_comp / 2
Bp = I + delta * D2y_comp / 2
Bm = I - delta * D2y_comp / 2
BB = Bp @ Ap

start = time.time()
for _ in range(N_time_steps):
    u_half = spsolve(Am, BB @ u)
    u = spsolve(Bm, u_half)
print("Djakonov")
print(time.time() - start)
u_D = u

# Peaceman--Rachford scheme
u = init_cond.copy()
Ap = I + delta * D2x_comp / 2
Am = I - delta * D2x_comp / 2
Bp = I + delta * D2y_comp / 2
Bm = I - delta * D2y_comp / 2

start = time.time()
for _ in range(N_time_steps):
    u_half = spsolve(Am, Bp @ u)
    u = spsolve(Bm, Ap @ u_half)
print("Peaceman--Rachford")
print(time.time() - start)
u_PR = u

# semi-analytic solution
start = time.time()
exp_sol = expm_multiply(full_lap, init_cond, start=0, stop=1, num=2)[-1]
print("analytic - exp")
print(time.time() - start)

# errors for the different procedures
u_real = np.exp(-2) * init_cond
factor = np.sqrt(hx * hy)
error_CN = factor * np.linalg.norm(u_CN - u_real)
error_EE = factor * np.linalg.norm(u_ee - u_real)
error_IE = factor * np.linalg.norm(u_ie - u_real)
error_D = factor * np.linalg.norm(u_D - u_real)
error_PR = factor * np.linalg.norm(u_PR - u_real)
error_anal = factor * np.linalg.norm(exp_sol - u_real)

error_CN, error_EE, error_IE, error_D, error_PR, error_anal
