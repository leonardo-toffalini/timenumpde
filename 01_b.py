import numpy as np
import matplotlib.pyplot as plt
from utils import tridiag
import scipy
import time


# space
a, b = 0, np.pi
h = 0.0047
N = int((b - a) / h) - 1

# time
delta = 1e-5
t0 = 0
T_max = 1
N_delta = int((T_max - t0) / delta) - 1
time_grid = np.linspace(t0, T_max, N_delta)

u = []
space_grid = np.linspace(a, b, N)
u.append(np.sin(space_grid))

D_0_square = tridiag(1, -2, 1, N)

errs = []

start_time = time.time()
for t in time_grid:
    u_exact = np.exp(-t) * np.sin(space_grid) + t**2
    u_prev = u[-1]
    boundary_handling_vector = np.zeros_like(u_prev)
    boundary_handling_vector[0] = delta / h**2 * t
    boundary_handling_vector[-1] = delta / h**2 * t
    u_next = (
        u_prev
        + delta / h**2 * (D_0_square @ u_prev)
        + boundary_handling_vector
        + delta * 2 * t * np.ones_like(u_prev)
    )
    u.append(u_next)

    err = np.linalg.norm(u_next - u_exact, ord=2) * np.sqrt(np.pi / (N + 1))
    errs.append(err)

    plotting = False
    if plotting:
        plt.plot(space_grid, u_next, label="num")
        plt.plot(space_grid, u_exact, label="exact")
        plt.title(f"t={t.item()}")
        plt.legend()
        plt.show()

print("time it took with Explicit Euler: ", time.time() - start_time)

u_final_ee = u[-1]
u_final_exact = np.exp(-t) * np.sin(space_grid) + t**2

err_ee = np.linalg.norm(u_final_ee - u_final_exact, ord=2) * np.sqrt(np.pi / (N + 1))

print("error of EE solution at the last time: ", err_ee)

plotters = False
if plotters:
    plt.plot(time_grid, errs)
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"errors in time | {h=}, {delta=}")
    plt.show()
