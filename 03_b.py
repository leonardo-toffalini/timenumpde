import numpy as np
import matplotlib.pyplot as plt
from utils import tridiag
from scipy.sparse import csr_matrix
import scipy
import time
from scipy.sparse import diags
from nodepy import rk
from nodepy.ivp import IVP

num_pairs = 1

# space
a, b = 0, np.pi
h = np.pi / 1001

# time
delta = 1 / 2 * h**2

print(f"{h=}")
print(f"{delta=}")

t0 = 0
T_max = 1


start_time = time.time()
N = int((b - a) / h) - 1
# space_grid = np.linspace(a, b, N)
space_grid = a + h * np.arange(1, N + 1)
u0 = np.sin(space_grid)
u_prev = u0

N_delta = int((T_max - t0) / delta)
time_grid = np.linspace(t0, T_max, N_delta + 1)

# seonc order
D_0_square = tridiag(1, -2, 1, N)

rk_method = rk.loadRKM("RK44")

f = lambda t, u: (1 / h**2) * D_0_square @ u  # noqa: E731
semi_discretized = IVP(f=f, u0=u0, T=T_max, name="semi_disc")
t, u = rk_method(semi_discretized, t0=t0, dt=delta)
u = np.array(u)
u_final = u[-1]
t_final = 1
u_exact = np.exp(-t_final) * np.sin(space_grid)
final_err = np.linalg.norm(u_final - u_exact, ord=2) * np.sqrt(np.pi / (N + 1))

end_time = time.time()


print("final errors ", final_err)
print("it took ", end_time - start_time, " seconds to run")
