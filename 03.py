import numpy as np
import matplotlib.pyplot as plt
from utils import tridiag
from scipy.sparse import (
    coo_matrix,
    csr_matrix,
    csc_matrix,
    lil_matrix,
    bsr_matrix,
    dia_matrix,
    dok_matrix,
)
import scipy
import time

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


N = int((b - a) / h) - 1
# space_grid = np.linspace(a, b, N)
space_grid = a + h * np.arange(1, N + 1)
u0 = np.sin(space_grid)
u_prev = u0

N_delta = int((T_max - t0) / delta) - 1
time_grid = np.linspace(t0, T_max, N_delta)

# seonc order
D_0_square = tridiag(1, -2, 1, N)
Q = D_0_square.toarray()

Q_EE = np.eye(N) + (delta / (h * h)) * Q
Q_EE = csr_matrix(Q_EE)
Q_IE = np.eye(N) - (delta / (h * h)) * Q
Q_IE = csr_matrix(Q_IE)

ie = False
ee = True

N_delta = int((T_max - t0) / delta)
num_runs = 10
times = []
for _ in range(num_runs):
    start_time = time.time()
    for n in range(N_delta):
        t = (n + 1) * delta
        u_exact = np.exp(-t) * np.sin(space_grid)
        if ie:
            u_next = scipy.sparse.linalg.spsolve(Q_IE, u_prev)
        if ee:
            u_next = Q_EE @ u_prev
        u_prev = u_next
    end_time = time.time()
    times.append(end_time - start_time)

avg_time = np.mean(times)
std_time = np.std(times)

final_err = np.linalg.norm(u_next - u_exact, ord=2) * np.sqrt(np.pi / (N + 1))


print("EE" if ee else "IE")
print("final errors ", final_err)
print(f"it took {avg_time} +- {std_time} seconds to run over {num_runs} runs")
