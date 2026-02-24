import numpy as np
import matplotlib.pyplot as plt
from utils import tridiag
from scipy.sparse import csr_matrix

# TODO: modify this to use a staggered grid in space

num_pairs = 4

# space
a, b = 0, np.pi
h_start = 0.0047 * 4
h = h_start
hs = list(h_start * np.array([1 / 2**i for i in range(num_pairs)]))

# time
delta_start = 1e-5 * 16
delta = delta_start
deltas = list(delta_start * np.array([1 / 4**i for i in range(num_pairs)]))

t0 = 0
T_max = 1

errs = []

for h, delta in zip(hs, deltas):
    N = int((b - a) / h) - 1
    space_grid = np.linspace(a + h, b - h, N)
    u_prev = np.cos(space_grid)

    N_delta = int((T_max - t0) / delta) - 1
    time_grid = np.linspace(t0, T_max, N_delta)

    D_0_square = tridiag(1, -2, 1, N)
    Q = D_0_square.toarray()
    Q[0, 0] = -1
    Q[-1, -1] = -1
    Q = np.eye(N) + (delta / (h * h)) * Q
    Q = csr_matrix(Q)

    N_delta = int((T_max - t0) / delta)
    for n in range(N_delta):
        t = (n + 1) * delta
        u_exact = np.exp(-t) * np.cos(space_grid)
        u_next = Q @ u_prev
        u_prev = u_next

    final_err = np.linalg.norm(u_next - u_exact, ord=2) * np.sqrt(np.pi / (N + 1))
    errs.append(final_err)


print(errs)
plt.plot(range(1, num_pairs + 1), errs)
plt.xlabel("pair number")
plt.ylabel("error magnitude")
plt.show()
