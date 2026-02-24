import numpy as np
import matplotlib.pyplot as plt
from utils import tridiag
from scipy.sparse import csr_matrix
import scipy


num_pairs = 5

# space
a, b = 0, np.pi
h_start = 0.0047 * (2**num_pairs)
h = h_start
hs = list(h_start * np.array([1 / 2**i for i in range(num_pairs)]))

# time
delta_start = 1e-5 * (4**num_pairs)
delta = delta_start
deltas = list(delta_start * np.array([1 / 4**i for i in range(num_pairs)]))

t0 = 0
T_max = 1

errs = []

for h, delta in zip(hs, deltas):
    N = int((b - a) / h) - 1
    # space_grid = np.linspace(a, b, N)
    space_grid = a + h * np.arange(1, N + 1)
    u0 = np.sin(space_grid)
    u_prev = u0

    N_delta = int((T_max - t0) / delta) - 1
    time_grid = np.linspace(t0, T_max, N_delta)

    D_0_square = tridiag(1, -2, 1, N)
    Q = D_0_square.toarray()
    Q = np.eye(N) - (delta / (h * h)) * Q
    Q = csr_matrix(Q)

    N_delta = int((T_max - t0) / delta)
    for n in range(N_delta):
        t = (n + 1) * delta
        u_exact = np.exp(-t) * np.sin(space_grid)
        u_next = scipy.sparse.linalg.spsolve(Q, u_prev)
        u_prev = u_next

    final_err = np.linalg.norm(u_next - u_exact, ord=2) * np.sqrt(np.pi / (N + 1))
    errs.append(final_err)


print(errs)
plt.plot(range(1, num_pairs + 1), errs)
plt.xlabel("pair number")
plt.ylabel("error magnitude")
plt.show()
