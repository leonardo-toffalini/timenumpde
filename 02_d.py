import numpy as np
import matplotlib.pyplot as plt
from utils import tridiag
from scipy.sparse import csr_matrix

num_pairs = 7

# space
a, b = 0, np.pi
h_start = 0.0047 * (2**num_pairs)
hs = list(h_start * np.array([1 / 2**i for i in range(num_pairs)]))

# time
delta_start = 1e-5 * (4**num_pairs)
deltas = list(delta_start * np.array([1 / 4**i for i in range(num_pairs)]))

t0 = 0
T_max = 1

errs = []

for h, delta in zip(hs, deltas):
    N = int((b - a) / h)
    space_grid = np.linspace(a + h / 2, b - h / 2, N)  # staggered points
    u0 = np.sin(space_grid)
    u_prev = u0

    N_delta = int((T_max - t0) / delta)
    time_grid = np.linspace(t0, T_max, N_delta)

    D_0_square = tridiag(1, -2, 1, N)
    Q = D_0_square.toarray()
    Q[0, 0] = -1
    Q[-1, -1] = -1
    Q = np.eye(N) + (delta / (h * h)) * Q
    Q = csr_matrix(Q)

    for n in range(N_delta):
        t = (n + 1) * delta
        bnd_vec = np.zeros(N)
        bnd_vec[0] = -np.exp(-t)
        bnd_vec[-1] = -np.exp(-t)
        bnd_vec = delta / h * bnd_vec
        u_exact = np.exp(-t) * np.sin(space_grid)
        u_next = Q @ u_prev + bnd_vec
        u_prev = u_next

    final_err = np.linalg.norm(u_next - u_exact, ord=2) * np.sqrt(np.pi / N)
    errs.append(final_err)

print(errs)
plt.plot(range(1, num_pairs + 1), errs)
plt.xlabel("pair number")
plt.ylabel("error magnitude")
plt.show()
