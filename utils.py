import numpy as np
from scipy.sparse import diags


def tridiag(a, b, c, n):
    k = [a * np.ones(n - 1), b * np.ones(n), c * np.ones(n - 1)]
    offset = [-1, 0, 1]
    return diags(k, offset)
