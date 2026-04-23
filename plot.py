import matplotlib.pyplot as plt
import numpy as np

a = 2
h = 0.1
delta = 0.045
mu = 2 * (delta / h)
print(f"{mu=}")
xi = np.linspace(0, 1, 100)

ome_real = xi * a
ome_approx = np.arcsin(mu * np.sin(xi * h)) / delta

plt.plot(xi, ome_real, label="exact")
plt.plot(xi, ome_approx, label="(a1)")
plt.legend()
plt.show()
