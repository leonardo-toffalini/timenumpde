import skfem as fem
from skfem.helpers import dot, grad
import numpy as np
from skfem.visuals.matplotlib import plot
import matplotlib.pyplot as plt


@fem.BilinearForm
def a(u, v, _):
    return dot(grad(u), grad(v))


@fem.LinearForm
def l(v, w):
    x, y = w.x  # global coordinates
    f = np.sin(np.pi * x) * np.sin(np.pi * y)
    return f * v


mesh = fem.MeshTri().refined(3)  # refine thrice
print(f"{mesh=}\n")

Vh = fem.Basis(mesh, fem.ElementTriP1())
print(f"{Vh=}\n")

A = a.assemble(Vh)
b = l.assemble(Vh)

print(f"{A.shape=}\n")
print(f"{b.shape=}\n")

D = Vh.get_dofs()

print(f"{D=}\n")

x = fem.solve(*fem.condense(A, b, D=D))
print(f"{x.shape=}\n")


@fem.Functional
def error(w):
    x, y = w.x
    uh = w["uh"]
    u = np.sin(np.pi * x) * np.sin(np.pi * y) / (2.0 * np.pi**2)
    return (uh - u) ** 2


print(f"error = {str(round(error.assemble(Vh, uh=Vh.interpolate(x)), 9))}\n")

plot(Vh, x, colorbar={"orientation": "horizontal"})
plt.show()
