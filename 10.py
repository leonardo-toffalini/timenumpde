import skfem as fem
from skfem.helpers import dot, grad
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt


@fem.BilinearForm
def stiffness(u, v, _):
    return dot(grad(u), grad(v))


@fem.BilinearForm
def mass(u, v, _):
    return u * v


mesh = fem.MeshTri().refined(3)  # refine thrice
print(f"{mesh=}\n")

Vh = fem.Basis(mesh, fem.ElementTriP1())
print(f"{Vh=}\n")

K = stiffness.assemble(Vh)
M = mass.assemble(Vh)

print(f"{K.shape=}")
print(f"{M.shape=}\n")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].spy(K, markersize=1)
axes[0].set_title("spy(K)")
axes[1].spy(M, markersize=1)
axes[1].set_title("spy(M)")
plt.tight_layout()
plt.show()

lam_max = float(sla.eigsh(K, M=M, k=1, which="LM", return_eigenvectors=False)[0])
print(f"\n{lam_max=}")
print(f"2/lam_max(M^-1 S) = {2.0 / lam_max}")
