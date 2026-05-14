import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
import skfem as fem
from skfem.helpers import dot, grad

INCLUDE_EXPLICIT_EULER = False
INCLUDE_IMPLICIT_EULER = False
INCLUDE_RK4 = False
INCLUDE_STORMER_VERLET = True
INCLUDE_TET_P1 = True
INCLUDE_TET_P2 = False
INCLUDE_HEX_GL4 = False
c = 320.0
t_end = 0.2
probe_point = (9.0, 9.0, 9.0)
DT_TET_P1 = 1.005e-3
DT_TET_P2 = 4e-5
DT_HEX_GL4 = 4e-5
N_TET = 8
N_HEX = 4
CUBE_SIZE = 10.0


@fem.BilinearForm
def stiffness(u, v, _):
    return dot(grad(u), grad(v))


@fem.BilinearForm
def mass(u, v, _):
    return u * v


def make_unit_cube_tet_mesh(n=4, cube_size=1.0):
    x = np.linspace(0.0, cube_size, n + 1)
    return fem.MeshTet.init_tensor(x, x, x)


def make_unit_cube_hex_mesh(n=2, cube_size=1.0):
    x = np.linspace(0.0, cube_size, n + 1)
    return fem.MeshHex.init_tensor(x, x, x)


def instantiate_element(class_name, degree=None):
    cls = getattr(fem.element, class_name, None)
    if cls is None:
        return None
    if degree is None:
        try:
            return cls()
        except TypeError:
            return None
    try:
        return cls(degree)
    except TypeError:
        try:
            return cls()
        except TypeError:
            return None


def make_gl4_hex_element():
    # Different skfem versions expose different names; try common candidates.
    candidates = [
        ("ElementHexGaussLobatto", 4),
        ("ElementHexGLL", 4),
        ("ElementHexGL", 4),
        ("ElementHexP", 4),
        ("ElementHexP4", None),
    ]
    for name, deg in candidates:
        elem = instantiate_element(name, deg)
        if elem is not None:
            return elem, name
    raise RuntimeError(
        "Could not find a 4th-order hex/Gauss-Lobatto element in this skfem version."
    )


def initial_pressure(basis):
    # Smooth compact pulse centered in the cube.
    x, y, z = basis.doflocs
    center = 0.5 * CUBE_SIZE
    r2 = (x - center) ** 2 + (y - center) ** 2 + (z - center) ** 2
    return np.exp(-200.0 * r2)


def build_system(mesh, element):
    basis = fem.Basis(mesh, element)
    M = mass.assemble(basis).tocsr()
    S = stiffness.assemble(basis).tocsr()
    return basis, M, S


def make_solvers(M):
    lu = sla.splu(M.tocsc())
    return lu.solve


def energy(M, S, p, v, c):
    return 0.5 * (v @ (M @ v) + (c * c) * (p @ (S @ p)))


def integrate(method, M, S, p0, v0, c=1.0, dt=2e-3, t_end=0.2, probe_idx=None):
    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps

    solve_M = make_solvers(M)

    p = p0.copy()
    v = v0.copy()
    Es = [energy(M, S, p, v, c)]
    ts = [0.0]
    probe_vals = [float(p[probe_idx])] if probe_idx is not None else None

    if method == "implicit_euler":
        A = (M + (dt * dt) * (c * c) * S).tocsc()
        solve_A = sla.splu(A).solve

    def accel(q):
        return -(c * c) * solve_M(S @ q)

    for k in range(n_steps):
        t = (k + 1) * dt

        if method == "explicit_euler":
            p_new = p + dt * v
            v_new = v + dt * accel(p)

        elif method == "implicit_euler":
            rhs = M @ p + dt * (M @ v)
            p_new = solve_A(rhs)
            v_new = (p_new - p) / dt

        elif method == "rk4":
            # System: p' = v, v' = a(p)
            k1p = v
            k1v = accel(p)

            k2p = v + 0.5 * dt * k1v
            k2v = accel(p + 0.5 * dt * k1p)

            k3p = v + 0.5 * dt * k2v
            k3v = accel(p + 0.5 * dt * k2p)

            k4p = v + dt * k3v
            k4v = accel(p + dt * k3p)

            p_new = p + (dt / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
            v_new = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)

        elif method == "stormer_verlet":
            v_half = v + 0.5 * dt * accel(p)
            p_new = p + dt * v_half
            v_new = v_half + 0.5 * dt * accel(p_new)

        else:
            raise ValueError(f"Unknown method: {method}")

        p, v = p_new, v_new
        Es.append(energy(M, S, p, v, c))
        ts.append(t)
        if probe_idx is not None:
            probe_vals.append(float(p[probe_idx]))

    if probe_vals is not None:
        probe_vals = np.array(probe_vals)
    return np.array(ts), p, v, np.array(Es), probe_vals


def run_case(
    case_name,
    mesh,
    element,
    dt,
    c=1.0,
    t_end=0.2,
    probe_point=(0.5, 0.5, 0.5),
    include_explicit_euler=True,
    include_implicit_euler=True,
    include_rk4=True,
    include_stormer_verlet=True,
):
    basis, M, S = build_system(mesh, element)
    p0 = initial_pressure(basis)
    v0 = np.zeros_like(p0)
    dof_xyz = basis.doflocs.T
    probe_point = np.array(probe_point, dtype=float)
    probe_idx = int(np.argmin(np.linalg.norm(dof_xyz - probe_point, axis=1)))
    probe_xyz = dof_xyz[probe_idx]

    methods = []
    if include_explicit_euler:
        methods.append("explicit_euler")
    if include_implicit_euler:
        methods.append("implicit_euler")
    if include_rk4:
        methods.append("rk4")
    if include_stormer_verlet:
        methods.append("stormer_verlet")
    if not methods:
        raise ValueError("At least one time integration method must be enabled.")
    print(f"\n=== {case_name} ===")
    print(f"DoFs: {basis.N}")
    print(f"M shape: {M.shape}, S shape: {S.shape}")
    print(f"Probe target point: {tuple(probe_point)}")
    print(f"Probe actual DoF: ({probe_xyz[0]:.4f}, {probe_xyz[1]:.4f}, {probe_xyz[2]:.4f})")

    results = {}
    for method in methods:
        ts, p, v, Es, probe_vals = integrate(
            method, M, S, p0, v0, c=c, dt=dt, t_end=t_end, probe_idx=probe_idx
        )
        rel_E_drift = (Es[-1] - Es[0]) / Es[0]
        results[method] = (ts, Es, probe_vals, rel_E_drift)
        print(f"{method:15s} relative energy drift: {rel_E_drift:+.3e}")

    fig, ax = plt.subplots(figsize=(8, 4))
    for method in methods:
        ts, Es, _, _ = results[method]
        ax.plot(ts, Es / Es[0], label=method)
    ax.set_title(f"Energy ratio E(t)/E(0): {case_name}")
    ax.set_xlabel("t")
    ax.set_ylabel("E(t)/E(0)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 4))
    for method in methods:
        ts, _, probe_vals, _ = results[method]
        ax.plot(ts, probe_vals, label=method)
    ax.set_title(
        "Probe pressure vs time: "
        f"{case_name} at ({probe_xyz[0]:.3f}, {probe_xyz[1]:.3f}, {probe_xyz[2]:.3f})"
    )
    ax.set_xlabel("t")
    ax.set_ylabel("p(t)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def main():
    include_explicit_euler = INCLUDE_EXPLICIT_EULER
    include_implicit_euler = INCLUDE_IMPLICIT_EULER
    include_rk4 = INCLUDE_RK4
    include_stormer_verlet = INCLUDE_STORMER_VERLET
    include_tet_p1 = INCLUDE_TET_P1
    include_tet_p2 = INCLUDE_TET_P2
    include_hex_gl4 = INCLUDE_HEX_GL4
    dt_tet_p1 = DT_TET_P1
    dt_tet_p2 = DT_TET_P2
    dt_hex_gl4 = DT_HEX_GL4
    n_tet = N_TET
    n_hex = N_HEX
    cube_size = CUBE_SIZE

    mesh_tet = make_unit_cube_tet_mesh(n=n_tet, cube_size=cube_size)
    # 1) First-order tetrahedral elements on unit cube
    if include_tet_p1:
        run_case(
            "Tet P1",
            mesh_tet,
            fem.ElementTetP1(),
            dt=dt_tet_p1,
            c=c,
            t_end=t_end,
            probe_point=probe_point,
            include_explicit_euler=include_explicit_euler,
            include_implicit_euler=include_implicit_euler,
            include_rk4=include_rk4,
            include_stormer_verlet=include_stormer_verlet,
        )

    # 2) Second-order tetrahedral elements on unit cube
    if include_tet_p2:
        run_case(
            "Tet P2",
            mesh_tet,
            fem.ElementTetP2(),
            dt=dt_tet_p2,
            c=c,
            t_end=t_end,
            probe_point=probe_point,
            include_explicit_euler=include_explicit_euler,
            include_implicit_euler=include_implicit_euler,
            include_rk4=include_rk4,
            include_stormer_verlet=include_stormer_verlet,
        )

    # 3) Tensor-product fourth-order Gauss-Lobatto elements on unit cube
    if include_hex_gl4:
        elem_gl4, elem_name = make_gl4_hex_element()
        mesh_hex = make_unit_cube_hex_mesh(n=n_hex, cube_size=cube_size)
        run_case(
            f"Hex GL4 ({elem_name})",
            mesh_hex,
            elem_gl4,
            dt=dt_hex_gl4,
            c=c,
            t_end=t_end,
            probe_point=probe_point,
            include_explicit_euler=include_explicit_euler,
            include_implicit_euler=include_implicit_euler,
            include_rk4=include_rk4,
            include_stormer_verlet=include_stormer_verlet,
        )


if __name__ == "__main__":
    main()
