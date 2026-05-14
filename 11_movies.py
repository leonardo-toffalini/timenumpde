import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import scipy.sparse.linalg as sla
import skfem as fem
from skfem.helpers import dot, grad


@fem.BilinearForm
def stiffness(u, v, _):
    return dot(grad(u), grad(v))


@fem.BilinearForm
def mass(u, v, _):
    return u * v


def make_unit_cube_tet_mesh(n=4):
    x = np.linspace(0.0, 1.0, n + 1)
    return fem.MeshTet.init_tensor(x, x, x)


def make_unit_cube_hex_mesh(n=2):
    x = np.linspace(0.0, 1.0, n + 1)
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
    x, y, z = basis.doflocs
    r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2
    return np.exp(-200.0 * r2)


def build_system(mesh, element):
    basis = fem.Basis(mesh, element)
    M = mass.assemble(basis).tocsr()
    S = stiffness.assemble(basis).tocsr()
    return basis, M, S


def integrate_with_snapshots(
    method, M, S, p0, v0, c=320.0, dt=8e-4, t_end=0.2, frame_stride=8
):
    n_steps = int(np.ceil(t_end / dt))
    dt = t_end / n_steps
    solve_M = sla.splu(M.tocsc()).solve

    p = p0.copy()
    v = v0.copy()
    ps = [p.copy()]
    ts = [0.0]

    if method == "implicit_euler":
        A = (M + (dt * dt) * (c * c) * S).tocsc()
        solve_A = sla.splu(A).solve

    def accel(q):
        return -(c * c) * solve_M(S @ q)

    for k in range(n_steps):
        if method == "explicit_euler":
            p_new = p + dt * v
            v_new = v + dt * accel(p)

        elif method == "implicit_euler":
            rhs = M @ p + dt * (M @ v)
            p_new = solve_A(rhs)
            v_new = (p_new - p) / dt

        elif method == "rk4":
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

        step = k + 1
        if step % frame_stride == 0 or step == n_steps:
            ps.append(p.copy())
            ts.append(step * dt)

    return np.array(ts), ps


def create_animation(xyz, ts, ps, title, out_path, fps=20):
    x, y, z = xyz
    vals = np.concatenate(ps)
    vabs = np.max(np.abs(vals))
    vmin, vmax = -vabs, vabs

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=ps[0], cmap="coolwarm", s=14, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(sc, ax=ax, pad=0.12)
    cb.set_label("pressure p")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    def update(i):
        sc.set_array(ps[i])
        ax.set_title(f"{title}\nt = {ts[i]:.4f}")
        return (sc,)

    anim = FuncAnimation(fig, update, frames=len(ps), interval=1000 / fps, blit=False)

    try:
        writer = FFMpegWriter(fps=fps, bitrate=2400)
        anim.save(out_path, writer=writer)
        print(f"Saved MP4 movie: {out_path}")
    except Exception as ex:
        fallback = out_path.rsplit(".", 1)[0] + ".gif"
        print(f"FFmpeg writer unavailable ({ex}), saving GIF instead: {fallback}")
        anim.save(fallback, writer=PillowWriter(fps=fps))
        out_path = fallback
        print(f"Saved GIF movie: {out_path}")

    plt.close(fig)


def choose_case(case):
    if case == "tet_p1":
        mesh = make_unit_cube_tet_mesh(n=4)
        element = fem.ElementTetP1()
        name = "Tet P1"
    elif case == "tet_p2":
        mesh = make_unit_cube_tet_mesh(n=4)
        element = fem.ElementTetP2()
        name = "Tet P2"
    elif case == "hex_gl4":
        mesh = make_unit_cube_hex_mesh(n=2)
        element, elem_name = make_gl4_hex_element()
        name = f"Hex GL4 ({elem_name})"
    else:
        raise ValueError(f"Unknown case: {case}")
    return mesh, element, name


def main():
    parser = argparse.ArgumentParser(description="Create 3D movie for acoustic wave FEM solution.")
    parser.add_argument(
        "--case",
        choices=["tet_p1", "tet_p2", "hex_gl4"],
        default="tet_p1",
        help="Spatial discretization case.",
    )
    parser.add_argument(
        "--method",
        choices=["explicit_euler", "implicit_euler", "rk4", "stormer_verlet"],
        default="stormer_verlet",
        help="Time integrator.",
    )
    parser.add_argument("--dt", type=float, default=8e-4, help="Time step size.")
    parser.add_argument("--frame-stride", type=int, default=8, help="Store every k-th step as frame.")
    parser.add_argument("--fps", type=int, default=20, help="Movie frame rate.")
    parser.add_argument("--output", type=str, default="wave3d.mp4", help="Output movie filename.")
    args = parser.parse_args()

    mesh, element, case_name = choose_case(args.case)
    c = 320.0
    t_end = 0.2
    basis, M, S = build_system(mesh, element)
    p0 = initial_pressure(basis)
    v0 = np.zeros_like(p0)

    ts, ps = integrate_with_snapshots(
        args.method,
        M,
        S,
        p0,
        v0,
        c=c,
        dt=args.dt,
        t_end=t_end,
        frame_stride=max(1, args.frame_stride),
    )

    print(f"Case: {case_name}")
    print(f"Method: {args.method}")
    print(f"DoFs: {basis.N}, frames: {len(ps)}")
    create_animation(
        basis.doflocs,
        ts,
        ps,
        title=f"{case_name} + {args.method}",
        out_path=args.output,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
