import time
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt


def solve_wave(hx, delta, T=1.0, a=0.0, b=np.pi / 2):
    cfl = delta / hx
    if cfl > 1:
        print(f"\n=============== CFL condition is not satisfied: cfl={cfl:.6e} ===============\n")

    interval_len = b - a

    n_space_intervals = int(round(interval_len / hx))
    if not np.isclose(interval_len / hx, n_space_intervals):
        raise ValueError("hx must divide (b-a) so the grid lands exactly on both boundaries.")

    n_time_steps = int(round(T / delta))
    if not np.isclose(T / delta, n_time_steps):
        raise ValueError("delta must divide T so the final time is reached exactly.")

    hx = interval_len / n_space_intervals
    delta = T / n_time_steps
    r_2 = delta * delta / (hx * hx)

    # unknowns are x_0, ..., x_(m-1) where x_m = b is Dirichlet.
    m = n_space_intervals
    space_grid = np.linspace(a, b - hx, m)

    # second derivative stencil without 1/hx^2
    D_c_2 = sp.diags([1, -2, 1], offsets=[-1, 0, 1], shape=(m, m), format="lil", dtype=None)
    D_c_2[0, 1] = 2
    D_c_2 = D_c_2.tocsr()

    force_vec = 2 * np.ones(m)
    bc_vec = np.zeros(m)
    bc_vec[-1] = 1

    u_0 = np.cos(space_grid)
    u_prev = u_0.copy()

    t0 = 0.0
    g = t0 * t0
    lap_u0 = (1 / (hx * hx)) * (D_c_2 @ u_prev + bc_vec * g)
    u = u_prev + 0.5 * delta * delta * (lap_u0 + force_vec)

    start = time.time()
    for i in range(1, n_time_steps):
        tn = i * delta
        g = tn * tn
        lap_u = (1 / (hx * hx)) * (D_c_2 @ u + bc_vec * g)
        u_next = 2 * u - u_prev + delta * delta * (lap_u + force_vec)
        u, u_prev = u_next, u
    elapsed = time.time() - start

    u_num = np.append(u, T * T)
    space_grid_full = np.append(space_grid, b)
    u_exact = T * T + np.cos(T) * np.cos(space_grid_full)
    err_l2 = np.linalg.norm(u_num - u_exact, ord=2) * np.sqrt(hx)

    return {
        "hx": hx,
        "delta": delta,
        "r_2": r_2,
        "n_time_steps": n_time_steps,
        "n_space_intervals": n_space_intervals,
        "elapsed": elapsed,
        "err_l2": err_l2,
        "space_grid": space_grid_full,
        "u_num": u_num,
        "u_exact": u_exact,
    }


def convergence_table(hx0, delta0, levels=4):
    prev_err = None
    errors = []
    hxs = []
    cfl = delta0 / hx0
    print(f"cfl={cfl:.6e}")
    for l in range(levels):
        hx = hx0 / (2**l)
        # keep r_2 = (delta/hx)^2 constant for conditional stability
        delta = cfl * hx
        out = solve_wave(hx, delta)
        err = out["err_l2"]
        rate = np.nan if prev_err is None else np.log(prev_err / err) / np.log(2)
        print(
            f"level={l}, hx={out['hx']:.6e}, delta={out['delta']:.6e}, "
            f"r_2={out['r_2']:.6e}, err_l2={err:.6e}, rate={rate:.4f}"
        )
        prev_err = err
        errors.append(err)
        hxs.append(out["hx"])

    # log-log fit: log(err) = p * log(hx) + c
    log_hx = np.log(hxs)
    log_err = np.log(errors)
    p, c = np.polyfit(log_hx, log_err, 1)
    fit_errors = np.exp(c) * np.power(hxs, p)
    print(f"fitted slope p={p:.4f}")

    plt.loglog(hxs, errors, "o-", label="errors")
    plt.loglog(hxs, fit_errors, "--", label=f"fit slope={p:.2f}")
    plt.xlabel("hx")
    plt.ylabel("L2 error")
    plt.title("Convergence with fitted line")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.show()


if __name__ == "__main__":
    convergence_table(hx0=(np.pi / 2) / 51, delta0=1 / 33, levels=6)
