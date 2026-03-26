import time
import numpy as np
import matplotlib.pyplot as plt

from ..utils.LBA import LBA


def compute_divergence_norm(ux, uy, obstacle):
    """L2 norm of div(u) over fluid cells (lattice spacing = 1)."""
    dux_dx = 0.5 * (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0))
    duy_dy = 0.5 * (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
    div = dux_dx + duy_dy

    finite = np.isfinite(ux) & np.isfinite(uy)
    fluid = (~obstacle) & finite
    if np.count_nonzero(fluid) == 0:
        return np.nan
    return np.sqrt(np.mean(div[fluid] ** 2))


def estimate_strouhal_from_probe(signal, sample_every, D, U, st_min=0.05, st_max=0.5):
    """
    Estimate St from FFT of a probe signal.
    dt (lattice units) = 1, so sample_dt = sample_every.
    """
    y = np.asarray(signal, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 128:
        return np.nan

    y = y - np.mean(y)
    y = y * np.hanning(len(y))

    freqs = np.fft.rfftfreq(len(y), d=float(sample_every))
    spec = np.abs(np.fft.rfft(y))
    st = freqs * D / U

    # Restrict to physically plausible shedding band.
    valid = (freqs > 0) & (st >= st_min) & (st <= st_max) & np.isfinite(spec)
    if not np.any(valid):
        return np.nan

    band_spec = spec[valid]
    peak_idx = np.argmax(band_spec)
    peak = band_spec[peak_idx]
    noise_floor = np.median(band_spec)
    if noise_floor <= 0.0:
        noise_floor = 1e-16

    # Avoid reporting random FFT noise as a shedding frequency.
    if peak / noise_floor < 4.0:
        return np.nan

    f_peak = freqs[valid][peak_idx]
    St = f_peak * D / U
    return St


def run_validation_sweep():
    target_res = [10, 15, 20, 25, 30]

    # Keep geometry/velocity fixed across Re sweep
    Nx, Ny = 300, 120
    U_inlet = 0.12

    re_list, div_list, st_list = [], [], []

    # Runtime controls
    n_steps = 30000
    sample_every = 10
    transient_steps = 8000

    for Re in target_res:
        print(f"Running LBA simulation for Re = {Re}")
        t0 = time.time()

        solver = LBA(Nx=Nx, Ny=Ny, U_inlet=U_inlet, Re=Re)

        # Probe point downstream of cylinder centerline
        i_probe = min(solver.cx_cyl + 6 * solver.r_cyl, solver.Nx - 2)
        j_probe = solver.cy_cyl

        probe_signal = []
        unstable = False
        for step in range(1, n_steps + 1):
            solver.step()

            if not np.isfinite(solver.rho).all() or not np.isfinite(solver.ux).all() or not np.isfinite(solver.uy).all():
                unstable = True
                print(f"  unstable state detected at step {step}; skipping Re={Re}")
                break

            if step >= transient_steps and step % sample_every == 0:
                probe_signal.append(solver.uy[i_probe, j_probe])

            if step % 5000 == 0:
                print(f"  step {step}/{n_steps}")

        if unstable:
            div_norm = np.nan
            St = np.nan
        else:
            div_norm = compute_divergence_norm(solver.ux, solver.uy, solver.obstacle)
            St = estimate_strouhal_from_probe(
                probe_signal, sample_every=sample_every, D=solver.D, U=solver.U_inlet
            )

        re_list.append(Re)
        div_list.append(div_norm)
        st_list.append(St)

        rho_min = np.nanmin(solver.rho)
        rho_max = np.nanmax(solver.rho)
        print(
            f"    Re: {Re}, L2-Div: {div_norm:.2e}, St: {St:.3f}, "
            f"rho[min,max]=[{rho_min:.3f},{rho_max:.3f}], Time: {time.time() - t0:.2f}s"
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Divergence plot
    re_arr = np.asarray(re_list, dtype=float)
    div_arr = np.asarray(div_list, dtype=float)
    st_arr = np.asarray(st_list, dtype=float)
    div_valid = np.isfinite(div_arr)
    st_valid = np.isfinite(st_arr)

    axes[0].plot(re_arr[div_valid], div_arr[div_valid], marker="o", linestyle="-", color="tab:blue")
    axes[0].scatter(re_arr[~div_valid], np.full(np.count_nonzero(~div_valid), 1e-12), marker="x", color="tab:gray", label="invalid")
    axes[0].set_yscale("log")
    axes[0].set_title("Divergence vs. Reynolds Number")
    axes[0].set_xlabel("Reynolds Number (Re)")
    axes[0].set_ylabel("L2-Norm of Divergence")
    if np.any(~div_valid):
        axes[0].legend()

    # Accuracy plot
    axes[1].scatter(
        re_arr[st_valid],
        st_arr[st_valid],
        facecolors="none",
        edgecolors="tab:red",
        linewidths=2,
        s=80,
        label="Simulated (LBA)",
        zorder=10,
    )
    if np.any(~st_valid):
        axes[1].scatter(
            re_arr[~st_valid],
            np.zeros(np.count_nonzero(~st_valid)),
            marker="x",
            color="tab:gray",
            s=60,
            label="Invalid / no clear shedding",
            zorder=8,
        )
    axes[1].scatter(
        [100],
        [0.300],
        marker="*",
        s=200,
        color="gold",
        edgecolor="black",
        label="DFG 2D-2 Benchmark",
        zorder=5,
    )
    axes[1].set_title("Physical Accuracy: Strouhal Number Validation")
    axes[1].set_xlabel("Reynolds Number (Re)")
    axes[1].set_ylabel("Strouhal Number (St)")
    axes[1].set_ylim(-0.02, 0.6)
    axes[1].legend(fancybox=True, shadow=True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_validation_sweep()