import time
import numpy as np
import matplotlib.pyplot as plt

from ..utils.LBA import LBA


def compute_divergence_norm(ux, uy, obstacle):
    """L2 norm of div(u) over fluid cells (lattice spacing = 1)."""
    dux_dx = 0.5 * (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0))
    duy_dy = 0.5 * (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
    div = dux_dx + duy_dy

    fluid = ~obstacle
    if np.count_nonzero(fluid) == 0:
        return np.nan
    return np.sqrt(np.mean(div[fluid] ** 2))


def estimate_strouhal_from_probe(signal, sample_every, D, U):
    """
    Estimate St from FFT of a probe signal.
    dt (lattice units) = 1, so sample_dt = sample_every.
    """
    if len(signal) < 128:
        return np.nan

    y = np.asarray(signal, dtype=float)
    y = y - np.mean(y)

    freqs = np.fft.rfftfreq(len(y), d=float(sample_every))
    spec = np.abs(np.fft.rfft(y))

    # Remove zero frequency
    valid = freqs > 0
    if not np.any(valid):
        return np.nan

    f_peak = freqs[valid][np.argmax(spec[valid])]
    St = f_peak * D / U
    return St


def run_validation_sweep():
    target_res = [10, 20, 50, 100, 150, 200]

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
        for step in range(1, n_steps + 1):
            solver.step()

            if step >= transient_steps and step % sample_every == 0:
                probe_signal.append(solver.uy[i_probe, j_probe])

            if step % 5000 == 0:
                print(f"  step {step}/{n_steps}")

        div_norm = compute_divergence_norm(solver.ux, solver.uy, solver.obstacle)
        St = estimate_strouhal_from_probe(
            probe_signal, sample_every=sample_every, D=solver.D, U=solver.U_inlet
        )

        re_list.append(Re)
        div_list.append(div_norm)
        st_list.append(St)

        print(
            f"    Re: {Re}, L2-Div: {div_norm:.2e}, "
            f"St: {St:.3f}, Time: {time.time() - t0:.2f}s"
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1) Divergence plot
    axes[0].plot(re_list, div_list, marker="o", linestyle="-", color="tab:blue")
    axes[0].set_yscale("log")
    axes[0].set_title("Divergence vs. Reynolds Number")
    axes[0].set_xlabel("Reynolds Number (Re)")
    axes[0].set_ylabel("L2-Norm of Divergence")

    # 2) Accuracy plot (Strouhal)
    axes[1].scatter(
        re_list,
        st_list,
        facecolors="none",
        edgecolors="tab:red",
        linewidths=2,
        s=80,
        label="Simulated (LBA)",
        zorder=10,
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
    axes[1].legend(fancybox=True, shadow=True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_validation_sweep()