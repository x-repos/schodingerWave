"""Warped-variable (p-space) diagnostics for the sponge Schrodingerisation.

(a) snapshots of ||w(t, p)|| over p (acoustic 8x8 sponge) against the
    exact e^{-p} ||z(t)|| on p > 0 -- the analogue of Fig. 5 of Jin, Liu,
    Yu, PRA 108, 032603;
(b) one-qubit post-selection: measuring the most significant p qubit
    (p >= 0 <=> MSB = 1) leaves the system register in z(T)/||z(T)||
    up to discretisation; success probability vs the closed form
    P = ||z(T)||^2 / ||z0||^2 * sum_{p_j>=0} e^{-2p_j} / sum_j g(p_j)^2;
(c) wrap-around: slice-recovery error vs T at fixed p_max, against the
    domain-of-dependence time T_dep = (p_max - p*) / max(Sigma) and the
    contamination estimate e^{2 p* + max(Sigma) T - 2 p_max}.

Run: python paper/experiments/pspace_analysis.py   (8 workers)
Out: paper/experiments/pspace_analysis_results.json, figures/fig_pspace.png
"""
import json
import os
import sys
import time
from multiprocessing import get_context

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from scipy.sparse.linalg import expm_multiply  # noqa: E402

OUT = os.path.join(HERE, "pspace_analysis_results.json")


def acoustic(nq=3, npml=2, sigma_max=1.0):
    from lib import WaveEquationSponge2d
    op = WaveEquationSponge2d(nq, nq, npml=npml, sigma_max=sigma_max)
    N = 2**nq
    b = np.exp(-0.5 * ((np.arange(N) - N // 2) / 1.0) ** 2)
    return op, op.initial_state(np.outer(b, b))


def elastic(nq=3, npml=2, sigma_max=1.0):
    from lib import ElasticSponge2d
    op = ElasticSponge2d(nq, nq, npml=npml, cp=1.0, cs=1 / np.sqrt(3),
                         sigma_max=sigma_max)
    N = 2**nq
    x = np.arange(N) - N // 2
    g = np.exp(-0.5 * (x[:, None]**2 + x[None, :]**2))
    return op, op.initial_state(g, 0.5 * g)


def build(kind, **kw):
    return acoustic(**kw) if kind == "acoustic" else elastic(**kw)


def snapshot_job(args):
    """(a) + (b) at one (kind, T)."""
    from lib import Schrodingerisation
    kind, T, n_p, p_max = args
    op, z0 = build(kind)
    sch = Schrodingerisation(op.hamiltonian_parts(), num_qubits_p=n_p, p_max=p_max)
    W = sch.evolve(z0, T) if T > 0 else sch.initial_w(z0)
    zT = expm_multiply(op.generator().tocsc() * T, z0) if T > 0 else z0
    norms = np.linalg.norm(W, axis=1)
    pos = sch.p >= 0
    P_num = float((norms[pos]**2).sum() / (np.linalg.norm(sch.initial_w(z0))**2))
    P_pred = float(np.linalg.norm(zT)**2 / np.linalg.norm(z0)**2
                   * (np.exp(-2 * sch.p[pos])).sum() / (sch.g**2).sum())
    # reduced system state after MSB = 1, fidelity with z(T)/||z(T)||
    zhat = zT / np.linalg.norm(zT)
    Wp = W[pos]
    rho_overlap = float(np.sum(np.abs(Wp.conj() @ zhat)**2) / np.sum(norms[pos]**2))
    # window p in [0, p_max/2): two top qubits = '10'
    win = pos & (sch.p < p_max / 2)
    Ww = W[win]
    fid_win = float(np.sum(np.abs(Ww.conj() @ zhat)**2) / np.sum(np.linalg.norm(Ww, axis=1)**2))
    z_slice, ps = sch.recover(W)
    err_slice = float(np.linalg.norm(z_slice - zT) / np.linalg.norm(zT))
    return dict(kind=kind, T=T, n_p=n_p, p_max=p_max, p=sch.p.tolist(),
                w_norm=norms.tolist(), zT_norm=float(np.linalg.norm(zT)),
                z0_norm=float(np.linalg.norm(z0)),
                P_msb_numeric=P_num, P_msb_predicted=P_pred,
                infidelity_msb=1 - rho_overlap, infidelity_window=1 - fid_win,
                slice_rel_err=err_slice, p_star=float(ps))


def wrap_job(args):
    """(c) slice error vs T at fixed p_max."""
    from lib import Schrodingerisation
    kind, sigma_max, T, n_p, p_max = args
    op, z0 = build(kind, sigma_max=sigma_max)
    sch = Schrodingerisation(op.hamiltonian_parts(), num_qubits_p=n_p, p_max=p_max)
    W = sch.evolve(z0, T)
    zT = expm_multiply(op.generator().tocsc() * T, z0)
    z, ps = sch.recover(W)
    smax = float(op.sigma_blocks.max())
    return dict(kind=kind, sigma_max=sigma_max, sigma_tot_max=smax, T=T,
                p_max=p_max, n_p=n_p, p_star=float(ps),
                abs_err=float(np.linalg.norm(z - zT) / np.linalg.norm(z0)),
                rel_err=float(np.linalg.norm(z - zT) / np.linalg.norm(zT)),
                zT_norm=float(np.linalg.norm(zT)),
                T_dep=float((p_max - ps) / smax),
                estimate=float(np.exp(2 * ps + smax * T - 2 * p_max)))


def figure(res):
    sys.path.insert(0, HERE)
    import figstyle as fs
    fs.setup()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(fs.TEXTWIDTH, 1.95))
    snaps = [r for r in res["snapshots"] if r["kind"] == "acoustic"]
    cols = plt.cm.viridis(np.linspace(0, 0.85, len(snaps)))
    for r, c in zip(snaps, cols):
        p = np.array(r["p"])
        ax[0].plot(p, r["w_norm"], color=c, lw=0.9, label=f"${r['T']:g}$")
        pp = p[p >= 0]
        ax[0].plot(pp, np.exp(-pp) * r["zT_norm"], "o", ms=2.0, color=c,
                   markevery=14, mfc="white", mew=0.6)
    ax[0].set_xlim(-6, 8)
    ax[0].set_ylim(0, None)
    ax[0].set_xlabel(r"$p$")
    ax[0].set_ylabel(r"$\|\mathbf{w}(t,p)\|$")
    ax[0].legend(title=r"$t$", loc="upper right", fontsize=6.5, title_fontsize=7,
                 handlelength=1.0, labelspacing=0.2, borderaxespad=0.3)
    for kind, mk, c in (("acoustic", "o", fs.BLUE), ("elastic", "s", fs.VERMILION)):
        rr = [r for r in res["snapshots"] if r["kind"] == kind and r["T"] > 0]
        T = [r["T"] for r in rr]
        ax[1].semilogy(T, [r["P_msb_predicted"] for r in rr], "-", lw=0.9, color=c)
        ax[1].semilogy(T, [r["P_msb_numeric"] for r in rr], mk, mfc="white", mew=0.8,
                       color=c, label=kind)
    ax[1].set_xlabel(r"$T$")
    ax[1].set_ylabel(r"success probability")
    ax[1].legend(loc="lower left", fontsize=7, handlelength=1.0)
    for sm, c, mk in ((1.0, fs.BLUE, "o"), (2.0, fs.VERMILION, "s")):
        rr = sorted([r for r in res["wrap"] if r["sigma_max"] == sm],
                    key=lambda r: r["T"])
        T = np.array([r["T"] for r in rr])
        ax[2].semilogy(T, [r["abs_err"] for r in rr], "-", marker=mk, color=c, ms=2.4,
                       mfc="white", mew=0.6, lw=0.9, label=rf"$\sigma_{{\max}}={sm:g}$")
        ax[2].axvline(rr[0]["T_dep"], color=c, ls=":", lw=0.8)
    ax[2].set_ylim(1e-8, 1e2)
    ax[2].set_xlabel(r"$T$")
    ax[2].set_ylabel(r"recovery error")
    ax[2].legend(loc="lower right", fontsize=7, handlelength=1.4)
    fs.panel_label(ax[0], "(a)", x=0.04, y=0.96)
    for k in (1, 2):
        ax[k].text(0.96, 0.96, f"({'abc'[k]})", transform=ax[k].transAxes, ha="right", va="top")
    ax[1].set_ylim(1e-3, 6e-1)
    fig.tight_layout(w_pad=0.4)
    fs.save(fig, "fig_pspace")

if __name__ == "__main__":
    if "--fig-only" in sys.argv:
        figure(json.load(open(OUT)))
        sys.exit()
    t0 = time.time()
    snap_jobs = [(k, T, 10, 16.0) for k in ("acoustic", "elastic")
                 for T in (0.0, 2.5, 5.0, 7.5, 10.0)]
    wrap_jobs = [("acoustic", sm, T, 9, 8.0) for sm in (1.0, 2.0)
                 for T in np.arange(1.0, 25.0, 1.0)]
    with get_context("fork").Pool(8) as pool:
        snaps = pool.map(snapshot_job, snap_jobs, chunksize=1)
        wraps = pool.map(wrap_job, wrap_jobs, chunksize=1)
    res = dict(meta=dict(script="paper/experiments/pspace_analysis.py",
                         wall_s=round(time.time() - t0, 1)),
               snapshots=snaps, wrap=wraps)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    for r in snaps:
        print(r["kind"], r["T"], "P %.4e pred %.4e  infid_msb %.2e infid_win %.2e slice %.2e"
              % (r["P_msb_numeric"], r["P_msb_predicted"], r["infidelity_msb"],
                 r["infidelity_window"], r["slice_rel_err"]))
    for r in wraps:
        print("wrap", r["sigma_max"], r["T"], "%.2e %.2e est %.2e Tdep %.1f"
              % (r["abs_err"], r["rel_err"], r["estimate"], r["T_dep"]))
    figure(res)
