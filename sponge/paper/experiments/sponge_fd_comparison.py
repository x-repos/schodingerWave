"""sponge_fd_comparison.py -- independent classical check of the sponge PDE (paper Eq. 14) with a
standard staggered-grid leapfrog finite-difference solver (Virieux-type),
written from scratch -- no project code.

    dv/dt = c div r - sigma v ,   dr/dt = c grad v - s_r * sigma r

mode 'both'  : s_r = 1   (our sponge, Eq. 14)
mode 'vonly' : s_r = 0, v damped with 2*sigma  (classical mask u_tt + eta u_t = c^2 lap u, eta = 2 sigma)
mode 'none'  : sigma = 0 (rigid walls)

Reference: same pulse on a 3x larger grid, no absorber; compared on the
interior window (layers excluded) until the big grid's walls could matter.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
c, h, dt = 1.0, 1.0, 0.4               # CFL c dt/h = 0.4 < 1/sqrt(2)
NI, NL = 100, 20                        # interior nodes, layer nodes per side
N = NI + 2 * NL
L = NL * h
SIG0 = 3 * c * np.log(1e3) / (2 * L)    # paper Eq. (sigma_0), m = 2, R0 = 1e-3
T_END = 160.0
SNAPS = [0, 40, 70, 100, 130, 150, 160]


def profile(n, nl, smax, shift):
    """sigma at positions x = j + shift (j = 0..n-1); layers of nl nodes."""
    x = np.arange(n) + shift
    dl = np.clip((nl - x) / nl, 0, 1)
    dr = np.clip((x - (n - 1 - nl)) / nl, 0, 1)
    return smax * np.maximum(dl, dr) ** 2


def run(n, nl, mode, smax=SIG0, record=None):
    off = (n - N) // 2                 # interior frame inside a bigger grid
    xc = n // 2
    x = np.arange(n) - xc
    v = np.exp(-0.5 * (x[:, None]**2 + x[None, :]**2) / 3.0**2)
    rx = np.zeros((n, n))              # at (i+1/2, j)
    ry = np.zeros((n, n))              # at (i, j+1/2)
    if mode == "none" or nl == 0:
        sv = np.zeros((n, n)); sx = np.zeros((n, n)); sy = np.zeros((n, n))
    else:
        p0, ph = profile(n, nl, smax, 0.0), profile(n, nl, smax, 0.5)
        sv = p0[:, None] + p0[None, :]
        sx = ph[:, None] + p0[None, :]
        sy = p0[:, None] + ph[None, :]
        if mode == "vonly":
            sv = 2 * sv; sx = 0 * sx; sy = 0 * sy
    nsteps = int(round(T_END / dt))
    frames, energy, times = {}, [], []
    for k in range(nsteps + 1):
        t = k * dt
        if record is not None and any(abs(t - s) < 1e-9 for s in SNAPS):
            frames[round(t)] = v[off:off + N, off:off + N].copy()
        win = v[off + NL:off + N - NL, off + NL:off + N - NL]
        energy.append(np.sum(v[off:off+N, off:off+N]**2)
                      + np.sum(rx[off:off+N, off:off+N]**2)
                      + np.sum(ry[off:off+N, off:off+N]**2))
        times.append(t)
        if record is not None:
            record.append(win.copy())
        # r update (half step): r^{n+1/2}
        gx = np.zeros((n, n)); gy = np.zeros((n, n))
        gx[:-1, :] = (v[1:, :] - v[:-1, :]) / h      # rigid ghost v=0 outside
        gx[-1, :] = -v[-1, :] / h
        gy[:, :-1] = (v[:, 1:] - v[:, :-1]) / h
        gy[:, -1] = -v[:, -1] / h
        rx = ((1 - sx * dt / 2) * rx + dt * c * gx) / (1 + sx * dt / 2)
        ry = ((1 - sy * dt / 2) * ry + dt * c * gy) / (1 + sy * dt / 2)
        # v update: div r (backward differences)
        div = np.zeros((n, n))
        div[1:, :] += (rx[1:, :] - rx[:-1, :]) / h
        div[0, :] += rx[0, :] / h
        div[:, 1:] += (ry[:, 1:] - ry[:, :-1]) / h
        div[:, 0] += ry[:, 0] / h
        v = ((1 - sv * dt / 2) * v + dt * c * div) / (1 + sv * dt / 2)
    return frames, np.array(times), np.array(energy)


ref_rec = []
run(3 * N, 0, "none", record=ref_rec)   # big absorber-free reference
res = {}
for mode in ("both", "vonly", "none"):
    rec = []
    frames, t, E = run(N, NL, mode, record=rec)
    err = np.array([np.linalg.norm(a - b) for a, b in zip(rec, ref_rec)])
    res[mode] = dict(frames=frames, t=t, E=E / E[0], err=err / np.linalg.norm(ref_rec[0]))
    print(f"{mode:6s}: max interior reflection {res[mode]['err'].max():.2e}, "
          f"energy left at T={T_END:g}: {res[mode]['E'][-1]:.2e}")

# amplitude scan for both masks (reflection vs sigma_max)
scan = {}
for mode in ("both", "vonly"):
    scan[mode] = []
    for f in (0.25, 0.5, 1, 2, 4, 8):
        rec = []
        run(N, NL, mode, smax=f * SIG0, record=rec)
        e = max(np.linalg.norm(a - b) for a, b in zip(rec, ref_rec)) / np.linalg.norm(ref_rec[0])
        scan[mode].append((f, e))
    print(mode, "scan", ["%.0e@%g" % (e, f) for f, e in scan[mode]])

# ---------------------------------------------------------------------------
# Elastic (P-SV) counterpart: Virieux staggered-grid leapfrog, written here.
#   vx at (i+1/2, j), vy at (i, j+1/2), sxx/syy at (i, j), sxy at (i+1/2, j+1/2)
#   rho dv/dt = div(stress) - s_v * sigma * rho v
#   d(stress)/dt = C eps(v)  - s_s * sigma * stress
# mode 'both': s_v = s_s = 1 (paper Eq. 15); 'vonly': v damped with 2 sigma,
# stresses undamped; 'none': no damping.
# ---------------------------------------------------------------------------
CP_E, CS_E, RHO = 1.0, 1.0 / np.sqrt(3.0), 1.0
MU = RHO * CS_E**2
LAM = RHO * CP_E**2 - 2 * MU
SIG0_E = 3 * CP_E * np.log(1e3) / (2 * L)


def run_elastic(n, nl, mode, smax=SIG0_E, record=None):
    off = (n - N) // 2
    x = np.arange(n) - n // 2
    g = np.exp(-0.5 * (x[:, None]**2 + x[None, :]**2) / 3.0**2)
    vx = g.copy(); vy = 0.5 * g.copy()
    sxx = np.zeros((n, n)); syy = np.zeros((n, n)); sxy = np.zeros((n, n))
    if mode == "none" or nl == 0:
        z = np.zeros((n, n)); svx = svy = sn = ss = z
    else:
        p0, ph = profile(n, nl, smax, 0.0), profile(n, nl, smax, 0.5)
        svx = ph[:, None] + p0[None, :]
        svy = p0[:, None] + ph[None, :]
        sn = p0[:, None] + p0[None, :]
        ss = ph[:, None] + ph[None, :]
        if mode == "vonly":
            svx, svy = 2 * svx, 2 * svy
            sn = ss = np.zeros((n, n))
    def fwd_x(a):
        d = np.zeros_like(a); d[:-1] = a[1:] - a[:-1]; d[-1] = -a[-1]; return d / h
    def bwd_x(a):
        d = np.zeros_like(a); d[1:] = a[1:] - a[:-1]; d[0] = a[0]; return d / h
    def fwd_y(a): return fwd_x(a.T).T
    def bwd_y(a): return bwd_x(a.T).T
    def energy(vx, vy, sxx, syy, sxy):
        # strain energy with the isotropic compliance
        tr = sxx + syy
        e_n = (sxx**2 + syy**2 - LAM / (2 * (LAM + MU)) * tr**2) / (2 * MU)
        return 0.5 * np.sum(RHO * (vx**2 + vy**2) + e_n + sxy**2 / MU)
    nsteps = int(round(T_END / dt))
    frames, E, times = {}, [], []
    for k in range(nsteps + 1):
        t = k * dt
        sl = (slice(off, off + N), slice(off, off + N))
        if record is not None and any(abs(t - s_) < 1e-9 for s_ in SNAPS):
            frames[round(t)] = vx[sl].copy()
        E.append(energy(vx[sl], vy[sl], sxx[sl], syy[sl], sxy[sl])); times.append(t)
        if record is not None:
            w = (slice(off + NL, off + N - NL), slice(off + NL, off + N - NL))
            record.append(np.concatenate([vx[w].ravel(), vy[w].ravel()]))
        # stresses (half step): strain rates from velocities
        exx = bwd_x(vx); eyy = bwd_y(vy); exy = fwd_y(vx) + fwd_x(vy)
        sxx = ((1 - sn * dt / 2) * sxx + dt * ((LAM + 2 * MU) * exx + LAM * eyy)) / (1 + sn * dt / 2)
        syy = ((1 - sn * dt / 2) * syy + dt * (LAM * exx + (LAM + 2 * MU) * eyy)) / (1 + sn * dt / 2)
        sxy = ((1 - ss * dt / 2) * sxy + dt * MU * exy) / (1 + ss * dt / 2)
        # velocities: divergence of stress
        fx = fwd_x(sxx) + bwd_y(sxy)
        fy = bwd_x(sxy) + fwd_y(syy)
        vx = ((1 - svx * dt / 2) * vx + dt * fx / RHO) / (1 + svx * dt / 2)
        vy = ((1 - svy * dt / 2) * vy + dt * fy / RHO) / (1 + svy * dt / 2)
    return frames, np.array(times), np.array(E)


ref_e = []
run_elastic(3 * N, 0, "none", record=ref_e)
res_e = {}
for mode in ("both", "vonly", "none"):
    rec = []
    frames, t, E = run_elastic(N, NL, mode, record=rec)
    err = np.array([np.linalg.norm(a - b) for a, b in zip(rec, ref_e)])
    res_e[mode] = dict(frames=frames, t=t, E=E / E[0], err=err / np.linalg.norm(ref_e[0]))
    print(f"elastic {mode:6s}: max interior reflection {res_e[mode]['err'].max():.2e}, "
          f"energy left at T={T_END:g}: {res_e[mode]['E'][-1]:.2e}")

# JSON with all reported numbers
json.dump(dict(script="paper/experiments/sponge_fd_comparison.py",
               setup=dict(interior=NI, layer=NL, c=c, h=h, dt=dt, T_end=T_END,
                          sigma0=SIG0, pulse_width=3.0, reference="3x grid, no absorber"),
               results={m: dict(max_reflection=float(res[m]["err"].max()),
                                energy_end=float(res[m]["E"][-1])) for m in res},
               amplitude_scan={m: [dict(factor=f, max_reflection=float(e)) for f, e in scan[m]]
                               for m in scan},
               elastic=dict(cp=CP_E, cs=CS_E, rho=RHO, sigma0=SIG0_E,
                            results={m: dict(max_reflection=float(res_e[m]["err"].max()),
                                             energy_end=float(res_e[m]["E"][-1])) for m in res_e})),
          open(os.path.join(HERE, "sponge_fd_comparison_results.json"), "w"), indent=1)

# full-width paper figure: acoustic (left) and elastic v_x (right),
# rows = damping variant, plus energy panels
import sys
sys.path.insert(0, HERE)
import figstyle as fs  # noqa: E402
fs.setup()
labels = {"both": r"damped $v$ and $\mathbf{r}$", "vonly": r"damped $v$ only", "none": r"no damping"}
rowlab = {"both": "damped\n" + r"$v$ and $\mathbf{r}$", "vonly": "damped\n" + r"$v$ only", "none": "no\ndamping"}
t_ac, t_el = (40, 70, 100), (40, 100, 150)
W = fs.TEXTWIDTH
fig = plt.figure(figsize=(W, 4.15))
gs = fig.add_gridspec(3, 7, width_ratios=[1, 1, 1, 0.14, 1, 1, 1], hspace=0.05, wspace=0.05,
                      left=0.095, right=0.995, top=0.895, bottom=0.375)
for blk, (rr, tt, col0, ttl) in enumerate(((res, t_ac, 0, r"(a) acoustic, $v$"),
                                           (res_e, t_el, 4, r"(b) elastic, $v_x$"))):
    vmax = 0.25 * np.abs(rr["both"]["frames"][40]).max()
    for i, mode in enumerate(("both", "vonly", "none")):
        for j, s_ in enumerate(tt):
            ax = fig.add_subplot(gs[i, col0 + j])
            ax.imshow(rr[mode]["frames"][s_].T, origin="lower", cmap="RdBu_r",
                      vmin=-vmax, vmax=vmax, interpolation="nearest")
            for e in (NL - 0.5, N - NL - 0.5):
                ax.axvline(e, color="0.25", ls=(0, (3, 2)), lw=0.45)
                ax.axhline(e, color="0.25", ls=(0, (3, 2)), lw=0.45)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
            if i == 0:
                ax.set_title(rf"$t={s_}$", pad=2.5, fontsize=8)
            if j == 0 and blk == 0:
                ax.set_ylabel(rowlab[mode], fontsize=8, labelpad=3)
    x0 = gs[0, col0].get_position(fig).x0
    x1 = gs[0, col0 + 2].get_position(fig).x1
    fig.text(0.5 * (x0 + x1), 0.965, ttl, ha="center", va="center", fontsize=9)
cols = {"both": fs.BLUE, "vonly": fs.VERMILION, "none": fs.BLACK}
lss = {"both": "-", "vonly": "--", "none": ":"}
for k, (rr, ttl, col0) in enumerate(((res, "(c) acoustic", 0), (res_e, "(d) elastic", 4))):
    x0 = gs[0, col0].get_position(fig).x0
    x1 = gs[0, col0 + 2].get_position(fig).x1
    ax = fig.add_axes([x0, 0.085, x1 - x0, 0.195])
    for mode in ("both", "vonly", "none"):
        ax.semilogy(rr[mode]["t"], rr[mode]["E"], color=cols[mode], ls=lss[mode],
                    label=labels[mode])
    ax.set_xlabel(r"$t$", labelpad=1)
    ax.set_ylim(1e-4, 3)
    ax.set_xlim(0, T_END)
    ax.set_yticks([1e-4, 1e-2, 1])
    ax.text(0.03, 0.08, ttl, transform=ax.transAxes, ha="left", va="bottom", fontsize=8)
    if k == 0:
        ax.set_ylabel(r"energy $\mathcal{E}(t)/\mathcal{E}(0)$", fontsize=8)
        handles, labs = ax.get_legend_handles_labels()
    else:
        ax.set_yticklabels([])
fig.legend(handles, labs, loc="center", bbox_to_anchor=(0.545, 0.325), ncol=3,
           handlelength=2.2, columnspacing=1.8)
fs.save(fig, "fig_fd_comparison")
