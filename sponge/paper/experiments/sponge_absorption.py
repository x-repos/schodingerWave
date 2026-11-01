"""Classical absorption quality of the sponge layer (acoustic + elastic).

Gold-standard protocol: a 32x32 sponge-terminated run is compared, on the
interior window (layers excluded), with the same initial data evolved on a
4x enlarged 128x128 grid with NO absorber (hard wall far away, centred
embedding) -- until T = 40 the enlarged-domain boundary reflections cannot
reach the window, so the windowed difference is the spurious reflection
from the sponge.  Reported: max over T in {8,...,40} of the windowed L2
error with ||z0|| = 1 (absolute reflected amplitude), and the energy left
in the 32x32 domain at T = 40.

Sweep: npml in {4, 8, 12}, sigma_max = f * sigma_0(npml) with
sigma_0 = 3 c ln(10^3) / (2 npml h) (the m = 2, R0 = 1e-3 design amplitude)
and f in {1/4, 1/2, 1, 2, 4}: the classic sponge trade-off between
transmission through a weak layer and reflection off a steep one.

Run: python paper/experiments/sponge_absorption.py  (8 workers)
Out: paper/experiments/sponge_absorption_results.json, figures/fig_absorption.png
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

NQ, NQ_REF = 5, 7
N, NR = 2**NQ, 2**NQ_REF
OFF = (NR - N) // 2
TIMES = [8.0, 16.0, 24.0, 32.0, 40.0]
NPMLS = [4, 8, 12]
FACTORS = [0.25, 0.5, 1.0, 2.0, 4.0]
CS = 1 / np.sqrt(3)
WIDTH = 1.5
OUT = os.path.join(HERE, "sponge_absorption_results.json")


def init_fields(n):
    x = np.arange(n) - n // 2
    g = np.exp(-0.5 * (x[:, None]**2 + x[None, :]**2) / WIDTH**2)
    return g


def nblocks(kind):
    return 4 if kind == "acoustic" else 8


def phys_blocks(kind):
    return 3 if kind == "acoustic" else 5


def state(kind, n, op):
    g = init_fields(n)
    if kind == "acoustic":
        return op.initial_state(g, normalize=False)
    return op.initial_state(g, 0.5 * g, normalize=False)


def window(kind, z, n, off, npml):
    """Interior window of the 32x32 frame, physical fields stacked."""
    M = n * n
    out = []
    for b in range(phys_blocks(kind)):
        F = z[b * M:(b + 1) * M].reshape(n, n)
        F = F[off:off + N, off:off + N]
        out.append(F[npml:N - npml, npml:N - npml].ravel())
    return np.concatenate(out)


def reference(kind):
    from lib import WaveEquationSponge2d, ElasticWaveEvolution2d
    if kind == "acoustic":
        op = WaveEquationSponge2d(NQ_REF, NQ_REF, npml=1, sigma_max=0.0)
        A = (-1j * op.hamiltonian()).tocsc()
    else:
        op = ElasticWaveEvolution2d(NQ_REF, NQ_REF, cp=1.0, cs=CS)
        A = op.generator_undamped().tocsc()
    z0 = state(kind, NR, op)
    nrm = np.linalg.norm(z0)
    snaps = expm_multiply(A, z0 / nrm, start=0, stop=TIMES[-1],
                          num=len(TIMES), endpoint=True)
    return kind, [s for s in snaps]


def job(args):
    kind, npml, f, refs = args
    from lib import WaveEquationSponge2d, ElasticSponge2d
    s0 = 3 * np.log(1e3) / (2 * npml)
    if kind == "acoustic":
        op = WaveEquationSponge2d(NQ, NQ, npml=npml, sigma_max=f * s0)
    else:
        op = ElasticSponge2d(NQ, NQ, npml=npml, cp=1.0, cs=CS, sigma_max=f * s0)
    z0 = state(kind, N, op)
    z0 = z0 / np.linalg.norm(z0)
    A = op.generator().tocsc()
    snaps = expm_multiply(A, z0, start=0, stop=TIMES[-1], num=len(TIMES),
                          endpoint=True)
    errs = {}
    for T, zs, zr in zip(TIMES, snaps, refs):
        a = window(kind, zs, N, 0, npml)
        b = window(kind, zr, NR, OFF, npml)
        errs[str(T)] = float(np.linalg.norm(a - b))
    return dict(kind=kind, npml=npml, factor=f, sigma_max=float(f * s0),
                errors_by_T=errs, reflection=max(errs.values()),
                energy_T40=float(np.linalg.norm(snaps[-1])**2))


def figure(res):
    import figstyle as fs
    fs.setup()
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(fs.TEXTWIDTH, 2.15), sharey=True)
    marks = {4: "o", 8: "s", 12: "^"}
    cols = {4: fs.BLUE, 8: fs.VERMILION, 12: fs.GREEN}
    for k, (ax, kind) in enumerate(zip(axs, ("acoustic", "elastic"))):
        for npml in NPMLS:
            rr = sorted([r for r in res["rows"] if r["kind"] == kind and r["npml"] == npml],
                        key=lambda r: r["factor"])
            ax.loglog([r["factor"] for r in rr], [r["reflection"] for r in rr],
                      marker=marks[npml], color=cols[npml], mfc="white", mew=0.9,
                      label=rf"$n_{{\mathrm{{L}}}}={npml}$")
        ax.set_xlabel(r"$\sigma_{\max}/\sigma_0$")
        fs.panel_label(ax, rf"({'ab'[k]}) {kind}")
        ax.set_xticks([0.25, 0.5, 1, 2, 4])
        ax.set_xticklabels([r"$1/4$", r"$1/2$", r"$1$", r"$2$", r"$4$"])
        ax.minorticks_off()
        ax.set_ylim(2.2e-2, 3.2e-1)
        ax.set_yticks([0.03, 0.05, 0.1, 0.2, 0.3])
        ax.set_yticklabels([r"$0.03$", r"$0.05$", r"$0.1$", r"$0.2$", r"$0.3$"])
        ax.yaxis.set_minor_locator(plt.NullLocator())
    axs[0].set_ylabel(r"max.\ windowed error")
    axs[0].legend(loc="upper right", handlelength=1.6)
    fig.tight_layout(w_pad=0.6)
    fs.save(fig, "fig_absorption")

if __name__ == "__main__":
    if "--fig-only" in sys.argv:
        figure(json.load(open(OUT)))
        sys.exit()
    t0 = time.time()
    ctx = get_context("fork")
    with ctx.Pool(2) as pool:
        refs = dict(pool.map(reference, ["acoustic", "elastic"]))
    jobs = [(k, n, f, refs[k]) for k in ("acoustic", "elastic")
            for n in NPMLS for f in FACTORS]
    with ctx.Pool(8) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    res = dict(meta=dict(script="paper/experiments/sponge_absorption.py",
                         grid="32x32", reference="128x128 undamped, centred",
                         times=TIMES, width=WIDTH, cs=CS,
                         wall_s=round(time.time() - t0, 1)), rows=rows)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    for r in rows:
        print(r["kind"], r["npml"], r["factor"], "%.2e" % r["reflection"],
              "E40 %.2e" % r["energy_T40"])
    figure(res)
