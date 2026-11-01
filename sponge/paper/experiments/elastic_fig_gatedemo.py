"""fig:gatedemo -- gate-level Schrodingerised elastic sponge demonstration.

PROVENANCE (agent H, 2026-06-12):
  - Field data: paper/experiments/elastic_gate_demo_long_fields.npz, written
    by paper/experiments/elastic_gate_demo_long.py (Trotterised
    Schrodingerised elastic sponge, Strang splitting, dt = 0.05, n_p = 10,
    p_max = 20,
    cubic recovery profile, qiskit-aer statevector; cp = 1, cs = 1/sqrt(3),
    Gaussian width 1.0). Run metadata / errors:
    paper/experiments/elastic_gate_demo_long_results.json.
  - Plotted run: 16x16 grid (nq = 4 per axis, 21 qubits total, npml = 3,
    sigma_max = 1.0), T in {8, 16, 24} with 20 Trotter steps per unit time
    (160/320/480 steps), long enough for the pulse to leave the interior.
    (The earlier T in {4, 8, 12} run with n_p = 9 still lives in
    elastic_gate_demo_fields.npz.)
  - Top row: vx (block 0) of the gate-level recovered state z(T) =
    e^{p*} norm0 w(T, p*); bottom row: the classical e^{At} z0 reference.
    Layout mirrors the acoustic companion's fig6_2d_gate_level_32x32.png.
  - Output: figures/fig_gatedemo.png (dpi 300).

Run from the repo root:
    python paper/experiments/elastic_fig_gatedemo.py --32
"""

import json
import os
import sys

import numpy as np                       # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         '..', '..'))
FIGDIR = os.path.join(REPO_ROOT, 'figures')
os.makedirs(FIGDIR, exist_ok=True)
NPZ_PATH = os.path.join(REPO_ROOT, 'paper', 'experiments',
                        'elastic_gate_demo_long_fields.npz')
JSON_PATH = os.path.join(REPO_ROOT, 'paper', 'experiments',
                         'elastic_gate_demo_long_results.json')

GRID = '16x16'
N, NPML = 16, 3
TS = None   # read from the results JSON (all snapshot times)
TS_SELECT = None
if '--32' in sys.argv:          # 32x32 run (elastic_gate_demo_32.py)
    NPZ_PATH = NPZ_PATH.replace('_long_', '_32_')
    JSON_PATH = JSON_PATH.replace('_long_', '_32_')
    GRID, N, NPML = '32x32', 32, 6
    TS_SELECT = (8.0, 16.0, 24.0, 32.0, 40.0)
for _a in sys.argv:
    if _a.startswith('--times='):
        TS_SELECT = tuple(float(x) for x in _a.split('=')[1].split(','))


def vx_field(z):
    """Block 0 (vx) of an 8-block state, index = ix * Ny + iy."""
    f = np.asarray(z)[:N * N].reshape(N, N)
    assert np.abs(f.imag).max() < 1e-6, "unexpected imaginary part"
    return f.real


def main():
    data = np.load(NPZ_PATH)
    with open(JSON_PATH) as f:
        runs = {(r['grid'], r['T']): r
                for r in json.load(f)['runs']}
    TS = tuple(sorted(T for (g, T) in runs if g == GRID))
    if TS_SELECT is not None:
        TS = tuple(T for T in TS_SELECT if T in TS)

    z0 = vx_field(data[f'{GRID}_T{TS[0]:g}_z0'])
    gate = [vx_field(data[f'{GRID}_T{T:g}_z_gate']) for T in TS]
    ref = [vx_field(data[f'{GRID}_T{T:g}_z_ref']) for T in TS]
    for T in TS:
        r = runs[(GRID, T)]
        print(f"T = {T:4.1f}: {r['steps']:3d} Trotter steps, "
              f"{r['num_qubits']} qubits, rel_error = {r['rel_error']:.4e}, "
              f"energy ref/gate = {r['energy_ref']:.4f}/"
              f"{r['energy_gate']:.4f}")

    steps_last = runs[(GRID, TS[-1])]['steps']
    nqub = runs[(GRID, TS[-1])]['num_qubits']

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import figstyle as fs
    fs.setup()
    vmax = float(np.abs(ref[0]).max())   # peak of the first propagated snapshot
    top = [z0.T] + [g.T for g in gate]
    bot = [z0.T] + [r.T for r in ref]
    notes = [[r"error" "\n" r"$0$"] + [rf"error" "\n" rf"${fs.sci(runs[(GRID, T)]['rel_error'])}$" for T in TS],
             [r"energy $1$"] + [rf"energy ${runs[(GRID, T)]['energy_ratio_ref']:.2g}$" for T in TS]]
    titles = [r"$t=0$"] + [rf"$t={T:g}$" for T in TS]
    fig, axs = fs.wavefield_grid([top, bot], ["quantum\ncircuit", "classical\nPDE"], titles,
                                 vmax, (NPML - 0.5, N - NPML - 0.5), notes=notes,
                                 cbar_label=r"$\tilde v_x/\tilde v_{\mathrm{ref}}$")
    fs.panel_label(axs[0][0], "(a)", x=0.05, y=0.95)
    fs.panel_label(axs[1][0], "(b)", x=0.05, y=0.95)
    fs.save(fig, 'elastic_fig_gatedemo_32' if '--32' in sys.argv else 'elastic_fig_gatedemo')

if __name__ == '__main__':
    main()
