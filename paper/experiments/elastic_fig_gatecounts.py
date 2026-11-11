"""fig_gatecounts.py -- produce figures/fig_gatecounts.png (label fig:gatecounts).

PROVENANCE
----------
All numbers read from paper/experiments/elastic_gatecounts_results.json
(elastic_gatecounts.py, qiskit 1.4.5, transpiled to the {cx, u} basis,
default transpile; Trotter block from elastic_trotter_orders.py merged as
D_trotter_order_slopes).  Visual register mirrors the acoustic companion's
paper_P3_gatecounts.png.

(a) elementary ops per Strang step (one second-order step of the full
    elastic generator, dt=0.05) vs system qubits n_sys in {5,7,9,11,13}
    (square grids a x a, a = 1..5 qubits/axis, n_sys = 2a + 3):
    totals 521/1178/2200/3595/5645 ({cx,u}) and the CX-only series
    245/554/1054/1750/2732; dotted power-law fit to the total,
    exponent 2.483858892828757, prefactor 9.45700052523121,
    r^2(loglog) = 0.9997321761638627 (annotated 2.484 / 0.99973);
    dashed quadratic fit c2 = 56.41071428571442, c1 = -382.1428571428597,
    c0 = 1046.5321428571542 (R^2 = 0.9994343022713689).
(b) Trotter splitting error vs number of steps (elastic sponge, (2,2)
    qubits per axis, n_sys=7, npml=1, sigma_max=1.0, cp=1, cs=0.5, T=4,
    n_p=5, p_max=8; reference = exact Schrodingerised evolution per
    Fourier mode so the p-discretisation floor cancels): order-1 errors
    1.683e-1/8.387e-2/4.186e-2/2.091e-2 at N = 25/50/100/200, fitted
    slope -1.0029045838772388 (r^2 = 0.9999994845419976); order-2 errors
    9.897e-3/2.471e-3/6.176e-4/1.544e-4, slope -2.0007506038388696
    (r^2 = 0.9999999622562804).  Slopes annotated as -1.003 / -2.001.

Run:  python \
        paper/experiments/fig_gatecounts.py
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
RESULTS = os.path.join(HERE, 'elastic_gatecounts_results.json')
FIGDIR = os.path.join(ROOT, 'figures')
FIG = os.path.join(FIGDIR, 'elastic_fig_gatecounts.png')

plt.rcParams.update({'font.size': 12.5, 'figure.facecolor': 'white',
                     'savefig.facecolor': 'white'})


def panel_a(ax, block):
    n = np.array([r['n_sys'] for r in block['data']], dtype=float)
    total = np.array([r['total'] for r in block['data']], dtype=float)
    cx = np.array([r['cx'] for r in block['data']], dtype=float)
    pl = block['fits']['total']['powerlaw']
    qd = block['fits']['total']['quadratic']

    ng = np.linspace(n.min(), n.max(), 200)
    ax.loglog(n, total, 'o-', color='C0', ms=7,
              label=r'total $\{\mathrm{CX},u\}$ per Strang step')
    ax.loglog(n, cx, 's-', color='tab:orange', ms=6, label='CX only')
    ax.loglog(ng, pl['prefactor'] * ng**pl['exponent'], 'k:', lw=2,
              label=rf"$\propto n_{{\rm sys}}^{{{pl['exponent']:.3f}}}$"
                    rf" ($r^2={pl['r2_loglog']:.5f}$)")
    ax.loglog(ng, qd['c2'] * ng**2 + qd['c1'] * ng + qd['c0'], '--',
              color='gray', lw=1.5,
              label=rf"quadratic, $c_2={qd['c2']:.1f}$")
    ax.set_xticks(n)
    ax.set_xticklabels([f'{int(v)}' for v in n])
    ax.minorticks_off()
    ax.set_xlabel(r'system qubits $n_{\rm sys}$')
    ax.set_ylabel(r'elementary ops per step ($\mathrm{CX}+u$)')
    ax.set_title('(a) gate cost per Strang step')
    ax.legend(loc='upper left', fontsize=10.5)


def panel_b(ax, block):
    fits = block['fits']
    styles = {'order1': ('o', 'C0', 'order 1'),
              'order2': ('s', 'tab:orange', 'order 2 (Strang)')}
    for key, (marker, color, name) in styles.items():
        N = np.array(fits[key]['num_steps'], dtype=float)
        err = np.array(fits[key]['errors_raw'], dtype=float)
        slope = fits[key]['fit_raw']['slope']
        intercept = fits[key]['fit_raw']['intercept']
        Ng = np.linspace(N.min(), N.max(), 100)
        ax.loglog(N, err, marker, color=color, ms=7, ls='none',
                  label=rf'{name}: slope ${slope:.3f}$')
        # fit was done in natural log: log err = slope*log N + intercept
        ax.loglog(Ng, np.exp(intercept) * Ng**slope, ':', color=color, lw=1.5)
    ax.set_xticks([25, 50, 100, 200])
    ax.set_xticklabels(['25', '50', '100', '200'])
    ax.minorticks_off()
    ax.set_xlabel('Trotter steps $N$')
    ax.set_ylabel('splitting error')
    ax.set_title('(b) Trotter error vs steps')
    ax.legend(loc='lower left', fontsize=10.5)


def main():
    with open(RESULTS) as fh:
        results = json.load(fh)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    panel_a(axes[0], results['B_ops_per_strang_step'])
    panel_b(axes[1], results['D_trotter_order_slopes'])
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(FIG, dpi=220, bbox_inches='tight')
    print(f"wrote {FIG}")


if __name__ == '__main__':
    main()
