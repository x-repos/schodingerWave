"""Gate-complexity measurements for the elastic Schrodingerisation paper.

Elastic analogues of the acoustic companion paper's complexity numbers:

  A. TERM COUNTS   len(ElasticSeparatedEvolution2d(nx, ny).op_list) on the
                   grid nx, ny in {1..5}; exact-linear law fitted as
                   count = alpha*nx + beta*ny + gamma and verified to be
                   exact (zero residual) on every grid point.
                   (Acoustic analogue: 4*nx + 4.)

  B. OPS PER STRANG STEP
                   transpile(get_bell_evolution_circ_strang(
                       ElasticSeparatedEvolution2d(a, a), dt=0.05),
                       basis_gates=['cx', 'u'])           # default opt level
                   for a = 1..5  ->  n_sys = 2a+3 = 5, 7, 9, 11, 13.
                   Records count_ops totals and cx counts; fits both a
                   log-log power law (exponent, r^2) and a quadratic
                   G(n) = c2 n^2 + c1 n + c0 (R^2, residuals).
                   (Acoustic analogue: 217/459/834/1343 at n=6/8/10/12,
                   exponent 2.63, r^2 = 0.99997, asymptotically O(n^2).)

  C. MCRZ COST     bare QuantumCircuit.mcrz with k = 4..12 controls (the
                   exact construction evolution.py uses), transpiled to
                   {cx, u}; checks/refits the ancilla-free V-chain law
                   total = 32k - 31 for k >= 8.

Independent transpiles run as parallel processes (<= 8 workers,
OMP_NUM_THREADS=2 per worker, per project convention).

Run from the repo root:
    python \
        paper/experiments/elastic_gatecounts.py

Results are written/merged into
    paper/experiments/elastic_gatecounts_results.json
(section D, the Trotter-order slopes, is added by elastic_trotter_orders.py).
"""

import os

os.environ.setdefault('OMP_NUM_THREADS', '2')  # before numpy/BLAS load

import json
import sys
import time
from multiprocessing import get_context

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

RESULTS_PATH = os.path.join(_HERE, 'elastic_gatecounts_results.json')
MAX_WORKERS = 8
DT = 0.05
GRID = range(1, 6)          # nx, ny in {1..5} for term counts
STRANG_SIZES = range(1, 6)  # a = 1..5 -> n_sys = 5, 7, 9, 11, 13
MCRZ_KS = range(4, 13)      # k = 4..12 controls


# --------------------------------------------------------------------- #
# fitting helpers                                                        #
# --------------------------------------------------------------------- #

def _r2(y, yfit):
    y, yfit = np.asarray(y, float), np.asarray(yfit, float)
    ss_res = float(((y - yfit) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


def powerlaw_fit(n, g):
    """g ~ C * n^p via least squares on log-log; returns dict."""
    ln, lg = np.log(np.asarray(n, float)), np.log(np.asarray(g, float))
    p, lc = np.polyfit(ln, lg, 1)
    fit = np.exp(lc) * np.asarray(n, float) ** p
    return {'exponent': float(p), 'prefactor': float(np.exp(lc)),
            'r2_loglog': _r2(lg, p * ln + lc),
            'r2_linear': _r2(g, fit),
            'fitted_values': [float(v) for v in fit]}


def quadratic_fit(n, g):
    """g ~ c2 n^2 + c1 n + c0; returns dict with R^2 and residuals."""
    c2, c1, c0 = np.polyfit(np.asarray(n, float), np.asarray(g, float), 2)
    fit = np.polyval([c2, c1, c0], np.asarray(n, float))
    resid = np.asarray(g, float) - fit
    return {'c2': float(c2), 'c1': float(c1), 'c0': float(c0),
            'r2': _r2(g, fit),
            'max_abs_residual': float(np.abs(resid).max()),
            'max_rel_residual': float(np.abs(resid / np.asarray(g, float)).max()),
            'fitted_values': [float(v) for v in fit]}


# --------------------------------------------------------------------- #
# A. term counts                                                         #
# --------------------------------------------------------------------- #

def term_count_job(args):
    from lib.elastic_separated import ElasticSeparatedEvolution2d
    nx, ny = args
    sep = ElasticSeparatedEvolution2d(nx, ny)
    return {'nx': nx, 'ny': ny, 'num_qubits': sep.num_qubits,
            'terms': len(sep.op_list),
            'max_weight': max(sum(c != 'I' for c in s) for s in sep.op_list)}


def fit_term_law(rows):
    """Least-squares count = alpha*nx + beta*ny + gamma; check exactness."""
    A = np.array([[r['nx'], r['ny'], 1.0] for r in rows])
    y = np.array([float(r['terms']) for r in rows])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    alpha, beta, gamma = (float(round(c)) for c in coef)
    pred = A @ np.array([alpha, beta, gamma])
    exact = bool(np.all(pred == y))
    return {'alpha_nx': alpha, 'beta_ny': beta, 'gamma': gamma,
            'formula': f'{alpha:g}*nx + {beta:g}*ny + {gamma:g}',
            'exactly_linear': exact,
            'max_abs_residual': float(np.abs(pred - y).max()),
            'note': ('exact on the whole grid; with n_sys = nx + ny + 3 '
                     'this is 6*n_sys - 6' if exact else 'NOT exact')}


# --------------------------------------------------------------------- #
# B. ops per Strang step                                                 #
# --------------------------------------------------------------------- #

def strang_transpile_job(a):
    from qiskit import transpile
    from lib.elastic_separated import ElasticSeparatedEvolution2d
    from lib.evolution import get_bell_evolution_circ_strang
    sep = ElasticSeparatedEvolution2d(a, a)
    circ = get_bell_evolution_circ_strang(sep, DT)
    raw_ops = {k: int(v) for k, v in circ.count_ops().items()}
    t0 = time.time()
    tq = transpile(circ, basis_gates=['cx', 'u'])  # default optimization level
    ops = {k: int(v) for k, v in tq.count_ops().items()}
    return {'a': a, 'n_sys': sep.num_qubits, 'terms': len(sep.op_list),
            'raw_ops': raw_ops,
            'transpiled_ops': ops,
            'total': int(sum(ops.values())),
            'cx': int(ops.get('cx', 0)),
            'depth': int(tq.depth()),
            'transpile_seconds': round(time.time() - t0, 3)}


# --------------------------------------------------------------------- #
# C. mcrz cost                                                           #
# --------------------------------------------------------------------- #

def mcrz_job(k):
    """Bare k-controlled RZ exactly as evolution.py builds it
    (QuantumCircuit.mcrz, ancilla-free), transpiled to {cx, u}."""
    from qiskit import QuantumCircuit, transpile
    qc = QuantumCircuit(k + 1)
    qc.mcrz(0.137, list(range(k)), k)   # generic angle: no special-case folding
    tq = transpile(qc, basis_gates=['cx', 'u'])  # default optimization level
    ops = {key: int(v) for key, v in tq.count_ops().items()}
    return {'k': k, 'transpiled_ops': ops,
            'total': int(sum(ops.values())),
            'cx': int(ops.get('cx', 0)),
            'law_32k_minus_31': 32 * k - 31,
            'matches_law': bool(sum(ops.values()) == 32 * k - 31)}


def fit_mcrz_law(rows, kmin=8):
    sub = [r for r in rows if r['k'] >= kmin]
    k = np.array([r['k'] for r in sub], float)
    tot = np.array([r['total'] for r in sub], float)
    cx = np.array([r['cx'] for r in sub], float)
    sl_t, ic_t = np.polyfit(k, tot, 1)
    sl_c, ic_c = np.polyfit(k, cx, 1)
    return {'k_range': [int(k.min()), int(k.max())],
            'total_fit': {'slope': float(sl_t), 'intercept': float(ic_t),
                          'r2': _r2(tot, sl_t * k + ic_t)},
            'cx_fit': {'slope': float(sl_c), 'intercept': float(ic_c),
                       'r2': _r2(cx, sl_c * k + ic_c)},
            'law_total_32k_minus_31_exact':
                bool(all(r['matches_law'] for r in sub))}


# --------------------------------------------------------------------- #
# driver                                                                 #
# --------------------------------------------------------------------- #

def main():
    t_start = time.time()
    ctx = get_context('spawn')

    grid = [(nx, ny) for nx in GRID for ny in GRID]
    with ctx.Pool(min(MAX_WORKERS, len(grid))) as pool:
        term_rows = pool.map(term_count_job, grid)
    term_law = fit_term_law(term_rows)
    print('A. term-count law:', term_law['formula'],
          '| exactly linear:', term_law['exactly_linear'])

    sizes = list(STRANG_SIZES)
    with ctx.Pool(min(MAX_WORKERS, len(sizes))) as pool:
        strang_rows = pool.map(strang_transpile_job, sizes)
    n = [r['n_sys'] for r in strang_rows]
    tot = [r['total'] for r in strang_rows]
    cx = [r['cx'] for r in strang_rows]
    fits_B = {
        'total': {'powerlaw': powerlaw_fit(n, tot),
                  'quadratic': quadratic_fit(n, tot)},
        'cx': {'powerlaw': powerlaw_fit(n, cx),
               'quadratic': quadratic_fit(n, cx)},
    }
    print('B. ops/Strang step:', dict(zip(n, tot)),
          '| exponent %.3f (r2=%.6f)' % (
              fits_B['total']['powerlaw']['exponent'],
              fits_B['total']['powerlaw']['r2_loglog']))

    ks = list(MCRZ_KS)
    with ctx.Pool(min(MAX_WORKERS, len(ks))) as pool:
        mcrz_rows = pool.map(mcrz_job, ks)
    mcrz_fit = fit_mcrz_law(mcrz_rows, kmin=8)
    print('C. mcrz totals:', {r['k']: r['total'] for r in mcrz_rows},
          '| 32k-31 exact for k>=8:',
          mcrz_fit['law_total_32k_minus_31_exact'])

    results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    results.update({
        'meta': {
            'script': 'paper/experiments/elastic_gatecounts.py',
            'qiskit': __import__('qiskit').__version__,
            'dt': DT,
            'basis_gates': ['cx', 'u'],
            'transpile_optimization_level':
                'default (plain transpile(circ, basis_gates=...) call)',
            'wall_seconds': round(time.time() - t_start, 2),
        },
        'A_term_counts': {'grid': term_rows, 'law': term_law},
        'B_ops_per_strang_step': {'data': strang_rows, 'fits': fits_B},
        'C_mcrz_cost': {'data': mcrz_rows, 'fit_k_ge_8': mcrz_fit},
    })
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print('wrote', RESULTS_PATH)


if __name__ == '__main__':
    main()
