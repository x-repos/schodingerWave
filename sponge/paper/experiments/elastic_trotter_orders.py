"""D. Trotter-order slopes for the gate-level Schrodingerised elastic sponge.

Elastic analogue of the acoustic paper's splitting-error slopes (-1.02 for
order 1, -2.01 for order 2 with both splitting levels symmetrised).

System (mirrors tests/test_elastic_circuits.py::test_gate_level_
schrodingerised_sponge): ElasticSponge2d(2, 2, npml=1, cp=1, cs=0.5,
sigma_max=1) -> n_sys = 7 system qubits (128 states, 8 blocks), warped-phase
register n_p = 5 (N_p = 32, p_max = 8), T = 4.0, Gaussian vx / 0.5*Gaussian
vy initial data.

WHAT IS COMPARED (so the p-discretisation floor cancels exactly):

  reference = the EXACT Schrodingerised evolution of the same warped initial
  state, W_exact(T) = ifft_k [ expm(-i T (eta_k H1 - H2)) fft_k W(0) ],
  computed by Schrodingerisation.evolve().  Per Fourier mode this is exactly
  expm(-i T H_tot) with H_tot = D_eta (x) H1 - I (x) H2 (the matrix
  Schrodingerisation.total_hamiltonian() returns), applied to the warped
  initial state w(0, p) = g(p) z0.  It contains the full p-discretisation
  and warping error relative to the physical solution, but ZERO splitting
  error.

  candidate = the final statevector of Schrodingerisation.get_circuit(...,
  num_steps=ns, trotter_order=ord, system_circuit_factory=f), simulated on
  qiskit-aer's StatevectorSimulator and rescaled by metadata['norm0'], i.e.
  the gate-level W_gate(T) on the same (p, system) grid.

  error(ns) = || W_gate - W_exact ||_F / || W_exact ||_F        (raw)
  plus a globally-phase-aligned variant (align W_gate to W_exact by the
  optimal phase before differencing) -- recorded both; they coincide to
  high accuracy here because every gate used (initialize, QFT, DiagonalGate,
  cx/h/p/rz/mcrz) is phase-exact.

  first order  : trotter_order=1 + get_bell_evolution_circ      (plain
                 Lie-Trotter at both levels: damping/wave splitting AND the
                 term-by-term Bell synthesis)
  second order : trotter_order=2 + get_bell_evolution_circ_strang (Strang
                 at both levels: symmetrised damping halves outside,
                 palindromic term product inside)

  num_steps in {25, 50, 100, 200}; slopes from least squares of
  ln(error) vs ln(num_steps).  Expected: ~ -1 and ~ -2.

The recovered-z error vs the exact non-unitary reference
expm(A T) z0 is recorded per point as context (it saturates at the
p-discretisation floor and is NOT used for the slope fits).

The 8 (order, num_steps) simulations run as parallel processes
(<= 8 workers, OMP_NUM_THREADS=2 per worker, per project convention).

Run from the repo root:
    python \
        paper/experiments/elastic_trotter_orders.py

Results are merged into paper/experiments/elastic_gatecounts_results.json
as section 'D_trotter_order_slopes'.
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

# problem definition (matches the unit test, n_p = 5 already small)
NQ, T, N_P, P_MAX = 2, 4.0, 5, 8.0
CP, CS, NPML, SIGMA_MAX = 1.0, 0.5, 1, 1.0
STEP_COUNTS = (25, 50, 100, 200)


def _setup():
    """Deterministic problem build (re-done per worker; cheap)."""
    from lib.elastic_sponge import ElasticSponge2d
    from lib.elastic_separated import ElasticSeparatedEvolution2d
    from lib.schrodingerisation import Schrodingerisation
    N = 2 ** NQ
    sponge = ElasticSponge2d(NQ, NQ, npml=NPML, cp=CP, cs=CS,
                             sigma_max=SIGMA_MAX)
    sep = ElasticSeparatedEvolution2d(NQ, NQ, cp=CP, cs=CS)
    x = np.arange(N) - N // 2
    g = np.exp(-0.5 * (x[:, None] ** 2 + x[None, :] ** 2) / 0.8 ** 2)
    z0 = sponge.initial_state(g, 0.5 * g)
    sch = Schrodingerisation(sponge.hamiltonian_parts(), num_qubits_p=N_P,
                             p_max=P_MAX)
    return sponge, sep, z0, sch


def trotter_job(args):
    """One (trotter order, num_steps) gate-level simulation."""
    order, num_steps = args
    from qiskit import transpile
    from qiskit_aer import StatevectorSimulator
    from scipy.linalg import expm
    from lib.evolution import (get_bell_evolution_circ,
                                get_bell_evolution_circ_strang)

    sponge, sep, z0, sch = _setup()

    # exact Schrodingerised reference: per-mode expm(-i T (eta_k H1 - H2)),
    # i.e. exactly expm(-i T H_tot) on the warped initial state
    W_exact = sch.evolve(z0, T)

    factory = (lambda dt: get_bell_evolution_circ(sep, dt)) if order == 1 \
        else (lambda dt: get_bell_evolution_circ_strang(sep, dt))
    t0 = time.time()
    circ = sch.get_circuit(z0, T, num_steps=num_steps,
                           system_circuit_factory=factory,
                           sigma_blocks=sponge.sigma_blocks,
                           trotter_order=order)
    backend = StatevectorSimulator()
    sv = backend.run(transpile(circ, backend)).result().get_statevector().data
    W_gate = np.asarray(sv).reshape(sch.Np, -1) * circ.metadata['norm0']

    d, e = W_gate.ravel(), W_exact.ravel()
    err_raw = float(np.linalg.norm(d - e) / np.linalg.norm(e))
    ph = np.vdot(e, d)
    ph /= abs(ph)
    err_aligned = float(np.linalg.norm(d / ph - e) / np.linalg.norm(e))

    # context only: recovered z vs the exact physical (non-unitary) reference
    z_ref = expm(sponge.generator().toarray() * T) @ z0
    z_gate, p_star = sch.recover_from_statevector(sv, circ.metadata['norm0'])
    err_phys = float(np.linalg.norm(z_gate - z_ref) / np.linalg.norm(z0))

    return {'trotter_order': order, 'num_steps': num_steps,
            'splitting_error_raw': err_raw,
            'splitting_error_phase_aligned': err_aligned,
            'recovered_z_error_vs_physical': err_phys,
            'p_star': float(p_star),
            'sim_seconds': round(time.time() - t0, 2)}


def _slope(ns, errs):
    s, ic = np.polyfit(np.log(np.asarray(ns, float)),
                       np.log(np.asarray(errs, float)), 1)
    fit = s * np.log(np.asarray(ns, float)) + ic
    ss_res = float(((np.log(errs) - fit) ** 2).sum())
    ss_tot = float(((np.log(errs) - np.log(errs).mean()) ** 2).sum())
    return {'slope': float(s), 'intercept': float(ic),
            'r2_loglog': 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0}


def main():
    t_start = time.time()
    jobs = [(order, ns) for order in (1, 2) for ns in STEP_COUNTS]
    ctx = get_context('spawn')
    with ctx.Pool(min(MAX_WORKERS, len(jobs))) as pool:
        rows = pool.map(trotter_job, jobs)

    fits = {}
    for order in (1, 2):
        sub = sorted((r for r in rows if r['trotter_order'] == order),
                     key=lambda r: r['num_steps'])
        ns = [r['num_steps'] for r in sub]
        fits[f'order{order}'] = {
            'num_steps': ns,
            'errors_raw': [r['splitting_error_raw'] for r in sub],
            'fit_raw': _slope(ns, [r['splitting_error_raw'] for r in sub]),
            'fit_phase_aligned': _slope(
                ns, [r['splitting_error_phase_aligned'] for r in sub]),
        }
        print(f"order {order}: errors "
              f"{['%.3e' % e for e in fits[f'order{order}']['errors_raw']]} "
              f"slope {fits[f'order{order}']['fit_raw']['slope']:+.3f} "
              f"(r2={fits[f'order{order}']['fit_raw']['r2_loglog']:.5f})")

    results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    results['D_trotter_order_slopes'] = {
        'meta': {
            'script': 'paper/experiments/elastic_trotter_orders.py',
            'system': {'num_qubits_x': NQ, 'num_qubits_y': NQ,
                       'n_sys': 2 * NQ + 3, 'npml': NPML, 'cp': CP, 'cs': CS,
                       'sigma_max': SIGMA_MAX, 'T': T, 'n_p': N_P,
                       'p_max': P_MAX},
            'reference': ('exact Schrodingerised evolution '
                          'expm(-i T (eta_k H1 - H2)) per Fourier mode '
                          '(= expm(-i T H_tot)) applied to the same warped '
                          'initial state; full-statevector Frobenius error, '
                          'p-discretisation floor cancels'),
            'order1': 'trotter_order=1 + get_bell_evolution_circ',
            'order2': ('trotter_order=2 (Strang damping halves) + '
                       'get_bell_evolution_circ_strang (both levels '
                       'symmetrised)'),
            'wall_seconds': round(time.time() - t_start, 2),
        },
        'data': rows,
        'fits': fits,
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print('wrote', RESULTS_PATH)


if __name__ == '__main__':
    main()
