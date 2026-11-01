"""Gate-count scaling measurements for the PML-Schrodingerisation paper.

Measures (all transpiled counts, qiskit 1.4.5 / aer 0.16.4):

1. 2D: one Strang wave Trotter step (get_bell_evolution_circ_strang of
   WaveEquationSeparatedEvolution2d(nx, nx), dt=0.1) for nx = ny in
   {2,3,4,5}; circuit lives on the system register alone (2*nx+2 qubits;
   the p register is idle during the wave factor). Counts are reported
   both transpiled to Aer's StatevectorSimulator (optimization_level=1,
   as Aer executes it: Aer keeps mcx_vchain-style blocks partially
   native) and to a hardware-style basis ['cx','u'] (elementary count).

2. The Schrodingerisation damping factor exp(+i dt D_eta (x) Sigma):
   a single DiagonalGate on the FULL register (2*nx+2 system + 9 p
   qubits), with the actual angles dt*outer(eta, sigma_blocks) of the 2D
   sponge layer.  Aer keeps 'diagonal' native (1 op); the elementary
   decomposition is the standard ucrz ladder: 2^n - 1 one-qubit + 2^n - 2
   CX = 2^(n+1) - 3 gates on n qubits.

3. QFTGate(9) (the p-register basis changes, twice per circuit).

4. 1D: one first-order wave Trotter step (get_bell_evolution_circ of
   WaveEquationCPML1d(nx, npml=2 if nx==3 else 4).wave_op, dt=0.1) for
   nx in {3..7} (nx+1 system qubits).

5. mcrz synthesis cost vs number of controls k (the per-term primitive of
   get_bell_evolution_circ) -- measured, not assumed: qiskit 1.4's
   ancilla-free mcrz routes through mcx_vchain-with-dirty-ancilla blocks
   and is LINEAR in k in this size range, not exponential.

6. Fits of ops-per-step vs total system qubit count, log-log (power law)
   and semilog (exponential), with R^2, to decide the regime.

Run from the repo root:
    python paper/experiments/gatecounts.py
Writes paper/experiments/gatecounts_results.json.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import DiagonalGate, QFTGate
from qiskit_aer import StatevectorSimulator

from lib import (Schrodingerisation, WaveEquationCPML1d,
                 WaveEquationSeparatedEvolution2d, WaveEquationSponge2d)
from lib.evolution import get_bell_evolution_circ, get_bell_evolution_circ_strang

BACKEND = StatevectorSimulator()
DT = 0.1
NUM_QUBITS_P = 9
P_MAX = 10.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'acoustic_gatecounts_results.json')


def transpiled_counts(circ, basis=None, optimization_level=1):
    if basis is None:
        tc = transpile(circ, BACKEND, optimization_level=optimization_level)
    else:
        tc = transpile(circ, basis_gates=basis,
                       optimization_level=optimization_level)
    ops = {k: int(v) for k, v in tc.count_ops().items()}
    return {'ops': ops, 'total': int(sum(ops.values())),
            'depth': int(tc.depth()),
            'cx': int(ops.get('cx', 0) + ops.get('ccx', 0))}


def fit_loglog(x, y):
    """y = a * x^b: returns (b, log-log R^2)."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    b, a = np.polyfit(lx, ly, 1)
    resid = ly - (b * lx + a)
    r2 = 1 - np.sum(resid**2) / np.sum((ly - ly.mean())**2)
    return float(b), float(np.exp(a)), float(r2)


def fit_semilog(x, y):
    """y = a * exp(b x): returns (b, semilog R^2)."""
    x, ly = np.asarray(x, float), np.log(np.asarray(y, float))
    b, a = np.polyfit(x, ly, 1)
    resid = ly - (b * x + a)
    r2 = 1 - np.sum(resid**2) / np.sum((ly - ly.mean())**2)
    return float(b), float(np.exp(a)), float(r2)


results = {'dt': DT, 'num_qubits_p': NUM_QUBITS_P,
           'qiskit_note': ('transpile(circ, StatevectorSimulator(), '
                           'optimization_level=1); elementary counts in '
                           "basis ['cx','u']")}

# ----------------------------------------------------------------- 2D wave
print('== 2D Strang wave step (system register only) ==', flush=True)
res2d = []
for nx in [2, 3, 4, 5]:
    sep = WaveEquationSeparatedEvolution2d(nx, nx)
    n_sys = 2 * nx + 2
    assert sep.num_qubits == n_sys
    circ = get_bell_evolution_circ_strang(sep, DT)
    assert circ.num_qubits == n_sys
    raw = {k: int(v) for k, v in circ.count_ops().items()}
    aer = transpiled_counts(circ)
    cxu = transpiled_counts(circ, basis=['cx', 'u'])
    rec = {'nx': nx, 'n_sys': n_sys, 'n_terms': len(sep.op_list),
           'raw_ops': raw, 'aer': aer, 'cxu': cxu}
    res2d.append(rec)
    print(f'  nx={nx} n_sys={n_sys} terms={rec["n_terms"]} '
          f'aer_total={aer["total"]} aer_depth={aer["depth"]} '
          f'cxu_total={cxu["total"]} cxu_depth={cxu["depth"]}', flush=True)
results['wave_step_2d_strang'] = res2d

# -------------------------------------------- damping DiagonalGate + QFT(9)
print('== damping DiagonalGate (full register) and QFT(9) ==', flush=True)
resdamp = []
for nx in [2, 3, 4, 5]:
    sponge = WaveEquationSponge2d(nx, nx, npml=max(1, min(2**nx // 4, 8)))
    sch = Schrodingerisation(sponge.hamiltonian_parts(), NUM_QUBITS_P, P_MAX)
    angles = DT * np.outer(sch.eta, sponge.sigma_blocks).ravel()
    n_full = 2 * nx + 2 + NUM_QUBITS_P
    assert len(angles) == 2**n_full
    qc = QuantumCircuit(n_full)
    qc.append(DiagonalGate(np.exp(1j * angles)), range(n_full))
    t0 = time.time()
    aer = transpiled_counts(qc)
    cxu = transpiled_counts(qc, basis=['cx', 'u'], optimization_level=0)
    rec = {'nx': nx, 'n_full': n_full, 'aer': aer, 'cxu': cxu,
           'cxu_formula_2^(n+1)-3': int(2**(n_full + 1) - 3),
           'transpile_seconds': round(time.time() - t0, 1)}
    resdamp.append(rec)
    print(f'  nx={nx} n_full={n_full} aer={aer["ops"]} '
          f'cxu_total={cxu["total"]} (=2^(n+1)-3: '
          f'{cxu["total"] == 2**(n_full+1)-3}) [{rec["transpile_seconds"]}s]',
          flush=True)
results['damping_diagonal_full_register'] = resdamp

qft = QuantumCircuit(NUM_QUBITS_P)
qft.append(QFTGate(NUM_QUBITS_P), range(NUM_QUBITS_P))
results['qft_p_register'] = {
    'n_p': NUM_QUBITS_P,
    'aer': transpiled_counts(qft),
    'cxu': transpiled_counts(qft, basis=['cx', 'u'])}
print(f'  QFT(9): aer={results["qft_p_register"]["aer"]["ops"]} '
      f'cxu_total={results["qft_p_register"]["cxu"]["total"]}', flush=True)

# ----------------------------------------------------------------- 1D wave
print('== 1D first-order wave step (collapsed CPML wave_op) ==', flush=True)
res1d = []
for nx in [3, 4, 5, 6, 7]:
    pml = WaveEquationCPML1d(nx, npml=2 if nx == 3 else 4)
    n_sys = nx + 1
    circ = get_bell_evolution_circ(pml.wave_op, DT)
    assert circ.num_qubits == n_sys
    raw = {k: int(v) for k, v in circ.count_ops().items()}
    aer = transpiled_counts(circ)
    cxu = transpiled_counts(circ, basis=['cx', 'u'])
    rec = {'nx': nx, 'n_sys': n_sys, 'n_terms': len(pml.wave_op.op_list),
           'raw_ops': raw, 'aer': aer, 'cxu': cxu}
    res1d.append(rec)
    print(f'  nx={nx} n_sys={n_sys} terms={rec["n_terms"]} '
          f'aer_total={aer["total"]} aer_depth={aer["depth"]} '
          f'cxu_total={cxu["total"]} cxu_depth={cxu["depth"]}', flush=True)
results['wave_step_1d_first_order'] = res1d

# ----------------------------------------------------- mcrz primitive cost
print('== mcrz cost vs number of controls (measured) ==', flush=True)
resmcrz = []
for k in range(1, 13):
    qc = QuantumCircuit(k + 1)
    qc.mcrz(0.123, list(range(k)), k)
    raw = {g: int(v) for g, v in qc.count_ops().items()}
    aer = transpiled_counts(qc)
    cxu = transpiled_counts(qc, basis=['cx', 'u'])
    resmcrz.append({'k_controls': k, 'raw_ops': raw,
                    'aer_total': aer['total'], 'cxu_total': cxu['total'],
                    'cxu_cx': cxu['ops'].get('cx', 0)})
    print(f'  k={k:2d} raw={raw} aer_total={aer["total"]:4d} '
          f'cxu_total={cxu["total"]:4d} cx={cxu["ops"].get("cx", 0):4d}',
          flush=True)
results['mcrz_vs_controls'] = resmcrz

# ----------------------------------------------------------------- fits
print('== fits ==', flush=True)
fits = {}

# term count vs nx (QTT structure)
for tag, recs, key in [('terms_2d_vs_nx', res2d, 'nx'),
                       ('terms_1d_vs_nx', res1d, 'nx')]:
    x = [r[key] for r in recs]
    y = [r['n_terms'] for r in recs]
    slope, intercept = np.polyfit(x, y, 1)
    fits[tag] = {'x': x, 'y': y, 'linear_slope': float(slope),
                 'linear_intercept': float(intercept)}
    print(f'  {tag}: terms = {slope:.3f}*nx + {intercept:.3f}  (data {y})',
          flush=True)

# ops/step vs system qubits, power law vs exponential
for tag, recs, count_key in [
        ('ops2d_aer', res2d, 'aer'), ('ops2d_cxu', res2d, 'cxu'),
        ('ops1d_aer', res1d, 'aer'), ('ops1d_cxu', res1d, 'cxu')]:
    n = [r['n_sys'] for r in recs]
    y = [r[count_key]['total'] for r in recs]
    b, a, r2p = fit_loglog(n, y)
    c, a2, r2e = fit_semilog(n, y)
    fits[tag] = {'n_sys': n, 'total': y,
                 'power_law': {'exponent': b, 'prefactor': a, 'r2': r2p},
                 'exponential': {'rate': c, 'prefactor': a2, 'r2': r2e}}
    print(f'  {tag}: total={y}  ~ {a:.3g}*n^{b:.2f} (R2={r2p:.4f})  '
          f'vs {a2:.3g}*e^({c:.2f} n) (R2={r2e:.4f})', flush=True)

# quadratic fit of the elementary counts (n terms x Theta(n) mcrz -> O(n^2);
# check whether the data is already in that regime via polyfit residuals)
for tag, recs in [('ops2d_cxu_quadratic', res2d), ('ops1d_cxu_quadratic', res1d)]:
    n = np.array([r['n_sys'] for r in recs], float)
    y = np.array([r['cxu']['total'] for r in recs], float)
    coef = np.polyfit(n, y, 2)
    resid = y - np.polyval(coef, n)
    r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean())**2)
    fits[tag] = {'n_sys': n.tolist(), 'total': y.tolist(),
                 'coeffs_c2_c1_c0': [float(c) for c in coef],
                 'r2': float(r2),
                 'max_rel_residual': float(np.max(np.abs(resid) / y))}
    print(f'  {tag}: y = {coef[0]:.2f} n^2 + {coef[1]:.2f} n + {coef[2]:.2f} '
          f'(R2={r2:.6f}, max rel resid {fits[tag]["max_rel_residual"]:.3f})',
          flush=True)

# combined 1D+2D elementary count fit
n_all = [r['n_sys'] for r in res1d + res2d]
y_all = [r['cxu']['total'] for r in res1d + res2d]
b, a, r2 = fit_loglog(n_all, y_all)
fits['ops_combined_cxu'] = {'n_sys': n_all, 'total': y_all,
                            'power_law': {'exponent': b, 'prefactor': a,
                                          'r2': r2}}
print(f'  combined cx+u: ~ {a:.3g}*n^{b:.2f} (R2={r2:.4f})', flush=True)

# depth fits
for tag, recs in [('depth2d_aer', res2d), ('depth1d_aer', res1d)]:
    n = [r['n_sys'] for r in recs]
    y = [r['aer']['depth'] for r in recs]
    b, a, r2 = fit_loglog(n, y)
    fits[tag] = {'n_sys': n, 'depth': y,
                 'power_law': {'exponent': b, 'prefactor': a, 'r2': r2}}
    print(f'  {tag}: depth={y} ~ {a:.3g}*n^{b:.2f} (R2={r2:.4f})', flush=True)

# mcrz scaling (k >= 5 = mcx_vchain regime)
ks = [r['k_controls'] for r in resmcrz if r['k_controls'] >= 5]
ys = [r['cxu_total'] for r in resmcrz if r['k_controls'] >= 5]
slope, intercept = np.polyfit(ks, ys, 1)
b, a, r2p = fit_loglog(ks, ys)
c, _, r2e = fit_semilog(ks, ys)
fits['mcrz_cxu_k_ge_5'] = {
    'k': ks, 'cxu_total': ys,
    'linear': {'slope': float(slope), 'intercept': float(intercept)},
    'power_law': {'exponent': b, 'r2': r2p},
    'exponential': {'rate': c, 'r2': r2e}}
print(f'  mcrz(k>=5) cx+u: ~ {slope:.1f}*k + {intercept:.1f}; '
      f'power n^{b:.2f} (R2={r2p:.4f}) vs exp rate {c:.2f} (R2={r2e:.4f})',
      flush=True)

results['fits'] = fits

# ------------------------------------------------------- total-count formula
p2d = fits['ops2d_aer']['power_law']
p2dc = fits['ops2d_cxu']['power_law']
results['conclusion'] = {
    'per_step_aer': (f"ops/step(2D, Aer-native) = "
                     f"{p2d['prefactor']:.3g} * n^{p2d['exponent']:.2f}"),
    'per_step_cxu': (f"ops/step(2D, cx+u) = "
                     f"{p2dc['prefactor']:.3g} * n^{p2dc['exponent']:.2f}"),
    'steps': ('Strang/2nd-order Trotter: per-step error O(dt^3), '
              'accumulated O(T dt^2) -> dt = sqrt(eps_trotter/(C T)), '
              'steps = T/dt = sqrt(C) T^{3/2} eps_trotter^{-1/2}'),
    'total': ('G_total ~ steps * [ops/step(n) + diagonal-damping] + 2*QFT(n_p): '
              'polynomial in n per step; the damping DiagonalGate is native '
              'on Aer (1 op) but 2^(n+n_p+1)-3 elementary gates if '
              'synthesised generically'),
}

with open(OUT, 'w') as f:
    json.dump(results, f, indent=1)
print(f'wrote {OUT}', flush=True)
