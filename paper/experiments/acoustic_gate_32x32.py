"""Gate-level 32x32 absorbing 2D wave simulation (21 qubits).

Assembles the Schrodingerised circuit from once-transpiled blocks
(initialize+QFT^dag, Strang step, QFT) to avoid re-transpiling hundreds of
identical Trotter steps, verifies the assembly against
Schrodingerisation.get_circuit on a short run, then produces snapshots at
T = 8, 16, 24 and the comparison figure.
"""
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate, DiagonalGate
from qiskit_aer import StatevectorSimulator
from scipy.sparse.linalg import expm_multiply

from lib import (WaveEquationSponge2d, WaveEquationSeparatedEvolution2d,
                 Schrodingerisation)
from lib.evolution import get_bell_evolution_circ_strang

nx = ny = 5
N = 2**nx
npml = 8
n_p, p_max = 9, 18.0
dt = 0.1
times = [8.0, 16.0, 24.0]

sponge = WaveEquationSponge2d(nx, ny, npml=npml)
sep = WaveEquationSeparatedEvolution2d(nx, ny)
bump = np.exp(-0.5 * ((np.arange(N) - N // 2) / 1.5) ** 2)
z0 = sponge.initial_state(np.outer(bump, bump))
sch = Schrodingerisation(sponge.hamiltonian_parts(), num_qubits_p=n_p, p_max=p_max)

n_sys = sponge.num_qubits
n = n_sys + n_p
sys_qubits = list(range(n_sys))
p_qubits = list(range(n_sys, n))
backend = StatevectorSimulator()
angles = dt * np.outer(sch.eta, sponge.sigma_blocks).ravel()

t0 = time.time()


def tp(circ):
    return transpile(circ, backend, optimization_level=1)


# building blocks, each transpiled once
W0 = sch.initial_w(z0).ravel()
norm0 = np.linalg.norm(W0)
prep = QuantumCircuit(n)
prep.initialize(W0 / norm0, prep.qubits)
prep.append(QFTGate(n_p).inverse(), p_qubits)
prep = tp(prep)

damp_full = QuantumCircuit(n)
damp_full.append(DiagonalGate(np.exp(1j * angles)), damp_full.qubits)
damp_full = tp(damp_full)

damp_half = QuantumCircuit(n)
damp_half.append(DiagonalGate(np.exp(0.5j * angles)), damp_half.qubits)
damp_half = tp(damp_half)

wave = QuantumCircuit(n)
wave.compose(get_bell_evolution_circ_strang(sep, dt), qubits=sys_qubits, inplace=True)
wave = tp(wave)

post = QuantumCircuit(n)
post.append(QFTGate(n_p), p_qubits)
post = tp(post)
print(f'blocks transpiled in {time.time() - t0:.0f}s '
      f'(wave step: {sum(wave.count_ops().values())} ops)', flush=True)


def assemble(num_steps):
    """prep . D_half . [W . D_full]^(n-1) . W . D_half . post"""
    qc = prep.copy()
    qc.compose(damp_half, inplace=True)
    for k in range(num_steps):
        qc.compose(wave, inplace=True)
        qc.compose(damp_full if k < num_steps - 1 else damp_half, inplace=True)
    qc.compose(post, inplace=True)
    return qc


def run(qc):
    return backend.run(qc).result().get_statevector().data


# correctness guard: assembly == get_circuit on a short run
guard_steps = 5
sv_a = run(assemble(guard_steps))
circ_ref = sch.get_circuit(z0, guard_steps * dt, num_steps=guard_steps,
                           system_circuit_factory=lambda d: get_bell_evolution_circ_strang(sep, d),
                           sigma_blocks=sponge.sigma_blocks)
sv_b = run(tp(circ_ref))
guard = np.linalg.norm(sv_a - sv_b)
print(f'assembly guard vs get_circuit: {guard:.2e}', flush=True)
assert guard < 1e-10

A = sponge.generator().tocsc()
fields_gate, fields_ref, errors, energies = [], [], [], []
for T in times:
    t1 = time.time()
    sv = run(assemble(int(round(T / dt))))
    z_gate, ps = sch.recover_from_statevector(sv, norm0)
    z_ref = expm_multiply(A * T, z0)
    err = np.linalg.norm(z_gate - z_ref) / np.linalg.norm(z_ref)
    errors.append(err)
    energies.append(np.linalg.norm(z_ref) ** 2)
    fields_gate.append(z_gate[:N * N].real.reshape(N, N).T)
    fields_ref.append(z_ref[:N * N].real.reshape(N, N).T)
    print(f'T={T:5.1f}: {int(round(T/dt))} steps, rel err = {err:.3e}, '
          f'||z||^2 = {energies[-1]:.2e}  ({time.time() - t1:.0f}s)', flush=True)

fig, ax = plt.subplots(2, len(times) + 1, figsize=(17, 8))
v0_field = z0[:N * N].real.reshape(N, N).T
for row in (0, 1):
    ax[row, 0].imshow(v0_field, origin='lower', cmap='RdBu_r', vmin=-0.05, vmax=0.05)
    ax[row, 0].set_title('t = 0')
for i, T in enumerate(times):
    ax[0, i + 1].imshow(fields_gate[i], origin='lower', cmap='RdBu_r', vmin=-0.05, vmax=0.05)
    ax[0, i + 1].set_title(f'quantum gates, t = {T:g}')
    ax[1, i + 1].imshow(fields_ref[i], origin='lower', cmap='RdBu_r', vmin=-0.05, vmax=0.05)
    ax[1, i + 1].set_title(f'classical $e^{{At}}$, t = {T:g}')
for a in ax.ravel():
    for s in (npml - 0.5, N - 0.5 - npml):
        a.axvline(s, color='k', ls='--', lw=0.6)
        a.axhline(s, color='k', ls='--', lw=0.6)
fig.suptitle('Gate-level absorbing 2D wave, 32x32 grid '
             f'(21 qubits, {int(times[-1]/dt)} Trotter steps at T={times[-1]:g})', y=0.99)
fig.savefig(os.path.join(ROOT, 'figures', 'fig6_2d_gate_level_32x32.png'), dpi=120, bbox_inches='tight')
np.savez(os.path.join(ROOT, 'figures', 'fig6_data.npz'), fields_gate=np.array(fields_gate),
         fields_ref=np.array(fields_ref), errors=np.array(errors),
         energies=np.array(energies), times=np.array(times))
import json
import qiskit
import qiskit_aer
json.dump(dict(script='paper/experiments/acoustic_gate_32x32.py', grid='32x32',
               num_qubits=n, n_sys=n_sys, n_p=n_p, p_max=p_max, npml=npml, dt=dt,
               sigma_max=float(sponge.sigma_blocks.max() / 2), pulse_width=1.5,
               versions=dict(qiskit=qiskit.__version__, qiskit_aer=qiskit_aer.__version__),
               runs=[dict(T=float(T), steps=int(round(T / dt)), rel_error=float(e),
                          energy_ref=float(E)) for T, e, E in zip(times, errors, energies)],
               wall_s=round(time.time() - t0, 1)),
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'acoustic_gate_32x32_results.json'), 'w'), indent=1)
print(f'TOTAL {time.time() - t0:.0f}s; errors: '
      + ', '.join(f'{e:.2e}' for e in errors), flush=True)
