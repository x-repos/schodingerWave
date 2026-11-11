"""One-dimensional acoustic sponge.

(a) Classical reflection: 64-node sponge-terminated run vs the same pulse on
    a 4x enlarged undamped grid (interior window, max over T), for
    n_L in {4, 8, 16} and sigma_max/sigma_0 in {1/4, 1/2, 1, 2, 4} -- in 1D
    the continuum sponge is reflectionless (Sec. III), so what remains is
    discretisation (staggered sampling) and the outer-edge round trip.
(b) Gate-level Schrodingerised run: 64 nodes (7 system qubits) + n_p = 9,
    n_L = 8, structured damping with one flag ancilla, Strang tau = 0.05,
    T in {10, 20, 30}.

Out: acoustic_1d_demo_results.json
"""
import json
import os
import sys
import time
from multiprocessing import get_context

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
from scipy.sparse.linalg import expm_multiply  # noqa: E402

NX, NX_REF = 6, 8
N, NR = 2**NX, 2**NX_REF
OFF = (NR - N) // 2
WIDTH = 2.5
TIMES = [20.0, 40.0, 60.0, 80.0]


def state(op, n):
    x = np.arange(n)
    v = np.exp(-0.5 * ((x - n // 2) / WIDTH) ** 2)
    z = np.concatenate([v, np.zeros(n)]).astype(complex)
    return z


def window(z, n, off, nl):
    v = z[:n][off:off + N][nl:N - nl]
    q = z[n:][off:off + N][nl:N - nl]
    return np.concatenate([v, q])


def reflection(args):
    nl, f = args
    from lib import WaveEquationSponge1d, WaveEquationEvolution
    s0 = 3 * np.log(1e3) / (2 * nl)
    op = WaveEquationSponge1d(NX, npml=nl, sigma_max=f * s0)
    ref = WaveEquationEvolution(NX_REF, dim=1, diff_type="forward", periodic=False)
    z0 = state(op, N); z0 /= np.linalg.norm(z0)
    r0 = state(None, NR); r0 /= np.linalg.norm(r0)
    # linspace(0, T_end, len(TIMES) + 1) = [0] + TIMES; drop t = 0
    zs = expm_multiply(op.generator().tocsc(), z0, start=0, stop=TIMES[-1],
                       num=len(TIMES) + 1, endpoint=True)[1:]
    Ar = (-1j * ref.to_matrix(sparse=True)).tocsc()
    rs = expm_multiply(Ar, r0, start=0, stop=TIMES[-1], num=len(TIMES) + 1,
                       endpoint=True)[1:]
    errs = [float(np.linalg.norm(window(a, N, 0, nl) - window(b, NR, OFF, nl)))
            for a, b in zip(zs, rs)]
    return dict(npml=nl, factor=f, sigma_max=float(f * s0), reflection=max(errs),
                errors_by_T=dict(zip(map(str, TIMES), errs)),
                energy_end=float(np.linalg.norm(zs[-1]) ** 2))


def gate(T):
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from lib import WaveEquationSponge1d, Schrodingerisation
    from lib.evolution import get_bell_evolution_circ_strang
    from lib.structured_damping import (get_circuit_structured, damping_circuit,
                                        cx_count_per_gate)
    op = WaveEquationSponge1d(NX, npml=8)
    z0 = state(op, N); z0 /= np.linalg.norm(z0)
    sch = Schrodingerisation(op.hamiltonian_parts(), num_qubits_p=9, p_max=16.0)
    steps = int(round(T / 0.05))
    circ = get_circuit_structured(sch, z0, T, steps,
                                  lambda dt: get_bell_evolution_circ_strang(op.wave_op, dt),
                                  op.sigma_blocks, 2, (NX,), flag_ancilla=True)
    circ.save_statevector()
    sim = AerSimulator(method="statevector")
    # optimization_level=0: multi-controlled gates run natively in Aer; other
    # levels can change the global phase (see lib/structured_damping.py)
    sv = np.asarray(sim.run(transpile(circ, sim, optimization_level=0)).result().get_statevector())
    dim = 2**(circ.num_qubits - 1)
    W = sv[:dim].reshape(sch.Np, -1) * circ.metadata["norm0"]
    j = int(np.searchsorted(sch.p, 0.0)) + 3
    z = np.exp(sch.p[j]) * W[j]
    zr = expm_multiply(op.generator().tocsc() * T, z0)
    return dict(T=T, steps=steps, num_qubits=circ.num_qubits,
                rel_err=float(np.linalg.norm(z - zr) / np.linalg.norm(zr)),
                energy=float(np.linalg.norm(zr) ** 2),
                flag_leak=float(np.linalg.norm(sv[dim:])),
                cx_damping_step=cx_count_per_gate(damping_circuit(
                    sch, op.sigma_blocks, 2, (NX,), T / steps, flag_ancilla=True)))


if __name__ == "__main__":
    t0 = time.time()
    with get_context("fork").Pool(8) as pool:
        refl = pool.map(reflection, [(nl, f) for nl in (4, 8, 16)
                                     for f in (0.25, 0.5, 1.0, 2.0, 4.0)])
        gates = pool.map(gate, [10.0, 20.0, 30.0])
    out = dict(meta=dict(script="paper/experiments/acoustic_1d_demo.py", N=N,
                         reference=f"{NR} nodes undamped, centred", width=WIDTH,
                         times=TIMES, wall_s=round(time.time() - t0, 1)),
               reflection=refl, gate=gates)
    json.dump(out, open(os.path.join(HERE, "acoustic_1d_demo_results.json"), "w"), indent=1)
    for r in refl:
        print("refl", r["npml"], r["factor"], "%.2e" % r["reflection"], "E %.1e" % r["energy_end"])
    for g in gates:
        print("gate", g)
