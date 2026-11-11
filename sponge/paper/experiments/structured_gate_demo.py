"""End-to-end gate-level sponge runs with the STRUCTURED damping circuit
(no DiagonalGate anywhere in the time loop): acoustic 8x8 (17 + 1 flag
qubits) and elastic 8x8 (18 + 1 flag qubits).

Every Trotter step = Bell/Strang wave circuit (elementary gates) +
structured damping exp(i dt D_eta (x) Sigma) (multi-controlled phases and
two MCX per strip/label piece onto one clean flag ancilla).  The flag
returns to |0>; we check that and slice it away.

Reports, per snapshot time: relative error of the single-slice recovery
and of the two-qubit-window estimator (top two p qubits = '10'), the
probability of that window, and per-step CX counts (wave / damping).

Run: python paper/experiments/structured_gate_demo.py
Out: paper/experiments/structured_gate_demo_results.json
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

OUT = os.path.join(HERE, "structured_gate_demo_results.json")

CASES = {
    # same parameters as the DiagonalGate demos of the source papers
    "acoustic_8x8": dict(kind="acoustic", nq=3, npml=2, sigma_max=1.0,
                         n_p=9, p_max=16.0, times=[2.5, 5.0, 7.5, 10.0],
                         steps_per_unit=20),
    "elastic_8x8": dict(kind="elastic", nq=3, npml=2, sigma_max=1.0,
                        n_p=9, p_max=10.0, times=[2.0, 4.0, 6.0, 8.0],
                        steps_per_unit=20),
}


def setup(c):
    from lib import (WaveEquationSponge2d, ElasticSponge2d, Schrodingerisation,
                     WaveEquationSeparatedEvolution2d, ElasticSeparatedEvolution2d)
    N = 2**c["nq"]
    if c["kind"] == "acoustic":
        op = WaveEquationSponge2d(c["nq"], c["nq"], npml=c["npml"], sigma_max=c["sigma_max"])
        sep = WaveEquationSeparatedEvolution2d(c["nq"], c["nq"])
        b = np.exp(-0.5 * ((np.arange(N) - N // 2) / 1.0) ** 2)
        z0 = op.initial_state(np.outer(b, b))
        nb = 4
    else:
        cs = 1 / np.sqrt(3)
        op = ElasticSponge2d(c["nq"], c["nq"], npml=c["npml"], cp=1.0, cs=cs,
                             sigma_max=c["sigma_max"])
        sep = ElasticSeparatedEvolution2d(c["nq"], c["nq"], cp=1.0, cs=cs)
        x = np.arange(N) - N // 2
        g = np.exp(-0.5 * (x[:, None]**2 + x[None, :]**2))
        z0 = op.initial_state(g, 0.5 * g)
        nb = 8
    sch = Schrodingerisation(op.hamiltonian_parts(), num_qubits_p=c["n_p"],
                             p_max=c["p_max"])
    return op, sep, z0, sch, nb


def run_one(args):
    name, T = args
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from scipy.sparse.linalg import expm_multiply
    from lib.evolution import get_bell_evolution_circ_strang
    from lib.structured_damping import (get_circuit_structured, damping_circuit,
                                        cx_count_per_gate)
    c = CASES[name]
    op, sep, z0, sch, nb = setup(c)
    steps = int(round(c["steps_per_unit"] * T))
    t0 = time.time()
    fac = lambda dt: get_bell_evolution_circ_strang(sep, dt)
    circ = get_circuit_structured(sch, z0, T, steps, fac, op.sigma_blocks, nb,
                                  (c["nq"], c["nq"]), flag_ancilla=True)
    circ.save_statevector()
    sim = AerSimulator(method="statevector")
    tc = transpile(circ, sim, optimization_level=0)
    sv = np.asarray(sim.run(tc).result().get_statevector())
    n = circ.num_qubits - 1
    dim = 2**n
    leak = float(np.linalg.norm(sv[dim:]))           # flag must be |0>
    W = sv[:dim].reshape(sch.Np, -1) * circ.metadata["norm0"]
    zT = expm_multiply(op.generator().tocsc() * T, z0)
    j = int(np.searchsorted(sch.p, 0.0)) + 3
    z_slice = np.exp(sch.p[j]) * W[j]
    win = (sch.p >= 0) & (sch.p < c["p_max"] / 2)
    # window estimator: least-squares fit of W_j = e^{-p_j} z over the window
    e = np.exp(-sch.p[win])
    z_win = (e[:, None] * W[win]).sum(0) / (e**2).sum()
    prob_win = float(np.sum(np.abs(sv[:dim].reshape(sch.Np, -1)[win])**2))
    # per-step CX
    dt = T / steps
    wave_cx = transpile(fac(dt), basis_gates=["cx", "u"], optimization_level=1).count_ops().get("cx", 0)
    # damping: sum of per-gate syntheses (a whole-circuit transpile of the
    # flag variant is not reliable, see lib/structured_damping.py)
    damp_cx = cx_count_per_gate(damping_circuit(sch, op.sigma_blocks, nb,
                                                (c["nq"], c["nq"]), dt, flag_ancilla=True))
    return dict(case=name, T=T, steps=steps, num_qubits=circ.num_qubits,
                rel_err_slice=float(np.linalg.norm(z_slice - zT) / np.linalg.norm(zT)),
                rel_err_window=float(np.linalg.norm(z_win - zT) / np.linalg.norm(zT)),
                prob_window=prob_win, flag_leak=leak,
                energy_ref=float(np.linalg.norm(zT)**2 / np.linalg.norm(z0)**2),
                cx_wave_step=int(wave_cx), cx_damping_step=int(damp_cx),
                seconds=round(time.time() - t0, 1))


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(name, T) for name, c in CASES.items() for T in c["times"]]
    with get_context("fork").Pool(8) as pool:
        rows = pool.map(run_one, jobs, chunksize=1)
    out = dict(meta=dict(script="paper/experiments/structured_gate_demo.py",
                         cases=CASES, wall_s=round(time.time() - t0, 1)),
               runs=rows)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    for r in rows:
        print(r)
