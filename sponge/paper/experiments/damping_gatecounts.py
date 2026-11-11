"""CX cost of one damping factor exp(i dt D_eta (x) Sigma): generic
DiagonalGate vs the structured construction (lib/structured_damping.py),
plus the wave (Bell, Strang) step for scale.  Acoustic and elastic sponges,
2^n x 2^n grids, layer width npml = N/8 (fixed physical fraction), n_p = 9.

CX counts: ancilla-free variant by whole-circuit transpile to {cx, u}
(optimization_level=1, verified exact); flag variant by per-gate synthesis
(each distinct gate verified exact) -- see the comment in job().  Generic diagonal CX
is measured by transpile up to n = MAX_GENERIC_N and equals 2^n - 2.

Run:  python paper/experiments/damping_gatecounts.py   (8 worker processes)
Out:  paper/experiments/damping_gatecounts_results.json
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

N_P, P_MAX, DT = 9, 10.0, 0.1
SIZES = [3, 4, 5, 6, 7]
MAX_GENERIC_N = 14
OUT = os.path.join(HERE, "damping_gatecounts_results.json")


def count_structured(circ):
    from lib.structured_damping import cx_count_per_gate
    return cx_count_per_gate(circ)


def build(kind, nq):
    from lib import (WaveEquationSponge2d, ElasticSponge2d, Schrodingerisation,
                     WaveEquationSeparatedEvolution2d, ElasticSeparatedEvolution2d)
    N = 2**nq
    npml = max(1, N // 8)
    if kind == "acoustic":
        op = WaveEquationSponge2d(nq, nq, npml=npml)
        sep = WaveEquationSeparatedEvolution2d(nq, nq)
        nb = 4
    else:
        op = ElasticSponge2d(nq, nq, npml=npml, cp=1.0, cs=1 / np.sqrt(3))
        sep = ElasticSeparatedEvolution2d(nq, nq, cp=1.0, cs=1 / np.sqrt(3))
        nb = 8
    sch = Schrodingerisation(op.hamiltonian_parts(), num_qubits_p=N_P, p_max=P_MAX)
    return op, sep, nb, sch, npml


def job(args):
    kind, nq = args
    from qiskit import transpile
    from lib.evolution import get_bell_evolution_circ_strang
    from lib.structured_damping import damping_circuit, damping_terms, dyadic_poly_pieces
    t0 = time.time()
    op, sep, nb, sch, npml = build(kind, nq)
    n_sys = op.num_qubits
    n_tot = n_sys + N_P
    terms, _ = damping_terms(sch, op.sigma_blocks, nb, (nq, nq))
    direct = damping_circuit(sch, op.sigma_blocks, nb, (nq, nq), DT)
    flag = damping_circuit(sch, op.sigma_blocks, nb, (nq, nq), DT, flag_ancilla=True)
    max_ctrl = max(len(f) - 1 for _, f in terms)
    res = dict(kind=kind, nq=nq, N=2**nq, npml=npml, n_sys=n_sys, n_p=N_P,
               n_total=n_tot, num_terms=len(terms), max_controls=max_ctrl,
               pieces_per_profile=len(dyadic_poly_pieces(
                   op.sigma_blocks[:2**(2 * nq)].reshape(2**nq, 2**nq)[:, 2**nq // 2])),
               cx_structured_direct=count_structured(direct),
               cx_structured_flag=count_structured(flag),
               cx_generic_formula=2**n_tot - 2)
    wave = get_bell_evolution_circ_strang(sep, DT)
    tw = transpile(wave, basis_gates=["cx", "u"], optimization_level=1)
    res["cx_wave_step"] = int(tw.count_ops().get("cx", 0))
    # exact counts by full transpile (the per-gate table is a cross-check)
    # Ancilla-free variant: whole-circuit transpile (verified exact).
    # Flag variant: by default transpile() assumes all qubits start in |0>
    # and borrows idle ones as clean ancillas, so the circuit transpiled on
    # its own is not the same unitary (qubits_initially_zero=False avoids
    # this; see lib/structured_damping.py).  Its count is the per-gate
    # synthesis sum (each gate verified exact), i.e. cx_structured_flag.
    tc = transpile(direct, basis_gates=["cx", "u"], optimization_level=1)
    res["cx_full_direct"] = int(tc.count_ops().get("cx", 0))
    res["cx_full_flag"] = res["cx_structured_flag"]
    res["seconds"] = round(time.time() - t0, 1)
    return res


def generic_measure(n):
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import DiagonalGate
    rng = np.random.default_rng(0)
    qc = QuantumCircuit(n)
    qc.append(DiagonalGate(np.exp(1j * rng.uniform(0, 6, 2**n))), range(n))
    t = transpile(qc, basis_gates=["cx", "u"], optimization_level=1)
    return n, int(t.count_ops().get("cx", 0))


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(k, nq) for nq in SIZES for k in ("acoustic", "elastic")]
    ctx = get_context("fork")
    with ctx.Pool(8) as pool:
        rows = pool.map(job, jobs, chunksize=1)
        gen = pool.map(generic_measure, range(6, MAX_GENERIC_N + 1))
    out = dict(meta=dict(script="paper/experiments/damping_gatecounts.py",
                         n_p=N_P, p_max=P_MAX, dt=DT, npml_rule="N/8",
                         basis=["cx", "u"], optimization_level=1,
                         wall_s=round(time.time() - t0, 1)),
               rows=rows, generic_measured={str(n): c for n, c in gen})
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    for r in rows:
        print(r)
    print(gen)
