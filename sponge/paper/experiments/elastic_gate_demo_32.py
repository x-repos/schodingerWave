"""32x32 gate-level elastic sponge, one time-marching pass with snapshots.

Same pipeline as elastic_gate_demo_long.py (Bell/Strang wave circuit +
structured damping with one flag ancilla, Schrodingerisation with QFTs),
at twice the spatial resolution of the 16x16 run: 32x32 grid, Gaussian
width 2.0, layer n_L = 6, sigma_max = 0.5 (the same fraction of the design
amplitude sigma_0 as the 16x16 run), tau = 0.05.

Execution (exact, no approximation beyond the circuit itself):
  * the Strang product D(t/2) [W D(t)]^{n-1} W D(t/2) is cut into blocks of
    K steps; each block body is compiled once (Aer, optimization_level=0)
    and re-run with set_statevector, so no circuit grows with T;
  * a snapshot at step n applies D(t/2) and QFT_p to a COPY of the state
    after the n-th wave circuit (exactly the closing half-step of the
    n-step Strang circuit), while the march continues with D(t).

Qubits: 13 system + 11 p + 1 flag = 25.  Usage:
  python elastic_gate_demo_32.py --calibrate   # time one 4-step block
  python elastic_gate_demo_32.py               # full run
Outputs: elastic_gate_demo_32_results.json, elastic_gate_demo_32_fields.npz
"""
import os
import sys

THREADS = int(os.environ.get("RUN_THREADS", "40"))
os.environ["OMP_NUM_THREADS"] = str(THREADS)

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

NQ, NPML, SIGMA_MAX, WIDTH = 5, 6, 0.5, 2.0
CP, CS = 1.0, 1.0 / np.sqrt(3.0)
N_P, P_MAX = 11, 40.0
DT = 0.05
K = 40                                  # steps per compiled block
SNAP_EVERY = 160                        # steps between snapshots (T = 8)
T_MAX = float(os.environ.get("T_MAX", "64"))
OUT_JSON = os.path.join(HERE, "elastic_gate_demo_32_results.json")
OUT_NPZ = os.path.join(HERE, "elastic_gate_demo_32_fields.npz")


def build():
    from lib import ElasticSponge2d, ElasticSeparatedEvolution2d, Schrodingerisation
    N = 2**NQ
    sponge = ElasticSponge2d(NQ, NQ, npml=NPML, cp=CP, cs=CS, sigma_max=SIGMA_MAX)
    sep = ElasticSeparatedEvolution2d(NQ, NQ, cp=CP, cs=CS)
    x = np.arange(N) - N // 2
    g = np.exp(-0.5 * (x[:, None]**2 + x[None, :]**2) / WIDTH**2)
    z0 = sponge.initial_state(g, 0.5 * g)
    sch = Schrodingerisation(sponge.hamiltonian_parts(), num_qubits_p=N_P, p_max=P_MAX)
    assert sch.lambda_max_plus() == 0.0
    return sponge, sep, z0, sch


def circuits(sponge, sep, sch):
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import QFTGate
    from qiskit_aer import AerSimulator
    from lib.evolution import get_bell_evolution_circ_strang
    from lib.structured_damping import damping_circuit
    n_sys = sponge.num_qubits
    n = n_sys + N_P + 1                       # + flag (most significant)
    sys_q = list(range(n_sys))
    p_q = list(range(n_sys, n_sys + N_P))
    sim = AerSimulator(method="statevector", max_parallel_threads=THREADS)
    full = damping_circuit(sch, sponge.sigma_blocks, 8, (NQ, NQ), DT, flag_ancilla=True)
    half = damping_circuit(sch, sponge.sigma_blocks, 8, (NQ, NQ), DT / 2, flag_ancilla=True)
    wave = get_bell_evolution_circ_strang(sep, DT)

    def compile_(qc):
        return transpile(qc, sim, optimization_level=0)   # level 0: see lib/structured_damping.py

    def block(first, k):
        qc = QuantumCircuit(n)
        if first:
            qc.append(QFTGate(N_P).inverse(), p_q)
            qc.compose(half, inplace=True)
        else:
            qc.compose(full, inplace=True)
        for s in range(k):
            qc.compose(wave, qubits=sys_q, inplace=True)
            if s < k - 1:
                qc.compose(full, inplace=True)
        return compile_(qc)

    snap = QuantumCircuit(n)
    snap.compose(half, inplace=True)
    snap.append(QFTGate(N_P), p_q)
    return sim, n, compile_, block, compile_(snap)


def run_block(sim, n, body, sv):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    qc.set_statevector(sv)
    qc.compose(body, inplace=True)
    qc.save_statevector()
    return np.asarray(sim.run(qc).result().get_statevector())


def main():
    calibrate = "--calibrate" in sys.argv
    t0 = time.time()
    sponge, sep, z0, sch = build()
    sim, n, compile_, block, snap = circuits(sponge, sep, sch)
    W0 = sch.initial_w(z0).ravel()
    norm0 = np.linalg.norm(W0)
    sv = np.zeros(2**n, dtype=complex)
    sv[:len(W0)] = W0 / norm0
    t_build = time.time() - t0

    if calibrate:
        b = block(True, 4)
        t1 = time.time()
        run_block(sim, n, b, sv)
        dt_run = time.time() - t1
        ops = sum(b.count_ops().values())
        print(f"qubits {n}; build {t_build:.1f}s; 4-step block: {dt_run:.1f}s "
              f"-> {dt_run/4:.2f} s/step; ops/step ~{ops/4:.0f}; threads {THREADS}")
        return

    steps_total = int(round(T_MAX / DT))
    first_body = block(True, K)
    body = block(False, K)
    A = sponge.generator().tocsc()
    from scipy.sparse.linalg import expm_multiply
    runs, fields = [], {}
    step = 0
    while step < steps_total:
        t1 = time.time()
        sv = run_block(sim, n, first_body if step == 0 else body, sv)
        step += K
        if step % SNAP_EVERY == 0:
            T = step * DT
            out = run_block(sim, n, snap, sv)
            dim = 2**(n - 1)
            leak = float(np.linalg.norm(out[dim:]))
            W = out[:dim].reshape(sch.Np, -1) * norm0
            j = int(np.searchsorted(sch.p, 0.0)) + 3
            z_gate = np.exp(sch.p[j]) * W[j]
            win = (sch.p >= 0) & (sch.p < P_MAX / 2)
            e = np.exp(-sch.p[win])
            z_win = (e[:, None] * W[win]).sum(0) / (e**2).sum()
            z_ref = expm_multiply(A * T, z0)
            nr = np.linalg.norm(z_ref)
            r = dict(grid="32x32", T=T, steps=step, num_qubits=n, n_p=N_P, p_max=P_MAX,
                     npml=NPML, sigma_max=SIGMA_MAX, width=WIDTH, dt=DT,
                     rel_error=float(np.linalg.norm(z_gate - z_ref) / nr),
                     rel_error_window=float(np.linalg.norm(z_win - z_ref) / nr),
                     prob_window=float(np.sum(np.abs(out[:dim].reshape(sch.Np, -1)[win])**2)),
                     flag_leak=leak, energy_ratio_ref=float(nr**2 / np.linalg.norm(z0)**2),
                     energy_ref=float(nr**2), energy_gate=float(np.linalg.norm(z_gate)**2),
                     wall_s=round(time.time() - t0, 1))
            runs.append(r)
            key = f"32x32_T{T:g}"
            fields[f"{key}_z_gate"] = z_gate
            fields[f"{key}_z_ref"] = z_ref
            fields[f"{key}_z0"] = z0
            print(f"[32] T={T:g} steps={step} err={r['rel_error']:.3e} "
                  f"win={r['rel_error_window']:.3e} E={r['energy_ratio_ref']:.2e} "
                  f"leak={leak:.1e} wall={r['wall_s']}s", flush=True)
            json.dump(dict(script=os.path.basename(__file__), runs=runs,
                           threads=THREADS, K=K), open(OUT_JSON, "w"), indent=2)
            np.savez_compressed(OUT_NPZ, **fields)
        else:
            print(f"  step {step}/{steps_total} ({time.time()-t1:.0f}s/block)", flush=True)
    print(f"total wall: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
