"""Long-horizon 16x16 gate-level elastic sponge run for fig:gatedemo.

Same pipeline and physics as elastic_gate_demo.py (run_snapshot is reused),
but carried to T = 24 so the pulse has left the interior (classical energy
0.0053 at T = 24 vs 0.236 at T = 12).

The p window is widened: at n_p = 9, p_max = 10 the warped variable wraps
around the periodic p domain and the exact per-mode recovery fails past
T ~ 14 (rel. error 8.8e-3 at T = 16, 0.43 at T = 20, 1.45 at T = 24).
n_p = 10, p_max = 20 keeps the same dp and gives an exact per-mode recovery
floor of 1.0e-4 (T = 20) and 3.3e-4 (T = 24).

  16x16 grid (nx=ny=4): system 11 qubits + n_p=11 -> 22 qubits, npml=3,
  sigma_max=1.0, dt=0.05 (Strang), T in {8, 16, 24, 32, 40} (160..800 steps).

Outputs (written next to this script):
  elastic_gate_demo_long_results.json
  elastic_gate_demo_long_fields.npz

Run from the repo root:
    python paper/experiments/elastic_gate_demo_long.py
"""

import os

THREADS = os.environ.get("LONG_THREADS", "8")   # per worker; 5 workers x 8 = 40 cores on gpu4
os.environ["OMP_NUM_THREADS"] = THREADS
os.environ["OPENBLAS_NUM_THREADS"] = THREADS
os.environ["MKL_NUM_THREADS"] = THREADS
os.environ.setdefault("QISKIT_PARALLEL", "FALSE")

import json
import multiprocessing as mp
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from elastic_gate_demo import (CP, CS, DT, FIELD_NAMES, GAUSS_WIDTH,  # noqa: E402
                               SEED, SIGMA_MAX_STRETCH, build_problem)


def run_snapshot(task):
    """Gate-level run with the STRUCTURED damping circuit (one flag
    ancilla) instead of a dense DiagonalGate: a 2^22-entry diagonal copied
    into every one of up to 800 Trotter steps needs >50 GB per process.
    The structured circuit is the same unitary (tests/test_structured_damping.py)."""
    np.random.seed(SEED)
    from scipy.linalg import expm
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from lib.evolution import get_bell_evolution_circ_strang
    from lib.schrodingerisation import Schrodingerisation
    from lib.structured_damping import get_circuit_structured

    grid, T = task["grid"], task["T"]
    steps = int(round(T / DT))
    t_total0 = time.time()
    n_p, p_max = grid["n_p"], grid["p_max"]
    sponge, sep, z0 = build_problem(grid)
    t0 = time.time()
    z_ref = expm(sponge.generator().toarray() * T) @ z0
    t_ref = time.time() - t0
    sch = Schrodingerisation(sponge.hamiltonian_parts(), num_qubits_p=n_p, p_max=p_max)
    assert sch.lambda_max_plus() == 0.0
    t0 = time.time()
    circ = get_circuit_structured(
        sch, z0, T, steps, lambda dt: get_bell_evolution_circ_strang(sep, dt),
        sponge.sigma_blocks, 8, (grid["nq"], grid["nq"]), flag_ancilla=True)
    circ.save_statevector()
    t_build = time.time() - t0
    sim = AerSimulator(method="statevector", max_parallel_threads=int(THREADS))
    t0 = time.time()
    tc = transpile(circ, sim, optimization_level=0)
    t_transpile = time.time() - t0
    t0 = time.time()
    sv = np.asarray(sim.run(tc).result().get_statevector())
    t_run = time.time() - t0
    dim = 2**(circ.num_qubits - 1)
    leak = float(np.linalg.norm(sv[dim:]))
    W = sv[:dim].reshape(sch.Np, -1) * circ.metadata["norm0"]
    j = int(np.searchsorted(sch.p, 0.0)) + 3
    z_gate = np.exp(sch.p[j]) * W[j]
    win = (sch.p >= 0) & (sch.p < p_max / 2)
    e = np.exp(-sch.p[win])
    z_win = (e[:, None] * W[win]).sum(0) / (e**2).sum()
    nr = np.linalg.norm(z_ref)
    res = dict(grid=grid["name"], T=T, steps=steps, dt=DT,
               num_qubits=circ.num_qubits, n_sys=circ.metadata["n_sys"],
               n_p=n_p, p_max=p_max, sigma_max=grid["sigma_max"],
               npml=grid["npml"], cp=CP, cs=float(CS), damping="structured+flag",
               rel_error=float(np.linalg.norm(z_gate - z_ref) / nr),
               rel_error_window=float(np.linalg.norm(z_win - z_ref) / nr),
               prob_window=float(np.sum(np.abs(sv[:dim].reshape(sch.Np, -1)[win])**2)),
               flag_leak=leak, p_star=float(sch.p[j]),
               energy_initial=float(np.linalg.norm(z0)**2),
               energy_ref=float(nr**2), energy_gate=float(np.linalg.norm(z_gate)**2),
               energy_ratio_ref=float(nr**2 / np.linalg.norm(z0)**2),
               circuit_ops_native={k: int(v) for k, v in tc.count_ops().items()},
               timings_s=dict(reference_expm=round(t_ref, 2),
                              circuit_build=round(t_build, 2),
                              transpile=round(t_transpile, 2),
                              aer_run=round(t_run, 2),
                              total=round(time.time() - t_total0, 2)))
    return res, dict(z_gate=z_gate, z_ref=z_ref, z0=z0)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(HERE, "elastic_gate_demo_long_results.json")
FIELDS_NPZ = os.path.join(HERE, "elastic_gate_demo_long_fields.npz")

# Extended horizon (run on gpu4, 2026-10-31): T up to 40 = 800 Strang steps.
# p window doubled again (n_p = 11, p_max = 40: same dp = 0.039 as
# n_p = 10, p_max = 20) so the absorbed mass cannot wrap the periodic p axis.
LONG_GRID = dict(name="16x16", nq=4, npml=3, sigma_max=SIGMA_MAX_STRETCH,
                 n_p=11, p_max=40.0, Ts=[8.0, 16.0, 24.0, 32.0, 40.0])


def main():
    t_wall0 = time.time()
    results = {"script": os.path.basename(__file__), "runs": [], "notes": []}
    fields = {}

    try:
        import qiskit, qiskit_aer, scipy
        results["versions"] = {"qiskit": qiskit.__version__,
                               "qiskit_aer": qiskit_aer.__version__,
                               "numpy": np.__version__,
                               "scipy": scipy.__version__}
    except Exception:
        pass

    tasks = [dict(grid=LONG_GRID, T=T) for T in LONG_GRID["Ts"]]
    with mp.get_context("spawn").Pool(processes=len(tasks)) as pool:
        out = pool.map(run_snapshot, tasks)
    for res, arrays in out:
        results["runs"].append(res)
        key = f"{res['grid']}_T{res['T']:g}"
        for nm, arr in arrays.items():
            fields[f"{key}_{nm}"] = arr
        print(f"[long] {key}: err={res['rel_error']:.3e} win={res['rel_error_window']:.3e} "
              f"E_ratio={res['energy_ratio_ref']:.4f} "
              f"t={res['timings_s']['total']}s", flush=True)

    results["total_wall_s"] = round(time.time() - t_wall0, 1)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    meta = dict(field_names=FIELD_NAMES, num_blocks=8,
                grid=json.dumps(LONG_GRID), dt=DT, cp=CP, cs=float(CS),
                gauss_width=GAUSS_WIDTH)
    np.savez_compressed(FIELDS_NPZ, **fields,
                        **{f"meta_{k}": np.asarray(v) for k, v in meta.items()})
    print(f"\nwrote {RESULTS_JSON}\nwrote {FIELDS_NPZ}")
    print(f"total wall: {results['total_wall_s']}s")


if __name__ == "__main__":
    main()
