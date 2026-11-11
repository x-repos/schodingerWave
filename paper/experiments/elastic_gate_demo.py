"""Headline gate-level demonstration: Trotterised Schrodingerised 2D
ELASTIC sponge (CAP) absorbing simulation, validated against the exact
non-unitary evolution exp(A T) z0.

Pipeline (per snapshot time T, one circuit per T):
  ElasticSponge2d (8 blocks: [vx~, vy~, s_xx, s_yy, s_xy, 0, 0, 0], damping
  sigma_x + sigma_y on the five physical fields)  ->  Schrodingerisation
  (warped phase transform, p register, cubic profile)  ->  Qiskit circuit:
  initialize(g (x) z0) + QFT^dag(p) + Strang steps [diagonal damping phases
  exp(+i dt D_eta (x) Sigma) interleaved with the Bell-basis elastic wave
  circuit get_bell_evolution_circ_strang(dt)] + QFT(p)  ->  Aer statevector
  ->  slice recovery z(T) = e^{p*} norm0 w(T, p*).

Runs (all dt = 0.05, Strang / trotter_order=2):
  MAIN    8x8 grid  (nx=ny=3): system 9 qubits + n_p=9 -> 18 qubits,
          npml=2, cp=1, cs=1/sqrt(3), sigma_max=SIGMA_MAX_MAIN, h=1,
          T in {2, 4, 6, 8}  (steps 40/80/120/160).
  STRETCH 16x16 grid (nx=ny=4): system 11 qubits + n_p=9 -> 20 qubits,
          npml=3, same speeds, T in {4, 8, 12}. Time-boxed; abandoned
          gracefully if a single run exceeds the budget.

Reference: dense expm(A T) z0 with A = sponge.generator() (dimension
8 * 2^(nx+ny): 512 for 8x8, 2048 for 16x16).

Parallelism: one worker process per (grid, T) snapshot, up to 8 workers
(multiprocessing 'spawn'); each worker limits BLAS/Aer to 2 threads via
OMP/OPENBLAS/MKL_NUM_THREADS (set below, before numpy import, so spawned
children inherit the limit when they re-import this module).

Outputs (written next to this script):
  elastic_gate_demo_results.json  -- configs, errors, energies, op counts,
                                     timings.
  elastic_gate_demo_fields.npz    -- recovered + reference state vectors and
                                     layout metadata for figure scripts.

Deterministic: no stochastic primitives (statevector simulation throughout);
seeds are set anyway for belt and braces.
"""

import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ.setdefault("QISKIT_PARALLEL", "FALSE")

import json
import multiprocessing as mp
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))

SEED = 12345
DT = 0.05
N_P = 9            # p-register qubits
P_MAX = 10.0       # p in [-10, 10)
CP = 1.0
CS = 1.0 / np.sqrt(3.0)   # 0.577350269189626 (Poisson-solid default)
SIGMA_MAX_MAIN = 1.0      # chosen by the classical pre-scan (see JSON notes)
SIGMA_MAX_STRETCH = 1.0
GAUSS_WIDTH = 1.0         # grid units; pulse resolved on the 8x8 grid
STRETCH_BUDGET_S = 25 * 60
FIELD_NAMES = ["vx", "vy", "sxx", "syy", "sxy", "pad0", "pad1", "pad2"]

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_JSON = os.path.join(HERE, "elastic_gate_demo_results.json")
FIELDS_NPZ = os.path.join(HERE, "elastic_gate_demo_fields.npz")

MAIN_GRID = dict(name="8x8", nq=3, npml=2, sigma_max=SIGMA_MAX_MAIN,
                 Ts=[2.0, 4.0, 6.0, 8.0])
STRETCH_GRID = dict(name="16x16", nq=4, npml=3, sigma_max=SIGMA_MAX_STRETCH,
                    Ts=[4.0, 8.0, 12.0])


def build_problem(grid):
    """Sponge + separated op-string Hamiltonian + initial state for a grid."""
    from lib.elastic_sponge import ElasticSponge2d
    from lib.elastic_separated import ElasticSeparatedEvolution2d

    nq = grid["nq"]
    N = 2**nq
    sponge = ElasticSponge2d(nq, nq, npml=grid["npml"], cp=CP, cs=CS,
                             sigma_max=grid["sigma_max"])
    sep = ElasticSeparatedEvolution2d(nq, nq, cp=CP, cs=CS)
    x = np.arange(N) - N // 2
    g = np.exp(-0.5 * (x[:, None]**2 + x[None, :]**2) / GAUSS_WIDTH**2)
    z0 = sponge.initial_state(g, 0.5 * g)   # normalised
    return sponge, sep, z0


def run_snapshot(task):
    """Worker: one gate-level Schrodingerised run at time T vs expm ref."""
    np.random.seed(SEED)
    from scipy.linalg import expm
    from qiskit import transpile
    from qiskit_aer import StatevectorSimulator

    from lib.evolution import get_bell_evolution_circ_strang
    from lib.schrodingerisation import Schrodingerisation

    grid, T = task["grid"], task["T"]
    steps = int(round(T / DT))
    t_total0 = time.time()

    n_p = grid.get("n_p", N_P)
    p_max = grid.get("p_max", P_MAX)
    sponge, sep, z0 = build_problem(grid)

    t0 = time.time()
    z_ref = expm(sponge.generator().toarray() * T) @ z0
    t_ref = time.time() - t0

    sch = Schrodingerisation(sponge.hamiltonian_parts(), num_qubits_p=n_p,
                             p_max=p_max)
    lam_plus = sch.lambda_max_plus()
    assert lam_plus == 0.0, f"sponge not dissipative: lambda_max_plus={lam_plus}"

    t0 = time.time()
    circ = sch.get_circuit(z0, T, num_steps=steps,
                           system_circuit_factory=lambda dt:
                               get_bell_evolution_circ_strang(sep, dt),
                           sigma_blocks=sponge.sigma_blocks)
    t_build = time.time() - t0

    backend = StatevectorSimulator()
    t0 = time.time()
    tc = transpile(circ, backend)
    t_transpile = time.time() - t0

    t0 = time.time()
    sv = backend.run(tc).result().get_statevector().data
    t_run = time.time() - t0

    z_gate, p_star = sch.recover_from_statevector(sv, circ.metadata["norm0"])
    err = float(np.linalg.norm(z_gate - z_ref) / np.linalg.norm(z_ref))

    # plateau diagnostic on the gate-level W (resolution sanity check)
    W = sv.reshape(sch.Np, -1) * circ.metadata["norm0"]
    plateau = sch.recovery_plateau(W, z_ref=z_ref)

    res = dict(
        grid=grid["name"], T=T, steps=steps, dt=DT,
        num_qubits=circ.num_qubits, n_sys=circ.metadata["n_sys"], n_p=n_p,
        p_max=p_max, sigma_max=grid["sigma_max"], npml=grid["npml"],
        cp=CP, cs=float(CS), lambda_max_plus=float(lam_plus),
        rel_error=err, p_star=float(p_star),
        energy_initial=float(np.linalg.norm(z0)**2),
        energy_ref=float(np.linalg.norm(z_ref)**2),
        energy_gate=float(np.linalg.norm(z_gate)**2),
        energy_ratio_ref=float(np.linalg.norm(z_ref)**2
                               / np.linalg.norm(z0)**2),
        plateau_spread=float(plateau["spread"]),
        plateau_best_err=float(np.min(plateau["rel_errors"])),
        circuit_size_native=int(tc.size()),
        circuit_ops_native={k: int(v) for k, v in tc.count_ops().items()},
        timings_s=dict(reference_expm=round(t_ref, 2),
                       circuit_build=round(t_build, 2),
                       transpile=round(t_transpile, 2),
                       aer_run=round(t_run, 2),
                       total=round(time.time() - t_total0, 2)),
    )
    arrays = dict(z_gate=z_gate, z_ref=z_ref, z0=z0)
    return res, arrays


def count_step_ops(grid, include_full_step=True):
    """Transpile ONE Strang step to cx+u and count.

    system_only : the Bell Strang wave circuit on the system register.
    full_step   : wave + one full-step diagonal damping phase gate on
                  (system + p) qubits (the QFTs and initialize are once-per-
                  circuit, not per step). The 2^n-entry diagonal synthesis is
                  the dominant cost and is only attempted when cheap.
    """
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import DiagonalGate

    from lib.evolution import get_bell_evolution_circ_strang
    from lib.schrodingerisation import Schrodingerisation

    sponge, sep, _ = build_problem(grid)
    sch = Schrodingerisation(sponge.hamiltonian_parts(), num_qubits_p=N_P,
                             p_max=P_MAX)
    out = {}

    step = get_bell_evolution_circ_strang(sep, DT)
    t0 = time.time()
    tc = transpile(step, basis_gates=["cx", "u"], optimization_level=1)
    out["system_only"] = dict(
        ops={k: int(v) for k, v in tc.count_ops().items()},
        depth=int(tc.depth()), size=int(tc.size()),
        num_qubits=int(step.num_qubits),
        raw_ops={k: int(v) for k, v in step.count_ops().items()},
        transpile_s=round(time.time() - t0, 2))

    if include_full_step:
        n_sys = sep.num_qubits
        n = n_sys + N_P
        angles = DT * np.outer(sch.eta, sponge.sigma_blocks).ravel()
        full = QuantumCircuit(n)
        full.compose(step, qubits=range(n_sys), inplace=True)
        full.append(DiagonalGate(np.exp(1j * angles)), range(n))
        t0 = time.time()
        tf = transpile(full, basis_gates=["cx", "u"], optimization_level=0)
        out["full_step"] = dict(
            ops={k: int(v) for k, v in tf.count_ops().items()},
            depth=int(tf.depth()), size=int(tf.size()),
            num_qubits=n, transpile_s=round(time.time() - t0, 2))
    return out


def main():
    np.random.seed(SEED)
    t_wall0 = time.time()
    results = {"experiment": "elastic_gate_demo",
               "description": __doc__.splitlines()[0],
               "date": "2026-06-12",
               "dt": DT, "n_p": N_P, "p_max": P_MAX,
               "cp": CP, "cs": float(CS), "h": 1.0,
               "gauss_width": GAUSS_WIDTH, "trotter": "strang",
               "profile": "cubic", "seed": SEED,
               "sigma_max_choice": {
                   "main": SIGMA_MAX_MAIN, "stretch": SIGMA_MAX_STRETCH,
                   "rationale":
                       "classical pre-scan (expm + exact per-mode "
                       "Schrodingerisation): sigma_max=1.0 absorbs the 8x8 "
                       "pulse energy to 4.5% by T=8 (vs 3.8% at 1.5, 3.7% "
                       "at 2.0 -- diminishing returns), while keeping the "
                       "warped-variable transport (speed sigma_x+sigma_y "
                       "<= 2 sigma_max in p) inside the periodic window "
                       "p in [-10,10) for T <= 8. At sigma_max=1.5 the "
                       "wrap-around contaminates the recovery slice by T=8 "
                       "(exact-mode floor 4.9e-2 > target); at 1.0 the "
                       "floor stays <= 1.4e-4 for all snapshots."},
               "runs": [], "op_counts": {}, "notes": []}
    fields = {}

    versions = {}
    try:
        import qiskit, qiskit_aer, scipy
        versions = {"qiskit": qiskit.__version__,
                    "qiskit_aer": qiskit_aer.__version__,
                    "numpy": np.__version__, "scipy": scipy.__version__}
    except Exception:
        pass
    results["versions"] = versions

    ctx = mp.get_context("spawn")

    # ---------------- MAIN: 8x8, all four T in parallel ----------------
    tasks = [dict(grid=MAIN_GRID, T=T) for T in MAIN_GRID["Ts"]]
    with ctx.Pool(processes=min(8, len(tasks))) as pool:
        main_out = pool.map(run_snapshot, tasks)
    for res, arrays in main_out:
        results["runs"].append(res)
        key = f"{res['grid']}_T{res['T']:g}"
        for nm, arr in arrays.items():
            fields[f"{key}_{nm}"] = arr
        print(f"[main] {key}: err={res['rel_error']:.3e} "
              f"E_ratio={res['energy_ratio_ref']:.4f} "
              f"t={res['timings_s']['total']}s", flush=True)

    main_errs = [r["rel_error"] for r in results["runs"]]
    results["main_max_rel_error"] = max(main_errs)

    # ---------------- op counts (8x8 per-step circuits) ----------------
    print("[ops] transpiling per-step circuits (8x8)...", flush=True)
    results["op_counts"]["8x8"] = count_step_ops(MAIN_GRID,
                                                 include_full_step=True)

    # ---------------- STRETCH: 16x16, time-boxed ----------------------
    stretch_tasks = [dict(grid=STRETCH_GRID, T=T) for T in STRETCH_GRID["Ts"]]
    pool = ctx.Pool(processes=min(8, len(stretch_tasks)))
    async_res = [(t["T"], pool.apply_async(run_snapshot, (t,)))
                 for t in stretch_tasks]
    pool.close()
    deadline = time.time() + STRETCH_BUDGET_S
    aborted = False
    for T, ar in async_res:
        try:
            remaining = max(1.0, deadline - time.time())
            res, arrays = ar.get(timeout=remaining)
        except mp.TimeoutError:
            aborted = True
            results["notes"].append(
                f"stretch 16x16 aborted: T={T:g} run still incomplete after "
                f"{STRETCH_BUDGET_S/60:.0f} min budget; remaining stretch "
                "runs abandoned")
            print(f"[stretch] T={T:g} timed out -> abandoning", flush=True)
            break
        except Exception as exc:  # noqa: BLE001 - record and continue
            results["notes"].append(f"stretch 16x16 T={T:g} failed: {exc!r}")
            print(f"[stretch] T={T:g} FAILED: {exc!r}", flush=True)
            continue
        results["runs"].append(res)
        key = f"{res['grid']}_T{res['T']:g}"
        for nm, arr in arrays.items():
            fields[f"{key}_{nm}"] = arr
        print(f"[stretch] {key}: err={res['rel_error']:.3e} "
              f"E_ratio={res['energy_ratio_ref']:.4f} "
              f"t={res['timings_s']['total']}s", flush=True)
    if aborted:
        pool.terminate()
    pool.join()

    if any(r["grid"] == "16x16" for r in results["runs"]):
        # per-step wave op count for the 16x16 system register (cheap);
        # the full-step 2^20-entry diagonal synthesis is skipped as too slow
        print("[ops] transpiling per-step circuit (16x16, system only)...",
              flush=True)
        results["op_counts"]["16x16"] = count_step_ops(
            STRETCH_GRID, include_full_step=False)
        results["op_counts"]["16x16"]["full_step"] = (
            "skipped: 2^20-entry DiagonalGate synthesis; simulated natively "
            "by Aer as one 'diagonal' instruction per step")

    # ---------------- outputs -----------------------------------------
    results["total_wall_s"] = round(time.time() - t_wall0, 1)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    meta = dict(
        field_names=FIELD_NAMES, num_blocks=8,
        layout="state index = block * block_size + (ix * Ny + iy); "
               "x register more significant than y within a block; "
               "fields vx,vy are sqrt(rho)-scaled velocities, sxx,syy,sxy "
               "the C^{-1/2}-scaled Voigt stresses (sxy carries sqrt(2))",
        grids=json.dumps({g["name"]: dict(nq=g["nq"], N=2**g["nq"],
                                          block_size=4**g["nq"],
                                          npml=g["npml"],
                                          sigma_max=g["sigma_max"],
                                          Ts=g["Ts"])
                          for g in (MAIN_GRID, STRETCH_GRID)}),
        dt=DT, n_p=N_P, p_max=P_MAX, cp=CP, cs=float(CS),
        gauss_width=GAUSS_WIDTH)
    np.savez_compressed(FIELDS_NPZ, **fields,
                        **{f"meta_{k}": np.asarray(v) for k, v in meta.items()})

    print(f"\nwrote {RESULTS_JSON}\nwrote {FIELDS_NPZ}")
    print(f"main max rel error: {results['main_max_rel_error']:.3e}")
    print(f"total wall: {results['total_wall_s']}s")


if __name__ == "__main__":
    main()
