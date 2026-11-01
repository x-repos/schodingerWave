"""2D acoustic sponge (8x8, 17 qubits): gate-level error vs number of
Strang steps at T = 10, first- and second-order outer splitting (both with
the palindromic inner Bell circuit for order 2, plain inner circuit for
order 1).  Error against exact e^{AT} psi0 and against the exact
Schrodingerised evolution (isolates the splitting).
Out: acoustic_trotter_check_results.json"""
import json, os, sys, time
from multiprocessing import get_context
os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import numpy as np

T, NQ, NPML, SM, NP, PMAX = 10.0, 3, 2, 1.0, 9, 16.0


def job(args):
    order, steps = args
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from scipy.sparse.linalg import expm_multiply
    from lib import WaveEquationSponge2d, WaveEquationSeparatedEvolution2d, Schrodingerisation
    from lib.evolution import get_bell_evolution_circ, get_bell_evolution_circ_strang
    op = WaveEquationSponge2d(NQ, NQ, npml=NPML, sigma_max=SM)
    sep = WaveEquationSeparatedEvolution2d(NQ, NQ)
    b = np.exp(-0.5 * ((np.arange(8) - 4) / 1.0) ** 2)
    z0 = op.initial_state(np.outer(b, b))
    sch = Schrodingerisation(op.hamiltonian_parts(), num_qubits_p=NP, p_max=PMAX)
    fac = (lambda dt: get_bell_evolution_circ_strang(sep, dt)) if order == 2 else \
          (lambda dt: get_bell_evolution_circ(sep, dt))
    circ = sch.get_circuit(z0, T, num_steps=steps, system_circuit_factory=fac,
                           sigma_blocks=op.sigma_blocks, trotter_order=order)
    circ.save_statevector()
    sim = AerSimulator(method="statevector")
    sv = np.asarray(sim.run(transpile(circ, sim)).result().get_statevector())
    z, _ = sch.recover_from_statevector(sv, circ.metadata["norm0"])
    zref = expm_multiply(op.generator().tocsc() * T, z0)
    zsch, _ = sch.recover(sch.evolve(z0, T))
    return dict(order=order, steps=steps,
                err_vs_exact=float(np.linalg.norm(z - zref) / np.linalg.norm(zref)),
                err_vs_schro=float(np.linalg.norm(z - zsch) / np.linalg.norm(zref)))


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(o, s) for o in (1, 2) for s in (25, 50, 100, 200)]
    with get_context("fork").Pool(8) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    fits = {}
    for o in (1, 2):
        rr = [r for r in rows if r["order"] == o]
        x = np.log([r["steps"] for r in rr]); y = np.log([r["err_vs_schro"] for r in rr])
        fits[str(o)] = float(np.polyfit(x, y, 1)[0])
    out = dict(meta=dict(T=T, grid="8x8", npml=NPML, sigma_max=SM, n_p=NP, p_max=PMAX,
                         wall_s=round(time.time() - t0, 1)), rows=rows,
               slope_vs_steps=fits)
    json.dump(out, open(os.path.join(HERE, "acoustic_trotter_check_results.json"), "w"), indent=1)
    for r in rows: print(r)
    print(fits)
