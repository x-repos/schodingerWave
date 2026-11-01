"""Circuit-level validation: op-string encoding, Bell synthesis, and the
end-to-end gate-level Schrödingerised elastic sponge."""

import numpy as np
import pytest
from scipy.linalg import expm

from lib.elastic import ElasticWaveEvolution2d
from lib.elastic_sponge import ElasticSponge2d
from lib.elastic_separated import ElasticSeparatedEvolution2d
from lib.evolution import get_bell_evolution_circ, get_bell_evolution_circ_strang
from lib.schrodingerisation import Schrodingerisation


def test_opstring_matches_matrix():
    for nx, ny, cp, cs in ((2, 2, 1.0, 0.5), (2, 3, 1.0, 0.45)):
        sep = ElasticSeparatedEvolution2d(nx, ny, cp=cp, cs=cs)
        H_op = np.asarray(sep.to_matrix(sparse=False))
        H_ref = 1j * ElasticWaveEvolution2d(nx, ny, cp=cp, cs=cs) \
            .generator_undamped().toarray()
        assert np.abs(H_op - H_ref).max() == 0.0


def test_bell_synthesis_orders():
    from qiskit.quantum_info import Operator
    sep = ElasticSeparatedEvolution2d(2, 2, cp=1.0, cs=0.5)
    H = np.asarray(sep.to_matrix(sparse=False))

    def err(circ, dt):
        U = Operator(circ).data
        Uref = expm(-1j * H * dt)
        ph = (Uref.conj() * U).sum()
        ph /= abs(ph)
        return np.abs(U / ph - Uref).max()

    e1 = [err(get_bell_evolution_circ(sep, dt), dt) for dt in (0.2, 0.1)]
    e2 = [err(get_bell_evolution_circ_strang(sep, dt), dt) for dt in (0.2, 0.1)]
    assert e1[0] == pytest.approx(0.5 * 0.2**2, rel=0.05)  # pure Trotter
    assert e1[0] / e1[1] > 3.5                             # O(dt^2)/step
    assert e2[0] / e2[1] > 6.0                             # O(dt^3)/step


def test_gate_level_schrodingerised_sponge():
    """Full gate pipeline for the absorbing elastic system: Bell wave
    circuit + diagonal damping phases + QFTs, vs the exact non-unitary
    evolution."""
    from qiskit import transpile
    from qiskit_aer import StatevectorSimulator

    nq, T, steps, n_p, p_max = 2, 4.0, 40, 5, 8.0
    N = 2**nq
    sponge = ElasticSponge2d(nq, nq, npml=1, cp=1.0, cs=0.5, sigma_max=1.0)
    sep = ElasticSeparatedEvolution2d(nq, nq, cp=1.0, cs=0.5)
    x = np.arange(N) - N // 2
    g = np.exp(-0.5 * (x[:, None]**2 + x[None, :]**2) / 0.8**2)
    z0 = sponge.initial_state(g, 0.5 * g)

    z_ref = expm(sponge.generator().toarray() * T) @ z0

    sch = Schrodingerisation(sponge.hamiltonian_parts(), num_qubits_p=n_p,
                             p_max=p_max)
    assert sch.lambda_max_plus() == 0.0
    circ = sch.get_circuit(z0, T, num_steps=steps,
                           system_circuit_factory=lambda dt:
                               get_bell_evolution_circ_strang(sep, dt),
                           sigma_blocks=sponge.sigma_blocks)
    backend = StatevectorSimulator()
    sv = backend.run(transpile(circ, backend)).result().get_statevector().data
    z_gate, _ = sch.recover_from_statevector(sv, circ.metadata['norm0'])
    err = np.linalg.norm(z_gate - z_ref) / np.linalg.norm(z0)
    assert err < 5e-2, f"gate-level elastic recovery error {err}"
