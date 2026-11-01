"""The structured damping circuit equals exp(i tau D_eta (x) Sigma) exactly."""

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from lib import (WaveEquationSponge2d, ElasticSponge2d, Schrodingerisation,
                 WaveEquationSeparatedEvolution2d, pml_sigma_profile)
from lib.evolution import get_bell_evolution_circ_strang
from lib.structured_damping import (dyadic_poly_pieces, damping_circuit,
                                    separable_components, get_circuit_structured,
                                    _eval_multilinear)


def _target_diag(sch, sigma, tau):
    return np.exp(1j * tau * np.outer(sch.eta, sigma).ravel())


@pytest.mark.parametrize("stagger", [0.0, -0.5, 0.5])
@pytest.mark.parametrize("npml", [2, 3, 4, 5])
def test_dyadic_pieces_exact(stagger, npml):
    f = pml_sigma_profile(32, npml, stagger=stagger)
    rec = np.zeros(32)
    for lo, q, co in dyadic_poly_pieces(f, m=2):
        rec[lo:lo + 2**q] += _eval_multilinear(co, q)
    assert np.abs(rec - f).max() < 1e-12


@pytest.mark.parametrize("flag", [False, True])
def test_acoustic_damping_circuit_exact(flag):
    sp_ = WaveEquationSponge2d(2, 2, npml=1, sigma_max=1.3)
    sch = Schrodingerisation(sp_.hamiltonian_parts(), num_qubits_p=3, p_max=6.0)
    tau = 0.37
    circ = damping_circuit(sch, sp_.sigma_blocks, 4, (2, 2), tau, flag_ancilla=flag)
    U = Operator(circ).data
    d = _target_diag(sch, sp_.sigma_blocks, tau)
    if flag:   # ancilla is the most significant qubit; restrict to |0>
        U = U[:len(d), :len(d)]
    assert np.abs(U - np.diag(d)).max() < 1e-10


def test_elastic_damping_circuit_exact():
    es = ElasticSponge2d(2, 2, npml=1, cp=1.0, cs=0.5, sigma_max=0.9)
    sch = Schrodingerisation(es.hamiltonian_parts(), num_qubits_p=2, p_max=5.0)
    tau = 0.21
    circ = damping_circuit(sch, es.sigma_blocks, 8, (2, 2), tau)
    U = Operator(circ).data
    d = _target_diag(sch, es.sigma_blocks, tau)
    assert np.abs(U - np.diag(d)).max() < 1e-10


def test_separable_components_default_sponge():
    sp_ = WaveEquationSponge2d(4, 4, npml=4)
    comps = separable_components(sp_.sigma_blocks, 4, (4, 4))
    # x: unstaggered on {v, wy}, staggered on {wx}; y: unstaggered on {v, wx},
    # staggered on {wy}
    assert sorted(len(b) for _, _, b in comps) == [1, 1, 2, 2]


def test_structured_pipeline_matches_diagonalgate():
    from qiskit import transpile
    from qiskit_aer import StatevectorSimulator
    nq = 2
    sp_ = WaveEquationSponge2d(nq, nq, npml=1, sigma_max=1.0)
    sep = WaveEquationSeparatedEvolution2d(nq, nq)
    bump = np.exp(-0.5 * ((np.arange(4) - 2) / 1.0) ** 2)
    z0 = sp_.initial_state(np.outer(bump, bump))
    sch = Schrodingerisation(sp_.hamiltonian_parts(), num_qubits_p=5, p_max=8.0)
    fac = lambda dt: get_bell_evolution_circ_strang(sep, dt)
    be = StatevectorSimulator()
    a = sch.get_circuit(z0, 2.0, num_steps=8, system_circuit_factory=fac,
                        sigma_blocks=sp_.sigma_blocks)
    b = get_circuit_structured(sch, z0, 2.0, 8, fac, sp_.sigma_blocks, 4, (nq, nq))
    sa = be.run(transpile(a, be)).result().get_statevector().data
    sb = be.run(transpile(b, be)).result().get_statevector().data
    assert np.abs(sa - sb).max() < 1e-10


def test_flag_pipeline_level0_matches_diagonalgate_1d():
    """Flag variant inside the full pipeline (1D sponge), transpiled at
    optimization_level=0, equals the DiagonalGate pipeline."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from lib import WaveEquationSponge1d
    op = WaveEquationSponge1d(4, npml=4)
    x = np.arange(16)
    z0 = np.concatenate([np.exp(-0.5 * ((x - 8) / 2.0) ** 2), np.zeros(16)]).astype(complex)
    z0 /= np.linalg.norm(z0)
    sch = Schrodingerisation(op.hamiltonian_parts(), num_qubits_p=5, p_max=8.0)
    fac = lambda dt: get_bell_evolution_circ_strang(op.wave_op, dt)
    sim = AerSimulator(method="statevector")
    a = sch.get_circuit(z0, 1.0, num_steps=4, system_circuit_factory=fac,
                        sigma_blocks=op.sigma_blocks)
    a.save_statevector()
    b = get_circuit_structured(sch, z0, 1.0, 4, fac, op.sigma_blocks, 2, (4,),
                               flag_ancilla=True)
    b.save_statevector()
    sa = np.asarray(sim.run(transpile(a, sim)).result().get_statevector())
    sb = np.asarray(sim.run(transpile(b, sim, optimization_level=0)).result().get_statevector())
    assert np.abs(sa - sb[:len(sa)]).max() < 1e-10
    assert np.linalg.norm(sb[len(sa):]) < 1e-10
