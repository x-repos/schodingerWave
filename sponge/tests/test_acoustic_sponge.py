"""Acoustic sponge-layer tests: damping profiles, Schrodingerisation
(warped transform, Fourier-in-p evolution, recovery), and the explicit
circuits (projector-extended Bell evolution, Strang, gate-level sponge).

Vendored from qawave-cpml tests/test_absorbing.py @ a13b82b with the
CPML memory-variable tests removed. WaveEquationCPML1d appears only in its
1D collapsed form, which is exactly diagonal damping (a matched 1D
sponge): H1 = -diag(sigma) <= 0."""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply, eigs

from lib import (
    WaveEquationEvolution,
    WaveEquationCPML1d,
    WaveEquationSponge2d,
    WaveEquationSeparatedEvolution2d,
    Schrodingerisation,
    pml_sigma_profile,
    split_generator,
    get_bell_evolution_circ,
    get_bell_evolution_circ_strang,
)


def gaussian_bump_1d(N, center, width):
    x = np.arange(N)
    return np.exp(-0.5 * ((x - center) / width) ** 2)


def evolve_dense(A, z0, T):
    return expm(np.asarray(A.todense()) * T) @ z0


# --------------------------------------------------------------------- #
# profiles and generators                                                #
# --------------------------------------------------------------------- #

def test_sigma_profile():
    N, npml = 64, 10
    s = pml_sigma_profile(N, npml, m=2, R0=1e-3)
    assert np.all(s[npml:N - 1 - npml] == 0)          # interior clean
    assert s[0] == pytest.approx(s.max())             # max at the wall
    assert s[N - 1] == pytest.approx(s.max())
    sigma_max = 3 * np.log(1e3) / (2 * npml)
    assert s.max() == pytest.approx(sigma_max)
    # interface points are undamped
    assert s[npml] == 0 and s[N - 1 - npml] == 0


def test_1d_generator_spectrum():
    op = WaveEquationCPML1d(num_qubits_x=5, npml=8)
    A = np.asarray(op.generator().todense())
    lam = np.linalg.eigvals(A)
    assert lam.real.max() < 1e-10


def test_1d_hermitian_split():
    op = WaveEquationCPML1d(num_qubits_x=4, npml=5)
    H1, H2 = op.hamiltonian_parts()
    A = op.generator()
    # A = H1 + iH2 with both Hermitian
    assert abs((H1 - H1.conj().T)).max() < 1e-13
    assert abs((H2 - H2.conj().T)).max() < 1e-13
    assert abs((H1 + 1j * H2 - A)).max() < 1e-13
    # collapsed 1D PML: H1 = -diag(sigma) <= 0
    assert np.linalg.eigvalsh(np.asarray(H1.todense())).max() <= 1e-14


# --------------------------------------------------------------------- #
# the convolution (memory) form collapses to plain damping in 1D         #
# --------------------------------------------------------------------- #

# --------------------------------------------------------------------- #
# absorption: outgoing waves leave, hard walls reflect                   #
# --------------------------------------------------------------------- #

def test_1d_absorption_vs_hard_wall():
    nx, npml = 6, 12
    N = 2**nx
    T = 80.0  # both pulses reach a layer and would have bounced back
    pml = WaveEquationCPML1d(num_qubits_x=nx, npml=npml)
    v0 = gaussian_bump_1d(N, N // 2, 3.0)
    z0 = np.concatenate([v0, np.zeros(N)]).astype(complex)
    z0 /= np.linalg.norm(z0)

    out_pml = evolve_dense(pml.generator(), z0, T)
    H = np.asarray(pml.hamiltonian(sparse=False))
    out_wall = expm(-1j * H * T) @ z0

    interior = slice(npml, N - npml)
    def interior_energy(z):
        return np.linalg.norm(z[:N][interior])**2 + np.linalg.norm(z[N:][interior])**2

    e_pml = interior_energy(out_pml)
    e_wall = interior_energy(out_wall)
    # hard wall keeps all the energy in the domain; PML absorbs it
    assert e_wall > 0.5
    assert e_pml < 1e-5
    # round-trip amplitude reflection ~ sqrt(energy ratio) near design R0
    assert np.sqrt(e_pml / e_wall) < 5e-3


def test_2d_sponge_absorption():
    nx = ny = 4
    op = WaveEquationSponge2d(nx, ny, npml=4)
    H1, _ = op.hamiltonian_parts()
    # provably dissipative: H1 diagonal and <= 0
    assert abs(H1 - sp.diags(H1.diagonal())).max() < 1e-14
    assert H1.diagonal().real.max() <= 1e-14

    v0 = np.outer(gaussian_bump_1d(2**nx, 2**(nx - 1), 1.5),
                  gaussian_bump_1d(2**ny, 2**(ny - 1), 1.5))
    z0 = op.initial_state(v0)
    out = expm_multiply(op.generator().tocsc() * 40.0, z0)
    assert np.linalg.norm(out)**2 < 0.10


# --------------------------------------------------------------------- #
# Schrodingerisation                                                     #
# --------------------------------------------------------------------- #

def test_schrodingerisation_scalar_decay():
    lam = -0.5
    T = 2.0
    A = sp.csr_matrix(np.array([[lam]], dtype=complex))
    sch = Schrodingerisation(A, num_qubits_p=9, p_max=12.0)
    W = sch.evolve(np.array([1.0 + 0j]), T)
    z, p_star = sch.recover(W)
    assert p_star > 0
    assert abs(z[0] - np.exp(lam * T)) < 1e-4


def test_schrodingerisation_profile_order():
    # cubic profile beats the plain e^{-|p|} kink at equal resolution
    lam, T = -0.7, 1.5
    A = sp.csr_matrix(np.array([[lam]], dtype=complex))
    errs = {}
    for kind in ('exp', 'cubic'):
        sch = Schrodingerisation(A, num_qubits_p=8, p_max=10.0, profile=kind)
        W = sch.evolve(np.array([1.0 + 0j]), T)
        z, _ = sch.recover(W)
        errs[kind] = abs(z[0] - np.exp(lam * T))
    assert errs['cubic'] < errs['exp']


def test_schrodingerisation_1d_wave_pml():
    nx, npml, T = 5, 8, 30.0
    N = 2**nx
    pml = WaveEquationCPML1d(num_qubits_x=nx, npml=npml)
    A = pml.generator()
    v0 = gaussian_bump_1d(N, N // 2, 2.5)
    z0 = np.concatenate([v0, np.zeros(N)]).astype(complex)
    z0 /= np.linalg.norm(z0)
    z_ref = evolve_dense(A, z0, T)

    sch = Schrodingerisation(pml.hamiltonian_parts(), num_qubits_p=10, p_max=18.0)
    assert sch.lambda_max_plus() == 0.0
    W = sch.evolve(z0, T)
    z, _ = sch.recover(W)
    err = np.linalg.norm(z - z_ref) / np.linalg.norm(z0)
    assert err < 2e-3
    # plateau diagnostic: flat over the recovery window
    diag = sch.recovery_plateau(W, z_ref=z_ref)
    assert diag['spread'] < 5e-2


def test_schrodingerisation_2d_sponge():
    # moderate sigma_max: the strongly damped components travel a distance
    # ~ sigma * T leftwards in p and must not wrap around the periodic
    # p-domain into the recovery region
    nx = ny = 3
    op = WaveEquationSponge2d(nx, ny, npml=2, sigma_max=1.0)
    z0 = op.initial_state(np.outer(gaussian_bump_1d(8, 4, 1.0),
                                   gaussian_bump_1d(8, 4, 1.0)))
    T = 10.0
    z_ref = expm_multiply(op.generator().tocsc() * T, z0)
    sch = Schrodingerisation(op.hamiltonian_parts(), num_qubits_p=10, p_max=16.0)
    assert sch.lambda_max_plus() == 0.0  # sponge H1 <= 0: any p* > 0 valid
    W = sch.evolve(z0, T)
    z, _ = sch.recover(W)
    assert np.linalg.norm(z - z_ref) / np.linalg.norm(z0) < 1e-3


# --------------------------------------------------------------------- #
# quantum circuit                                                        #
# --------------------------------------------------------------------- #

@pytest.fixture(scope='module')
def backend():
    from qiskit_aer import StatevectorSimulator
    return StatevectorSimulator()


def run_statevector(circ, backend):
    from qiskit import transpile
    return backend.run(transpile(circ, backend)).result().get_statevector().data


def test_circuit_matches_numpy_exact(backend):
    """Exact HamiltonianGate circuit reproduces the numpy per-mode result
    (validates QFT/fft conventions and qubit ordering)."""
    nx, npml, T = 3, 2, 4.0
    N = 2**nx
    pml = WaveEquationCPML1d(num_qubits_x=nx, npml=npml)
    v0 = gaussian_bump_1d(N, N // 2, 1.0)
    z0 = np.concatenate([v0, np.zeros(N)]).astype(complex)
    z0 /= np.linalg.norm(z0)

    sch = Schrodingerisation(pml.hamiltonian_parts(), num_qubits_p=5, p_max=8.0)
    W_np = sch.evolve(z0, T)

    circ = sch.get_circuit(z0, T)  # exact HamiltonianGate path
    sv = run_statevector(circ, backend)
    z_circ, _ = sch.recover_from_statevector(sv, circ.metadata['norm0'])
    z_np, _ = sch.recover(W_np)
    assert np.linalg.norm(z_circ - z_np) < 1e-8


def test_circuit_trotter_with_bell_evolution(backend):
    """Trotterised circuit using the library's own Bell-evolution wave
    circuit for exp(-i dt H) plus diagonal damping phases."""
    nx, npml, T, steps = 3, 2, 4.0, 40
    N = 2**nx
    # moderate sigma_max: the Trotter splitting error grows with
    # ||[D_eta x Sigma, I x H]|| ~ eta_max * sigma_max * ||H||
    pml = WaveEquationCPML1d(num_qubits_x=nx, npml=npml, sigma_max=1.0)
    v0 = gaussian_bump_1d(N, N // 2, 1.0)
    z0 = np.concatenate([v0, np.zeros(N)]).astype(complex)
    z0 /= np.linalg.norm(z0)

    z_ref = evolve_dense(pml.generator(), z0, T)

    sch = Schrodingerisation(pml.hamiltonian_parts(), num_qubits_p=5, p_max=8.0)
    factory = lambda dt: get_bell_evolution_circ(pml.wave_op, dt)
    circ = sch.get_circuit(z0, T, num_steps=steps, system_circuit_factory=factory,
                           sigma_blocks=pml.sigma_blocks)
    sv = run_statevector(circ, backend)
    z_circ, _ = sch.recover_from_statevector(sv, circ.metadata['norm0'])
    err = np.linalg.norm(z_circ - z_ref) / np.linalg.norm(z0)
    assert err < 5e-2, f"Trotter circuit error {err}"


# --------------------------------------------------------------------- #
# gate-level 2D: separated op strings + projector-aware Bell circuit      #
# --------------------------------------------------------------------- #

def _phase_aligned_error(U_circ, U_ref):
    ph = (U_ref.conj() * U_circ).sum()
    ph /= abs(ph)
    return np.abs(U_circ / ph - U_ref).max()


def test_separated_2d_op_matches_block_hamiltonian():
    for nx, ny in ((2, 2), (2, 3), (3, 2)):
        sep = WaveEquationSeparatedEvolution2d(nx, ny)
        H_op = np.asarray(sep.to_matrix(sparse=False))
        H_ref = np.asarray(WaveEquationSponge2d(nx, ny, npml=1).hamiltonian().todense())
        assert np.abs(H_op - H_ref).max() < 1e-13
        assert np.abs(H_op - H_op.conj().T).max() < 1e-13


def test_bell_circ_supports_projector_strings():
    """The extended circuit builder handles 'z' (block projector) characters;
    error must be pure first-order Trotter error ~ 0.5 dt^2."""
    from qiskit.quantum_info import Operator
    sep = WaveEquationSeparatedEvolution2d(2, 2)
    H = np.asarray(sep.to_matrix(sparse=False))
    errs = [_phase_aligned_error(Operator(get_bell_evolution_circ(sep, dt)).data,
                                 expm(-1j * H * dt))
            for dt in (0.2, 0.1)]
    assert errs[0] < 0.025
    assert errs[0] / errs[1] > 3.0  # ~ O(dt^2) per step


def test_strang_wave_circuit_higher_order():
    from qiskit.quantum_info import Operator
    sep = WaveEquationSeparatedEvolution2d(2, 2)
    H = np.asarray(sep.to_matrix(sparse=False))
    errs = [_phase_aligned_error(Operator(get_bell_evolution_circ_strang(sep, dt)).data,
                                 expm(-1j * H * dt))
            for dt in (0.2, 0.1)]
    assert errs[0] / errs[1] > 6.0  # ~ O(dt^3) per step


def test_2d_sponge_gate_level(backend):
    """Full gate-level absorbing 2D run: 17 qubits, Strang-in-Strang
    Trotterisation, recovery vs the exact non-unitary evolution."""
    from qiskit import transpile
    nq, T, n_p, p_max, steps = 3, 10.0, 9, 16.0, 100
    sponge = WaveEquationSponge2d(nq, nq, npml=2, sigma_max=1.0)
    sep = WaveEquationSeparatedEvolution2d(nq, nq)
    bump = gaussian_bump_1d(8, 4, 1.0)
    z0 = sponge.initial_state(np.outer(bump, bump))
    z_ref = expm_multiply(sponge.generator().tocsc() * T, z0)

    sch = Schrodingerisation(sponge.hamiltonian_parts(), num_qubits_p=n_p, p_max=p_max)
    factory = lambda dt: get_bell_evolution_circ_strang(sep, dt)
    circ = sch.get_circuit(z0, T, num_steps=steps, system_circuit_factory=factory,
                           sigma_blocks=sponge.sigma_blocks)
    sv = backend.run(transpile(circ, backend)).result().get_statevector().data
    z_gate, _ = sch.recover_from_statevector(sv, circ.metadata['norm0'])
    err = np.linalg.norm(z_gate - z_ref) / np.linalg.norm(z_ref)
    assert err < 3e-2, f"gate-level 2D recovery error {err}"


# --------------------------------------------------------------------- #
# guards                                                                  #
# --------------------------------------------------------------------- #

def test_recover_rejects_out_of_range_p_star():
    A = sp.csr_matrix(np.array([[-0.5]], dtype=complex))
    sch = Schrodingerisation(A, num_qubits_p=6, p_max=8.0)
    W = sch.evolve(np.array([1.0 + 0j]), 1.0)
    with pytest.raises(ValueError):
        sch.recover(W, p_star=50.0)
    with pytest.raises(ValueError):
        sch.recover(W, p_star=-100.0)
    with pytest.raises(ValueError):
        sch.recover_from_statevector(W.ravel(), 1.0, p_star=50.0)


def test_profile_rejects_degenerate_layers():
    with pytest.raises(ValueError):
        pml_sigma_profile(16, 0)
    with pytest.raises(ValueError):
        pml_sigma_profile(16, 8)  # no interior left


def test_circuit_rejects_inconsistent_sigma_blocks():
    pml = WaveEquationCPML1d(num_qubits_x=3, npml=2)
    sch = Schrodingerisation(pml.hamiltonian_parts(), num_qubits_p=4, p_max=6.0)
    z0 = np.zeros(16, dtype=complex)
    z0[4] = 1.0
    with pytest.raises(ValueError):
        sch.get_circuit(z0, 1.0, num_steps=4,
                        sigma_blocks=pml.sigma_blocks + 1.0)


# --------------------------------------------------------------------- #
# gold standard: truncated domain vs large-domain reference              #
# --------------------------------------------------------------------- #

def test_1d_gold_standard_reflection():
    """The true PML error: compare the CPML solution on a small domain with
    the same initial data evolved in a 4x larger hard-wall domain (before
    any wave returns from the far walls), restricted to the small interior."""
    nx, npml = 6, 12
    N = 2**nx
    nx_big = 8
    N_big = 2**nx_big
    off = (N_big - N) // 2  # embed the small domain centred in the big one

    v0 = gaussian_bump_1d(N, N // 2, 3.0)
    z0 = np.concatenate([v0, np.zeros(N)]).astype(complex)
    nrm = np.linalg.norm(z0)
    z0 /= nrm

    v0_big = np.zeros(N_big)
    v0_big[off:off + N] = v0
    z0_big = np.concatenate([v0_big, np.zeros(N_big)]).astype(complex) / nrm

    pml = WaveEquationCPML1d(num_qubits_x=nx, npml=npml)
    A = np.asarray(pml.generator().todense())
    H_big = WaveEquationEvolution(nx_big, dim=1, diff_type='forward').to_matrix(sparse=False)

    win = slice(npml, N - npml)
    for T in (20.0, 60.0):  # transition error, then round-trip reflection
        out = expm(A * T) @ z0
        ref = expm(-1j * H_big * T) @ z0_big
        err = np.sqrt(
            np.linalg.norm(out[:N][win] - ref[:N_big][off:off + N][win])**2
            + np.linalg.norm(out[N:][win] - ref[N_big:][off:off + N][win])**2)
        assert err < 1e-3, f"true PML error {err} at T={T}"


def test_discrete_calibration_flat_at_design():
    """With calibration='discrete' (joint pair rescale) the realised
    reflection equals the design R0 at every layer width; per-profile
    rescaling would instead create an in-layer impedance mismatch."""
    nx, nx_big = 6, 8
    N, N_big = 2**nx, 2**nx_big
    off = (N_big - N) // 2
    v0 = gaussian_bump_1d(N, N // 2, 3.0)
    z0 = np.concatenate([v0, np.zeros(N)]).astype(complex)
    nrm = np.linalg.norm(z0)
    z0 /= nrm
    v0b = np.zeros(N_big)
    v0b[off:off + N] = v0
    z0b = np.concatenate([v0b, np.zeros(N_big)]).astype(complex) / nrm
    H_big = WaveEquationEvolution(nx_big, dim=1, diff_type='forward').to_matrix(sparse=False)
    ref = expm(-1j * H_big * 60.0) @ z0b

    for npml in (4, 16):
        pml = WaveEquationCPML1d(num_qubits_x=nx, npml=npml,
                                 calibration='discrete')
        # characteristic sum equals the design integral exactly
        target = 2 * (-np.log(1e-3) / 2)
        assert 0.5 * (pml.sigma_v + pml.sigma_w).sum() == pytest.approx(target)
        out = expm(np.asarray(pml.generator().todense()) * 60.0) @ z0
        win = slice(npml, N - npml)
        err = np.sqrt(
            np.linalg.norm(out[:N][win] - ref[:N_big][off:off + N][win])**2
            + np.linalg.norm(out[N:][win] - ref[N_big:][off:off + N][win])**2)
        assert 0.6e-3 < err < 1.3e-3, f"npml={npml}: {err} not at design R0"


# --------------------------------------------------------------------- #
# regression: the existing library is untouched                          #
# --------------------------------------------------------------------- #

def test_wave_evolution_regression():
    op = WaveEquationEvolution(num_qubits_x=4, dim=1, diff_type='forward')
    H = op.to_matrix(sparse=False)
    assert np.abs(H - H.conj().T).max() < 1e-13
    half = 2**(op.num_qubits - 1)
    D = H[:half, half:]
    L = -D @ D.conj().T
    # mixed-BC Laplacian: tridiag(1,-2,1) with Neumann at the right edge
    assert L[0, 0] == pytest.approx(-2.0)
    assert L[-1, -1] == pytest.approx(-1.0)
    assert L[3, 2] == pytest.approx(1.0) and L[3, 4] == pytest.approx(1.0)
