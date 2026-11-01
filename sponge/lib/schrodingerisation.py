"""Schrodingerisation of non-unitary linear dynamics (Jin-Liu-Yu).

Simulates d z/dt = A z with non-Hermitian A (e.g. a wave equation with an
absorbing PML layer) by a unitary evolution in one extra dimension, after
Jin, Liu & Yu (PRA 108, 032603 (2023), arXiv:2212.14703) and the
artificial-boundary-condition application Jin, Li, Liu & Yu (SIAM J. Sci.
Comput. 46 (2024), arXiv:2304.00667).

Recipe (sign conventions stated explicitly):

  A = H1 + i H2,  H1 = (A + A^dag)/2,  H2 = (A - A^dag)/(2i)  (both Hermitian)

  warped phase transformation w(t, p) = e^{-p} z(t) for p > 0, initialised
  with w(0, p) = g(p) z(0), g(p) = e^{-|p|} (or a C^1 smoothing, below):

      dw/dt = -H1 dw/dp + i H2 w .

  Fourier series in p with the *synthesis* convention w = sum_k w_hat_k
  e^{+i eta_k p}  (numpy: w = ifft(w_hat), w_hat = fft(w)) gives decoupled
  unitary evolutions per mode:

      d w_hat_k/dt = -i (eta_k H1 - H2) w_hat_k
      w_hat_k(T)   = exp(-i T (eta_k H1 - H2)) w_hat_k(0) .

  Recovery: z(T) = e^{p*} w(T, p*) for any p* >= max(0, lambda_max^+(H1)) T;
  when H1 <= 0 (true for the collapsed 1D PML and the sponge) any small
  p* > 0 works.

The kink of e^{-|p|} at p = 0 makes the p-discretisation only first order;
profile='cubic' replaces it on (-1, 0) by the C^1 cubic of Jin-Liu-Ma
(arXiv:2402.14696, Remark 4.9), restoring O(dp^2) while keeping
g(p) = e^{-p} exactly for p >= 0 (so the recovery formula is unchanged).
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply, eigsh

from .absorbing import split_generator


def warping_profile(p, kind='cubic'):
    """Initial profile g(p): e^{-|p|}, optionally C^1-smoothed on (-1, 0)."""
    p = np.asarray(p, dtype=float)
    g = np.exp(-np.abs(p))
    if kind == 'cubic':
        mask = (p > -1.0) & (p < 0.0)
        q = p[mask]
        e = np.e
        g[mask] = (-3 + 3 / e) * q**3 + (-5 + 4 / e) * q**2 - q + 1
    elif kind != 'exp':
        raise ValueError(f"unknown profile kind '{kind}'")
    return g


class Schrodingerisation:
    """Unitary simulation of dz/dt = A z via the warped phase transformation.

    Parameters
    ----------
    A : sparse or dense square matrix (the non-Hermitian generator), or a
        tuple (H1, H2) of the Hermitian split.
    num_qubits_p : p-register size; N_p = 2**num_qubits_p grid points.
    p_max : the p domain is [-p_max, p_max) (periodic, spectral in p).
    profile : 'cubic' (default, O(dp^2)) or 'exp' (plain e^{-|p|}, O(dp)).
    """

    def __init__(self, A, num_qubits_p, p_max, profile='cubic'):
        if isinstance(A, tuple):
            self.H1, self.H2 = (sp.csr_matrix(M) for M in A)
        else:
            self.H1, self.H2 = split_generator(A)
        self.num_qubits_p = num_qubits_p
        self.Np = 2**num_qubits_p
        self.p_max = float(p_max)
        self.dp = 2 * self.p_max / self.Np
        self.p = -self.p_max + self.dp * np.arange(self.Np)
        self.eta = 2 * np.pi * np.fft.fftfreq(self.Np, d=self.dp)
        self.profile = profile
        self.g = warping_profile(self.p, profile)

    # ------------------------------------------------------------------ #
    # classical (matrix-level) pipeline                                   #
    # ------------------------------------------------------------------ #

    def initial_w(self, z0):
        """w(0, p_j) = g(p_j) z0 as an (N_p, n_sys) array (a product state)."""
        return np.outer(self.g, np.asarray(z0, dtype=complex))

    def evolve(self, z0, T, dense_threshold=512):
        """Evolve w to time T exactly (per Fourier mode). Returns W(T)."""
        W = self.initial_w(z0)
        What = np.fft.fft(W, axis=0)
        n = self.H1.shape[0]
        if n <= dense_threshold:
            H1d = self.H1.toarray()
            H2d = self.H2.toarray()
            for k in range(self.Np):
                U = expm(-1j * T * (self.eta[k] * H1d - H2d))
                What[k] = U @ What[k]
        else:
            for k in range(self.Np):
                M = -1j * T * (self.eta[k] * self.H1 - self.H2)
                What[k] = expm_multiply(M.tocsc(), What[k])
        return np.fft.ifft(What, axis=0)

    def lambda_max_plus(self, k=4):
        """Largest eigenvalue of H1 (0 clipped from below); the recovery
        slice must satisfy p* >= lambda_max^+ * T."""
        n = self.H1.shape[0]
        if n <= 512:
            lam = np.linalg.eigvalsh(self.H1.toarray()).max()
        else:
            lam = eigsh(self.H1, k=k, which='LA',
                        return_eigenvectors=False).max()
        return max(0.0, float(lam))

    def _slice_index(self, p_star, num_offset):
        if p_star is None:
            j = int(np.searchsorted(self.p, 0.0)) + num_offset
        else:
            if not (self.p[0] <= p_star <= self.p[-1]):
                raise ValueError(
                    f"p_star={p_star:g} lies outside the p grid "
                    f"[{self.p[0]:g}, {self.p[-1]:g}]; increase p_max or "
                    f"reduce T so that lambda_max_plus()*T + margin fits")
            j = int(np.searchsorted(self.p, p_star))
        return min(max(j, 0), self.Np - 1)

    def recover(self, W, p_star=None, num_offset=3):
        """z(T) = e^{p*} w(T, p*). Default p*: num_offset grid points above 0.

        Returns (z, p_star_used). An explicit p_star outside the p grid
        raises (the slice would silently violate the validity threshold).

        Caution: the default small p* is only valid when H1 is negative
        semidefinite (lambda_max_plus() == 0); for indefinite H1 pass
        p_star >= lambda_max_plus() * T explicitly.
        """
        j = self._slice_index(p_star, num_offset)
        return np.exp(self.p[j]) * W[j], self.p[j]

    def recovery_plateau(self, W, z_ref=None, window=12):
        """Diagnostic: relative spread of e^{p_j} ||w(T,p_j)|| over the first
        `window` grid points right of p = 0. A flat plateau (small spread)
        indicates p_max, N_p and the profile are adequate."""
        j0 = int(np.searchsorted(self.p, 0.0)) + 1
        js = np.arange(j0, min(j0 + window, self.Np))
        vals = np.array([np.exp(self.p[j]) * np.linalg.norm(W[j]) for j in js])
        spread = np.ptp(vals) / vals.mean() if vals.mean() > 0 else np.inf
        out = {'p': self.p[js], 'norms': vals, 'spread': float(spread)}
        if z_ref is not None:
            errs = [np.linalg.norm(np.exp(self.p[j]) * W[j] - z_ref)
                    / np.linalg.norm(z_ref) for j in js]
            out['rel_errors'] = np.array(errs)
        return out

    # ------------------------------------------------------------------ #
    # quantum circuit pipeline                                            #
    # ------------------------------------------------------------------ #

    def total_hamiltonian(self, sparse=True):
        """H_tot = D_eta (x) H1 - I (x) H2 acting on (p-register, system),
        p-register most significant. The Schrodingerised evolution is
        w_hat(T) = exp(-i T H_tot) w_hat(0) in the eta (Fourier) basis."""
        Deta = sp.diags(self.eta)
        Isys = sp.identity(self.H1.shape[0], format='csr')
        Ip = sp.identity(self.Np, format='csr')
        Htot = sp.kron(Deta, self.H1) - sp.kron(Ip, self.H2)
        return Htot.tocsr() if sparse else Htot.toarray()

    def get_circuit(self, z0, T, num_steps=None, system_circuit_factory=None,
                    sigma_blocks=None, trotter_order=2):
        """Qiskit circuit implementing the Schrodingerised evolution.

        Layout: qubits [0 .. n_sys-1] = system register (little-endian),
        qubits [n_sys .. n_sys+n_p-1] = p register (most significant).

        Structure:
          initialize(g (x) z0)  ->  QFT^dag on p (p -> eta basis)
          -> num_steps Trotter steps of exp(-i dt H_tot)
          -> QFT on p (eta -> p basis).

        If `num_steps` is None a single exact HamiltonianGate of H_tot is
        used (statevector demo). Otherwise the evolution is Trotterised into
        damping factors exp(+i dt D_eta (x) Sigma) [diagonal phases] and
        wave factors exp(-i dt I (x) H), the latter supplied by
        `system_circuit_factory(dt)` (e.g. the library's
        get_bell_evolution_circ) and `sigma_blocks` being the diagonal of
        -H1 (requires H1 diagonal, i.e. the collapsed 1D PML or the
        sponge). trotter_order=2 (default) uses Strang splitting -- a
        half-damping factor on each side, consecutive halves merged -- with
        O(dt^2) error at the same circuit depth; trotter_order=1 is the
        plain first-order product.
        """
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import QFTGate, DiagonalGate, HamiltonianGate

        n_sys = int(np.log2(self.H1.shape[0]))
        if 2**n_sys != self.H1.shape[0]:
            raise ValueError("system dimension must be a power of two")
        n = n_sys + self.num_qubits_p
        sys_qubits = list(range(n_sys))
        p_qubits = list(range(n_sys, n))

        W0 = self.initial_w(np.asarray(z0, dtype=complex)).ravel()  # index = j_p * 2^n_sys + s
        norm0 = np.linalg.norm(W0)

        circ = QuantumCircuit(n)
        circ.initialize(W0 / norm0, circ.qubits)
        # p -> eta basis: amplitudes become fft(W)/sqrt(Np) = QFT^dag
        circ.append(QFTGate(self.num_qubits_p).inverse(), p_qubits)

        if num_steps is None:
            Htot = self.total_hamiltonian(sparse=False)
            circ.append(HamiltonianGate(Htot, T), circ.qubits)
        else:
            dt = T / num_steps
            if abs(sp.csr_matrix(self.H1) - sp.diags(self.H1.diagonal())).max() > 1e-12:
                raise ValueError("Trotter circuit requires diagonal H1; "
                                 "pass num_steps=None for general H1")
            d = -np.real(np.asarray(self.H1.diagonal()))
            if sigma_blocks is None:
                sigma_blocks = d
            elif not np.allclose(np.asarray(sigma_blocks), d, atol=1e-10):
                raise ValueError("sigma_blocks is inconsistent with -diag(H1)")
            if trotter_order not in (1, 2):
                raise ValueError("trotter_order must be 1 or 2")
            angles = dt * np.outer(self.eta, sigma_blocks).ravel()
            damp = DiagonalGate(np.exp(1j * angles))
            wave = QuantumCircuit(n)
            if system_circuit_factory is None:
                wave.append(HamiltonianGate((-self.H2).toarray(), dt), sys_qubits)
            else:
                wave.compose(system_circuit_factory(dt), qubits=sys_qubits, inplace=True)
            if trotter_order == 1:
                for _ in range(num_steps):
                    circ.compose(wave, inplace=True)
                    circ.append(damp, circ.qubits)
            else:
                # Strang: D(dt/2) [W(dt) D(dt)]^{n-1} W(dt) D(dt/2)
                damp_half = DiagonalGate(np.exp(0.5j * angles))
                circ.append(damp_half, circ.qubits)
                for k in range(num_steps):
                    circ.compose(wave, inplace=True)
                    circ.append(damp if k < num_steps - 1 else damp_half,
                                circ.qubits)

        # eta -> p basis
        circ.append(QFTGate(self.num_qubits_p), p_qubits)
        circ.metadata = {'norm0': norm0, 'n_sys': n_sys}
        return circ

    def recover_from_statevector(self, statevector, norm0, p_star=None,
                                 num_offset=3):
        """Recover z(T) from the final statevector of get_circuit().

        statevector index = j_p * 2^n_sys + s; slice at p* and rescale by
        e^{p*} * norm0 (undoing the initial normalisation).
        """
        sv = np.asarray(statevector)
        n_sys_dim = self.H1.shape[0]
        W = sv.reshape(self.Np, n_sys_dim)
        j = self._slice_index(p_star, num_offset)
        return np.exp(self.p[j]) * norm0 * W[j], self.p[j]
