"""2D isotropic elastic wave equation as Hamiltonian simulation.

Velocity-stress first-order form, symmetrised so that the semi-discrete
generator is exactly anti-Hermitian (the elastic analogue of the acoustic
encoding of Sato et al., PRR 6, 033246; cf. Jin & Zhang,
arXiv:2505.18711 for the Schrodingerised elastic framework):

    rho dv/dt = Div sigma,      d sigma/dt = C eps(v),

with scaled fields  v~ = sqrt(rho) v  and  s = C^{-1/2} sigma  (Voigt
vector (s_xx, s_yy, sqrt(2) s_xy) so the elastic energy
(rho|v|^2 + sigma : C^{-1} : sigma)/2 = (|v~|^2 + |s|^2)/2 is Euclidean):

    d s/dt  =  G v~,        d v~/dt = -G^dag s,
    G = C^{1/2} E / sqrt(rho),
    E = [[Dx, 0], [0, Dy], [Dy/sqrt2, Dx/sqrt2]],

where Dx, Dy are the forward-difference operators and the adjoint pair
uses D_f^dag = -D_b exactly as in the acoustic library. The full state
psi = [v~x; v~y; s_xx; s_yy; s_xy; 0; 0; 0] occupies 8 blocks (three
block-label qubits) of size 2^(nx+ny); i d psi/dt = H psi with
H = i A_0 Hermitian.

Material parameters enter through the P/S speeds: with rho = 1 (lattice
units), mu = cs^2 and lambda = cp^2 - 2 cs^2 (requires cp >= sqrt(2) cs
for lambda >= 0). C^{1/2} is closed-form for the isotropic Voigt matrix
C~ = [[lam+2mu, lam, 0], [lam, lam+2mu, 0], [0, 0, 2mu]].
"""

import numpy as np
import scipy.sparse as sp

from .op_library import DifferentialOperator1d

NUM_BLOCKS = 8  # 5 physical + 3 padding


def isotropic_voigt_sqrt(cp, cs, rho=1.0):
    """Closed-form C~^{1/2} for the sqrt(2)-scaled Voigt convention."""
    mu = rho * cs**2
    lam = rho * cp**2 - 2 * mu
    if lam < -1e-12:
        raise ValueError(f"cp={cp}, cs={cs} give lambda={lam} < 0; "
                         "need cp >= sqrt(2) cs")
    s1 = np.sqrt(2 * (lam + mu))   # eigenvector (1,1)/sqrt2 of the 2x2 block
    s2 = np.sqrt(2 * mu)           # eigenvector (1,-1)/sqrt2
    C12 = np.array([
        [(s1 + s2) / 2, (s1 - s2) / 2, 0.0],
        [(s1 - s2) / 2, (s1 + s2) / 2, 0.0],
        [0.0, 0.0, np.sqrt(2 * mu)],
    ])
    return C12


class ElasticWaveEvolution2d:
    """Undamped (Hermitian) 2D isotropic elastic wave operator.

    State layout: 8 blocks [v~x, v~y, s_xx, s_yy, s_xy, 0, 0, 0] of size
    2^(nx+ny) each (x register more significant, as in the acoustic
    library); num_qubits = nx + ny + 3.
    """

    def __init__(self, num_qubits_x, num_qubits_y=None, cp=1.0, cs=None,
                 rho=1.0, h=1.0, diff_type='forward', periodic=False):
        if diff_type not in ('forward', 'backward'):
            raise NotImplementedError("use 'forward' or 'backward'")
        if num_qubits_y is None:
            num_qubits_y = num_qubits_x
        if cs is None:
            cs = cp / np.sqrt(3.0)   # Poisson solid-ish default
        self.num_qubits_x = num_qubits_x
        self.num_qubits_y = num_qubits_y
        self.num_qubits = num_qubits_x + num_qubits_y + 3
        self.cp, self.cs, self.rho, self.h = cp, cs, rho, h
        Nx, Ny = 2**num_qubits_x, 2**num_qubits_y

        d1x = DifferentialOperator1d(num_qubits_x, diff_type=diff_type,
                                     h=h, periodic=periodic)
        d1y = DifferentialOperator1d(num_qubits_y, diff_type=diff_type,
                                     h=h, periodic=periodic)
        D1x = sp.csr_matrix(d1x.to_matrix(sparse=True))
        D1y = sp.csr_matrix(d1y.to_matrix(sparse=True))
        Ix = sp.identity(Nx, format='csr')
        Iy = sp.identity(Ny, format='csr')
        self.Dx = sp.kron(D1x, Iy, format='csr')
        self.Dy = sp.kron(Ix, D1y, format='csr')
        self.C12 = isotropic_voigt_sqrt(cp, cs, rho)

    @property
    def block_size(self):
        return 2**(self.num_qubits_x + self.num_qubits_y)

    def _G_blocks(self):
        """G = C^{1/2} E / sqrt(rho) as a 3x2 array of sparse blocks
        (rows: s_xx, s_yy, s_xy; columns: v~x, v~y).

        E uses the CLASSICAL STAGGERED pairing: backward differences for
        the diagonal strains, forward for the shear. This is the unique
        choice with a consistent staggered geometry [vx at (i+1/2, j),
        vy at (i, j+1/2), s_xx/s_yy at (i, j), s_xy at (i+1/2, j+1/2)];
        the adjoint -G^dag then reproduces the textbook staggered
        velocity updates. (The all-forward alternative is Hermitian but
        geometrically inconsistent -- the position constraints have no
        solution -- and measurably degrades the PML.)"""
        r = 1.0 / np.sqrt(self.rho)
        s2 = 1.0 / np.sqrt(2.0)
        Dxb = (-self.Dx.T).tocsr()
        Dyb = (-self.Dy.T).tocsr()
        E = [[Dxb, None],
             [None, Dyb],
             [s2 * self.Dy, s2 * self.Dx]]
        C = self.C12
        G = [[None, None], [None, None], [None, None]]
        for a in range(3):
            for b in range(2):
                acc = None
                for k in range(3):
                    if C[a, k] == 0.0 or E[k][b] is None:
                        continue
                    term = (r * C[a, k]) * E[k][b]
                    acc = term if acc is None else acc + term
                G[a][b] = acc
        return G

    def generator_undamped(self):
        """A_0 (real, exactly antisymmetric): d psi/dt = A_0 psi."""
        M = self.block_size
        Z = sp.csr_matrix((M, M))
        G = self._G_blocks()
        blocks = [[None] * 8 for _ in range(8)]
        # v rows: dv~/dt = -G^T s   (blocks 0,1 <- 2,3,4)
        for b in range(2):
            for a in range(3):
                if G[a][b] is not None:
                    blocks[b][2 + a] = -G[a][b].T.tocsr()
        # s rows: ds/dt = G v~      (blocks 2,3,4 <- 0,1)
        for a in range(3):
            for b in range(2):
                if G[a][b] is not None:
                    blocks[2 + a][b] = G[a][b]
        for j in (5, 6, 7):
            blocks[j][j] = Z
        return sp.bmat(blocks, format='csr')

    def to_matrix(self, sparse=True):
        """Hermitian H = i A_0, so that i d psi/dt = H psi."""
        H = 1j * self.generator_undamped()
        return H if sparse else H.toarray()

    def initial_state(self, vx0, vy0=None, normalize=True):
        """Embed initial velocities (x-major flattened) into the 8-block
        state; stresses zero."""
        M = self.block_size
        out = np.zeros(8 * M, dtype=complex)
        out[:M] = np.asarray(vx0, dtype=complex).ravel() * np.sqrt(self.rho)
        if vy0 is not None:
            out[M:2 * M] = np.asarray(vy0, dtype=complex).ravel() * np.sqrt(self.rho)
        if normalize:
            out /= np.linalg.norm(out)
        return out

    def symbol_matrix(self, kx, ky):
        """Fourier symbol M(k) of G^dag G on a periodic grid: the discrete
        squared frequencies are its eigenvalues (exact for periodic
        constant-coefficient operators). Forward-difference symbol
        d(k) = (e^{ikh}-1)/h."""
        dxf = (np.exp(1j * kx * self.h) - 1.0) / self.h
        dyf = (np.exp(1j * ky * self.h) - 1.0) / self.h
        dxb = (1.0 - np.exp(-1j * kx * self.h)) / self.h
        dyb = (1.0 - np.exp(-1j * ky * self.h)) / self.h
        s2 = 1.0 / np.sqrt(2.0)
        E = np.array([[dxb, 0.0], [0.0, dyb], [s2 * dyf, s2 * dxf]])
        Gk = (self.C12 / np.sqrt(self.rho)) @ E
        return Gk.conj().T @ Gk
