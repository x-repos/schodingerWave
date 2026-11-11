"""Convolution absorbing boundary conditions (CPML) for the wave equation.

This module extends the Hamiltonian-simulation framework of Sato et al.,
PRR 6, 033246 (2024) with absorbing layers of the convolution (CPML /
recursive-convolution) type, following Roden & Gedney (2000) and
Komatitsch & Martin (2007).

The undamped solver evolves i d(psi)/dt = H psi with

    psi = [v; w],   v = du/dt,   w = i c * (difference of u),
    H = [[0, D], [D^dag, 0]],

so the absorbing layer is added at the level of the non-unitary generator

    d(psi)/dt = A psi,    A = -i H - (damping / convolution-memory terms).

The complex-frequency-shifted (CFS) coordinate stretch

    s_x(omega) = kappa_x + sigma_x / (alpha_x + i omega)

turns every spatial derivative into  (1/kappa_x) d_x + (zeta_x * d_x)  with
the causal exponential kernel

    zeta_x(t) = -(sigma_x/kappa_x^2) exp(-(sigma_x/kappa_x + alpha_x) t),

and each convolution is carried by one auxiliary "memory" field obeying

    d(psi_mem)/dt = -(sigma/kappa + alpha) psi_mem - (sigma/kappa^2) d_x f.

In 1D with kappa=1, alpha=0 the convolution collapses exactly to plain
block-diagonal damping (WaveEquationCPML1d); the general memory-variable
form is WaveEquationCFSPML1d (2 extra block qubits) and WaveEquationCPML2d
(3 extra block qubits, 7 active blocks of 8). WaveEquationSponge2d is the
simpler non-matched absorber whose Hermitian part is diagonal and provably
negative semidefinite (the clean case for Schrodingerisation).

All generators are returned as scipy sparse matrices in the same state
encoding / qubit ordering as the rest of the library (block label qubits
first / most significant, x register before y register).
"""

import numpy as np
import scipy.sparse as sp

from .op_library import OperatorList, WaveEquationEvolution, DifferentialOperator1d


def pml_sigma_profile(num_points, npml, h=1.0, m=2, R0=1e-3, c=1.0,
                      sigma_max=None, left=True, right=True, stagger=0.0,
                      calibration='continuum'):
    """Polynomially graded PML damping profile sampled at x_j = (j + stagger) h.

    sigma(x) = sigma_max * (depth/L)^m inside the layers of width L = npml*h,
    zero in the interior. The default amplitude follows Collino & Tsogka:

        sigma_max = -(m+1) c ln(R0) / (2 L),

    giving a normal-incidence round-trip reflection of about R0.

    calibration='continuum' (default) uses the sampled values as-is; the
    discrete sum then overestimates the design integral by a factor
    1 + O(1/npml) (node sampling of the convex profile), so thin layers
    over-damp and the realised round-trip reflection is
    R0^(sum sigma h / int sigma) < R0, converging up to R0 as the layer
    widens. calibration='discrete' rescales THIS profile so its own
    layer sum matches the continuum integral; for staggered field pairs
    use calibrate_sigma_pair instead -- per-profile rescaling gives the
    two fields different damping functions (the node and midpoint
    quadratures differ), an in-layer impedance mismatch that reflects
    at O(1/npml).
    """
    if npml < 1:
        raise ValueError("npml must be >= 1")
    if 2 * npml >= num_points:
        raise ValueError(f"layers of {npml} points each leave no interior "
                         f"on a {num_points}-point grid")
    if calibration not in ('continuum', 'discrete'):
        raise ValueError("calibration must be 'continuum' or 'discrete'")
    L = npml * h
    if sigma_max is None:
        sigma_max = -(m + 1) * c * np.log(R0) / (2 * L)
    x = (np.arange(num_points) + stagger) * h
    xl = npml * h
    xr = (num_points - 1 - npml) * h

    def layer(depth):
        s = sigma_max * np.clip(depth, 0.0, 1.0)**m
        if calibration == 'discrete' and s.any():
            s *= (sigma_max * L / (m + 1)) / (s.sum() * h)
        return s

    sigma = np.zeros(num_points)
    if left:
        sigma = np.maximum(sigma, layer((xl - x) / L))
    if right:
        sigma = np.maximum(sigma, layer((x - xr) / L))
    return sigma


def calibrate_sigma_pair(sigma_a, sigma_b, npml, h=1.0, m=2, R0=1e-3,
                         c=1.0, sigma_max=None, num_sides=2):
    """Jointly rescale a staggered profile pair (e.g. sigma_v, sigma_w) by a
    single common factor so that the characteristic damping sum
    sum_j (sigma_a + sigma_b)/2 * h equals the continuum design integral
    num_sides * sigma_max L/(m+1) exactly. A common factor preserves the
    perfect matching between the two fields (unlike per-profile
    calibration) while making the realised round-trip reflection equal
    the design R0 at every npml."""
    L = npml * h
    if sigma_max is None:
        sigma_max = -(m + 1) * c * np.log(R0) / (2 * L)
    target = num_sides * sigma_max * L / (m + 1)
    actual = 0.5 * (sigma_a + sigma_b).sum() * h
    f = target / actual if actual > 0 else 1.0
    return sigma_a * f, sigma_b * f


def split_generator(A):
    """Hermitian split A = H1 + i H2 with H1, H2 Hermitian.

    H1 = (A + A^dag)/2 is the dissipative part, H2 = (A - A^dag)/(2i) the
    Hamiltonian part. Accepts and returns scipy sparse matrices.
    """
    A = sp.csr_matrix(A)
    Ad = A.conj().T.tocsr()
    H1 = 0.5 * (A + Ad)
    H2 = (A - Ad) / 2j
    return H1, H2


def _extract_D(wave_op):
    """Top-right block D of H = [[0, D],[D^dag, 0]] from a WaveEquationEvolution."""
    H = sp.csr_matrix(wave_op.to_matrix(sparse=True))
    half = 2**(wave_op.num_qubits - 1)
    return H[:half, half:].tocsr()


class WaveEquationCPML1d:
    """1D wave equation with an exact (collapsed) convolution PML.

    With kappa = 1, alpha = 0 the CPML memory convolution collapses exactly
    (for data supported outside the layer) to block-diagonal damping:

        dv/dt = -i D w - sigma_v(x) v
        dw/dt = -i D^dag v - sigma_w(x) w

    i.e. A = -i H - diag(sigma_v, sigma_w). The state layout is identical to
    WaveEquationEvolution(num_qubits_x, dim=1): num_qubits_x + 1 qubits,
    v in the first half of the statevector, w in the second half.

    sigma_v is sampled at the integer nodes x_j; sigma_w at the staggered
    points where the w field effectively lives (x_{j-1/2} for the forward
    difference type, x_{j+1/2} for backward).

    The Hermitian split is H1 = -diag(sigma) <= 0 (guaranteed) and
    H2 = -H, which makes the Schrodingerised recovery unconditionally valid.
    """

    def __init__(self, num_qubits_x, npml=10, h=1.0, m=2, R0=1e-3,
                 sigma_max=None, diff_type='forward', left=True, right=True,
                 calibration='continuum'):
        if diff_type not in ('forward', 'backward'):
            raise NotImplementedError(
                "central differences checkerboard-decouple on non-periodic "
                "grids; use 'forward' or 'backward' (cf. PRR 6, 033246, Sec. III)")
        self.wave_op = WaveEquationEvolution(
            num_qubits_x, dim=1, diff_type=diff_type, h=h, periodic=False)
        self.num_qubits = self.wave_op.num_qubits
        self._h = h
        N = 2**num_qubits_x
        stagger_w = -0.5 if diff_type == 'forward' else 0.5
        kw = dict(h=h, m=m, R0=R0, sigma_max=sigma_max, left=left, right=right)
        self.sigma_v = pml_sigma_profile(N, npml, stagger=0.0, **kw)
        self.sigma_w = pml_sigma_profile(N, npml, stagger=stagger_w, **kw)
        if calibration == 'discrete':
            self.sigma_v, self.sigma_w = calibrate_sigma_pair(
                self.sigma_v, self.sigma_w, npml, h=h, m=m, R0=R0,
                sigma_max=sigma_max, num_sides=int(left) + int(right))
        elif calibration != 'continuum':
            raise ValueError("calibration must be 'continuum' or 'discrete'")
        self.sigma_blocks = np.concatenate([self.sigma_v, self.sigma_w])

    def hamiltonian(self, sparse=True):
        return self.wave_op.to_matrix(sparse=sparse)

    def generator(self):
        """Sparse non-Hermitian generator A = -iH - Sigma."""
        H = sp.csr_matrix(self.wave_op.to_matrix(sparse=True))
        return (-1j * H - sp.diags(self.sigma_blocks)).tocsr()

    def hamiltonian_parts(self):
        """(H1, H2) with A = H1 + iH2: H1 = -diag(sigma) <= 0, H2 = -H."""
        H = sp.csr_matrix(self.wave_op.to_matrix(sparse=True))
        return -sp.diags(self.sigma_blocks).tocsr(), (-H).tocsr()


class WaveEquationCFSPML1d:
    """1D wave equation with the general CFS convolution PML (memory form).

    The full convolution (recursive-convolution / CPML) formulation with
    per-field memory variables, supporting kappa >= 1 and alpha >= 0:

        dv/dt      = -i (1/kappa_v) D w + gamma * psi_v
        dw/dt      = -i (1/kappa_w) D^dag v + gamma * psi_w
        d(psi_v)/dt = -(sigma_v/kappa_v + alpha_v) psi_v
                      - (1/gamma)(sigma_v/kappa_v^2)(-i D) w
        d(psi_w)/dt = -(sigma_w/kappa_w + alpha_w) psi_w
                      - (1/gamma)(sigma_w/kappa_w^2)(-i D^dag) v

    State: 4 blocks [v; w; psi_v; psi_w] -> num_qubits_x + 2 qubits.
    The memory fields are identically zero outside the layer; the injection
    columns are masked there so the Hermitian part carries no spurious
    interior coupling. gamma is a free rescaling of the memory variables
    (dynamics of v, w independent of it) used to balance the Hermitian part.

    For kappa = 1, alpha = 0 and initial data supported outside the layer
    this reproduces WaveEquationCPML1d exactly (to round-off); the
    discrepancy otherwise scales with the initial-data norm inside the layer.

    Notes: alpha_max > 0 trades late-time/low-frequency absorption for
    improved grazing/evanescent behaviour (the layer is matched only for
    omega >> alpha), so absorption benchmarks at fixed T should use
    alpha_max = 0. The memory forms are marginally stable (exact zero
    eigenvalues with layer support); the absorption floor equals the
    projection of the initial data onto those modes -- zero for data
    supported strictly outside the layer.
    """

    def __init__(self, num_qubits_x, npml=10, h=1.0, m=2, R0=1e-3,
                 sigma_max=None, kappa_max=1.0, alpha_max=0.0,
                 gamma=None, diff_type='forward', left=True, right=True,
                 calibration='continuum'):
        if diff_type not in ('forward', 'backward'):
            raise NotImplementedError("use 'forward' or 'backward'")
        self.wave_op = WaveEquationEvolution(
            num_qubits_x, dim=1, diff_type=diff_type, h=h, periodic=False)
        self.num_qubits = num_qubits_x + 2
        N = 2**num_qubits_x
        stagger_w = -0.5 if diff_type == 'forward' else 0.5
        kw = dict(h=h, m=m, R0=R0, sigma_max=sigma_max, left=left, right=right)
        self.sigma_v = pml_sigma_profile(N, npml, stagger=0.0, **kw)
        self.sigma_w = pml_sigma_profile(N, npml, stagger=stagger_w, **kw)
        self._calibration = calibration
        # kappa, alpha graded like sigma (from the uncalibrated profiles) (kappa from 1 to kappa_max with the
        # same polynomial; alpha maximal just inside the interface, zero at
        # the wall). Normalised by the analytic sigma_max so node and
        # staggered samples see the same physical grading function.
        smax = sigma_max if sigma_max is not None \
            else -(m + 1) * np.log(R0) / (2 * npml * h)
        ref_v = self.sigma_v / smax
        ref_w = self.sigma_w / smax
        self.kappa_v = 1.0 + (kappa_max - 1.0) * ref_v
        self.kappa_w = 1.0 + (kappa_max - 1.0) * ref_w
        self.alpha_v = np.where(self.sigma_v > 0, alpha_max * (1.0 - ref_v**(1.0 / m)), 0.0)
        self.alpha_w = np.where(self.sigma_w > 0, alpha_max * (1.0 - ref_w**(1.0 / m)), 0.0)
        if calibration == 'discrete':
            self.sigma_v, self.sigma_w = calibrate_sigma_pair(
                self.sigma_v, self.sigma_w, npml, h=h, m=m, R0=R0,
                sigma_max=sigma_max, num_sides=int(left) + int(right))
        elif calibration != 'continuum':
            raise ValueError("calibration must be 'continuum' or 'discrete'")
        if gamma is None:
            smax = max(self.sigma_v.max(), self.sigma_w.max(), 1e-12)
            gamma = np.sqrt(smax * 2.0 / h)  # balances the two H1 cross blocks
        self.gamma = gamma

    def generator(self):
        D = _extract_D(self.wave_op)
        N = D.shape[0]
        Dd = D.conj().T.tocsr()
        g = self.gamma
        mask_v = (self.sigma_v > 0).astype(float)
        mask_w = (self.sigma_w > 0).astype(float)
        Kv = sp.diags(1.0 / self.kappa_v)
        Kw = sp.diags(1.0 / self.kappa_w)
        Sv = sp.diags(self.sigma_v / self.kappa_v**2)
        Sw = sp.diags(self.sigma_w / self.kappa_w**2)
        Cv = sp.diags(self.sigma_v / self.kappa_v + self.alpha_v)
        Cw = sp.diags(self.sigma_w / self.kappa_w + self.alpha_w)
        Inj_v = g * sp.diags(mask_v)
        Inj_w = g * sp.diags(mask_w)
        A = sp.bmat([
            [None,                Kv @ (-1j * D),  Inj_v, None],
            [Kw @ (-1j * Dd),     None,            None,  Inj_w],
            [None,                (1 / g) * Sv @ (1j * D), -Cv, None],
            [(1 / g) * Sw @ (1j * Dd), None,       None,  -Cw],
        ], format='csr')
        return A

    def hamiltonian_parts(self):
        return split_generator(self.generator())

    def embed(self, psi2):
        """Embed a 2-block state [v; w] into the 4-block state (memory = 0)."""
        out = np.zeros(2 * len(psi2), dtype=complex)
        out[:len(psi2)] = psi2
        return out


class WaveEquationCPML2d:
    """2D wave equation with the convolution (CPML) absorbing layer.

    The 2D wave equation u_tt = c^2 (u_xx + u_yy) in separated first-order
    form (one block per field, paper Eq. (32) d=3 style):

        dv/dt    = -i (Dx wx + Dy wy)         v  = du/dt
        dwx/dt   = -i Dx^dag v                wx = i c d_x u
        dwy/dt   = -i Dy^dag v                wy = i c d_y u

    with Dx = kron(D, I), Dy = kron(I, D) (x register more significant).
    The CFS stretch of each direction adds one convolution memory field per
    (field, stretched direction) pair (kappa = 1 here):

        dv/dt      += gamma (psi_vx + psi_vy)
        dwx/dt     += gamma psi_x
        dwy/dt     += gamma psi_y
        d(psi_vx)/dt = -(sigma_x + alpha_x) psi_vx - (1/gamma) sigma_x (-i Dx) wx
        d(psi_vy)/dt = -(sigma_y + alpha_y) psi_vy - (1/gamma) sigma_y (-i Dy) wy
        d(psi_x)/dt  = -(sigma_x + alpha_x) psi_x  - (1/gamma) sigma_x (-i Dx^dag) v
        d(psi_y)/dt  = -(sigma_y + alpha_y) psi_y  - (1/gamma) sigma_y (-i Dy^dag) v

    Corners need no special treatment: both directions' memory fields are
    simultaneously active there (the structural advantage of unsplit CPML).

    State: 8 blocks [v; wx; wy; psi_vx; psi_vy; psi_x; psi_y; 0] of size
    2^(nx+ny) each -> nx + ny + 3 qubits. The last block is unused padding
    so the register is a power of two. Memory injections are masked to the
    layer strips (the memory fields vanish identically outside them).
    """

    NUM_BLOCKS = 8

    def __init__(self, num_qubits_x, num_qubits_y=None, npml=8, h=1.0, m=2,
                 R0=1e-3, sigma_max=None, alpha_max=0.0, gamma=None,
                 diff_type='forward', calibration='continuum'):
        if diff_type not in ('forward', 'backward'):
            raise NotImplementedError("use 'forward' or 'backward'")
        if num_qubits_y is None:
            num_qubits_y = num_qubits_x
        self.num_qubits_x = num_qubits_x
        self.num_qubits_y = num_qubits_y
        self.num_qubits = num_qubits_x + num_qubits_y + 3
        self._h = h
        Nx, Ny = 2**num_qubits_x, 2**num_qubits_y

        d1x = DifferentialOperator1d(num_qubits_x, diff_type=diff_type, h=h, periodic=False)
        d1y = DifferentialOperator1d(num_qubits_y, diff_type=diff_type, h=h, periodic=False)
        D1x = sp.csr_matrix(d1x.to_matrix(sparse=True))
        D1y = sp.csr_matrix(d1y.to_matrix(sparse=True))
        Ix, Iy = sp.identity(Nx, format='csr'), sp.identity(Ny, format='csr')
        self.Dx = sp.kron(D1x, Iy, format='csr')
        self.Dy = sp.kron(Ix, D1y, format='csr')

        stagger = -0.5 if diff_type == 'forward' else 0.5
        kw = dict(h=h, m=m, R0=R0, sigma_max=sigma_max)
        s1x = pml_sigma_profile(Nx, npml, stagger=0.0, **kw)
        s1y = pml_sigma_profile(Ny, npml, stagger=0.0, **kw)
        s1x_st = pml_sigma_profile(Nx, npml, stagger=stagger, **kw)
        s1y_st = pml_sigma_profile(Ny, npml, stagger=stagger, **kw)
        if calibration == 'discrete':
            s1x, s1x_st = calibrate_sigma_pair(s1x, s1x_st, npml, h=h, m=m,
                                               R0=R0, sigma_max=sigma_max)
            s1y, s1y_st = calibrate_sigma_pair(s1y, s1y_st, npml, h=h, m=m,
                                               R0=R0, sigma_max=sigma_max)
        elif calibration != 'continuum':
            raise ValueError("calibration must be 'continuum' or 'discrete'")
        ones_x, ones_y = np.ones(Nx), np.ones(Ny)
        # sigma_x depends on x only, sigma_y on y only; staggered sampling on
        # the grid of the field each equation updates.
        self.sx_v = np.kron(s1x, ones_y)
        self.sy_v = np.kron(ones_x, s1y)
        self.sx_w = np.kron(s1x_st, ones_y)
        self.sy_w = np.kron(ones_x, s1y_st)

        smax_ref = sigma_max if sigma_max is not None \
            else -(m + 1) * np.log(R0) / (2 * npml * h)

        def alpha_of(s):
            if alpha_max == 0.0:
                return np.zeros_like(s)
            return np.where(s > 0, alpha_max * (1.0 - (s / smax_ref)**(1.0 / m)), 0.0)
        self.ax_v, self.ay_v = alpha_of(self.sx_v), alpha_of(self.sy_v)
        self.ax_w, self.ay_w = alpha_of(self.sx_w), alpha_of(self.sy_w)

        if gamma is None:
            smax = max(self.sx_v.max(), self.sy_v.max(), 1e-12)
            gamma = np.sqrt(smax * 2.0 / h)
        self.gamma = gamma

    @property
    def block_size(self):
        return 2**(self.num_qubits_x + self.num_qubits_y)

    def hamiltonian(self):
        """Undamped Hermitian H (8-block padded): A_0 = -iH."""
        M = self.block_size
        Z = None
        Dx, Dy = self.Dx, self.Dy
        blocks = [[Z] * 8 for _ in range(8)]
        blocks[0][1] = Dx
        blocks[0][2] = Dy
        blocks[1][0] = Dx.conj().T
        blocks[2][0] = Dy.conj().T
        for j in (3, 4, 5, 6, 7):  # pin empty memory/padding blocks
            blocks[j][j] = sp.csr_matrix((M, M))
        return sp.bmat(blocks, format='csr')

    def generator(self):
        g = self.gamma
        Dx, Dy = self.Dx, self.Dy
        Dxd, Dyd = Dx.conj().T.tocsr(), Dy.conj().T.tocsr()
        M = self.block_size
        mvx = (self.sx_v > 0).astype(float)
        mvy = (self.sy_v > 0).astype(float)
        mwx = (self.sx_w > 0).astype(float)
        mwy = (self.sy_w > 0).astype(float)
        Z = None
        blocks = [[Z] * 8 for _ in range(8)]
        # physical fields
        blocks[0][1] = -1j * Dx
        blocks[0][2] = -1j * Dy
        blocks[1][0] = -1j * Dxd
        blocks[2][0] = -1j * Dyd
        # memory injection (masked to the strips)
        blocks[0][3] = g * sp.diags(mvx)
        blocks[0][4] = g * sp.diags(mvy)
        blocks[1][5] = g * sp.diags(mwx)
        blocks[2][6] = g * sp.diags(mwy)
        # memory dynamics
        blocks[3][1] = (1 / g) * sp.diags(self.sx_v) @ (1j * Dx)
        blocks[4][2] = (1 / g) * sp.diags(self.sy_v) @ (1j * Dy)
        blocks[5][0] = (1 / g) * sp.diags(self.sx_w) @ (1j * Dxd)
        blocks[6][0] = (1 / g) * sp.diags(self.sy_w) @ (1j * Dyd)
        blocks[3][3] = -sp.diags(self.sx_v + self.ax_v)
        blocks[4][4] = -sp.diags(self.sy_v + self.ay_v)
        blocks[5][5] = -sp.diags(self.sx_w + self.ax_w)
        blocks[6][6] = -sp.diags(self.sy_w + self.ay_w)
        blocks[7][7] = sp.csr_matrix((M, M))
        return sp.bmat(blocks, format='csr')

    def hamiltonian_parts(self):
        return split_generator(self.generator())

    def initial_state(self, v0, normalize=True):
        """Full 8-block state from an initial velocity field v0 (flattened
        x-major, i.e. index = jx * 2^ny + jy), u(0) = 0."""
        v0 = np.asarray(v0, dtype=complex).ravel()
        out = np.zeros(8 * self.block_size, dtype=complex)
        out[:self.block_size] = v0
        if normalize:
            out /= np.linalg.norm(out)
        return out

    def extract_v(self, state):
        return state[:self.block_size]


class WaveEquationSeparatedEvolution2d(OperatorList):
    """Op-string form of the separated-block 2D wave Hamiltonian.

    H = |00><01| (x) Dx + |00><10| (x) Dy + h.c. on the 4-block state
    [v; wx; wy; 0] (two leading block qubits), i.e. the Hermitian part of
    WaveEquationSponge2d / WaveEquationCPML2d. Unlike those classes this one
    carries the operator as qubit strings, so get_bell_evolution_circ can
    synthesise the Trotter-step circuit out of elementary gates (the block
    projectors appear as 'z' characters, supported as 0-controls).
    """

    def __init__(self, num_qubits_x, num_qubits_y=None, diff_type='forward', h=1.0):
        if diff_type not in ('forward', 'backward'):
            raise NotImplementedError("use 'forward' or 'backward'")
        if num_qubits_y is None:
            num_qubits_y = num_qubits_x
        self._num_qubits_x = num_qubits_x
        self._num_qubits_y = num_qubits_y
        self._num_qubits = num_qubits_x + num_qubits_y + 2

        conj_type = 'backward' if diff_type == 'forward' else 'forward'
        dx = DifferentialOperator1d(num_qubits_x, diff_type=diff_type, h=h, periodic=False)
        dxc = DifferentialOperator1d(num_qubits_x, diff_type=conj_type, h=h, periodic=False)
        dy = DifferentialOperator1d(num_qubits_y, diff_type=diff_type, h=h, periodic=False)
        dyc = DifferentialOperator1d(num_qubits_y, diff_type=conj_type, h=h, periodic=False)

        Ix, Iy = 'I' * num_qubits_x, 'I' * num_qubits_y
        op_list, coeffs = [], {}

        def add(prefix, ops, cs, sign):
            for op, c in zip(ops, cs):
                key = prefix + op
                coeffs[key] = coeffs.get(key, 0.0) + sign * c
                if key not in op_list:
                    op_list.append(key)

        # |v><wx| (x) Dx  and  h.c. = |wx><v| (x) Dx^dag, Dx^dag = -D_conj
        add('zm', [op + Iy for op in dx.op_list], dx.coeffs, +1)
        add('zp', [op + Iy for op in dxc.op_list], dxc.coeffs, -1)
        # |v><wy| (x) Dy  and  h.c.
        add('mz', [Ix + op for op in dy.op_list], dy.coeffs, +1)
        add('pz', [Ix + op for op in dyc.op_list], dyc.coeffs, -1)

        self._op_list = [op_list[i] for i in np.argsort(op_list)]
        self._coeffs = [coeffs[op] for op in np.sort(op_list)]


class WaveEquationSponge2d:
    """2D wave equation with a graded sponge (CAP) absorbing layer.

    Damps all three physical fields with the local rate sigma_x + sigma_y:

        A = -i H - diag(sigma_tot, sigma_tot_wx, sigma_tot_wy, 0).

    Not perfectly matched (a graded layer still reflects ~ percent level),
    but the Hermitian part H1 = -diag(...) <= 0 is diagonal and provably
    negative semidefinite -- the unconditionally clean input for
    Schrodingerisation, and the damping circuit is diagonal.

    State: 4 blocks [v; wx; wy; 0] -> nx + ny + 2 qubits.
    """

    NUM_BLOCKS = 4

    def __init__(self, num_qubits_x, num_qubits_y=None, npml=8, h=1.0, m=2,
                 R0=1e-3, sigma_max=None, diff_type='forward',
                 calibration='continuum'):
        if num_qubits_y is None:
            num_qubits_y = num_qubits_x
        self.num_qubits_x = num_qubits_x
        self.num_qubits_y = num_qubits_y
        self.num_qubits = num_qubits_x + num_qubits_y + 2
        Nx, Ny = 2**num_qubits_x, 2**num_qubits_y

        d1x = DifferentialOperator1d(num_qubits_x, diff_type=diff_type, h=h, periodic=False)
        d1y = DifferentialOperator1d(num_qubits_y, diff_type=diff_type, h=h, periodic=False)
        D1x = sp.csr_matrix(d1x.to_matrix(sparse=True))
        D1y = sp.csr_matrix(d1y.to_matrix(sparse=True))
        Ix, Iy = sp.identity(Nx, format='csr'), sp.identity(Ny, format='csr')
        self.Dx = sp.kron(D1x, Iy, format='csr')
        self.Dy = sp.kron(Ix, D1y, format='csr')

        stagger = -0.5 if diff_type == 'forward' else 0.5
        kw = dict(h=h, m=m, R0=R0, sigma_max=sigma_max)
        s1x = pml_sigma_profile(Nx, npml, stagger=0.0, **kw)
        s1y = pml_sigma_profile(Ny, npml, stagger=0.0, **kw)
        s1x_st = pml_sigma_profile(Nx, npml, stagger=stagger, **kw)
        s1y_st = pml_sigma_profile(Ny, npml, stagger=stagger, **kw)
        if calibration == 'discrete':
            s1x, s1x_st = calibrate_sigma_pair(s1x, s1x_st, npml, h=h, m=m,
                                               R0=R0, sigma_max=sigma_max)
            s1y, s1y_st = calibrate_sigma_pair(s1y, s1y_st, npml, h=h, m=m,
                                               R0=R0, sigma_max=sigma_max)
        elif calibration != 'continuum':
            raise ValueError("calibration must be 'continuum' or 'discrete'")
        ones_x, ones_y = np.ones(Nx), np.ones(Ny)
        st_v = np.kron(s1x, ones_y) + np.kron(ones_x, s1y)
        st_wx = np.kron(s1x_st, ones_y) + np.kron(ones_x, s1y)
        st_wy = np.kron(s1x, ones_y) + np.kron(ones_x, s1y_st)
        zeros = np.zeros(Nx * Ny)
        self.sigma_blocks = np.concatenate([st_v, st_wx, st_wy, zeros])

    @property
    def block_size(self):
        return 2**(self.num_qubits_x + self.num_qubits_y)

    def hamiltonian(self):
        M = self.block_size
        Z = None
        blocks = [[Z] * 4 for _ in range(4)]
        blocks[0][1] = self.Dx
        blocks[0][2] = self.Dy
        blocks[1][0] = self.Dx.conj().T
        blocks[2][0] = self.Dy.conj().T
        blocks[3][3] = sp.csr_matrix((M, M))
        return sp.bmat(blocks, format='csr')

    def generator(self):
        return (-1j * self.hamiltonian() - sp.diags(self.sigma_blocks)).tocsr()

    def hamiltonian_parts(self):
        """H1 = -diag(sigma) <= 0 (diagonal), H2 = -H."""
        return (-sp.diags(self.sigma_blocks).tocsr(),
                (-self.hamiltonian()).tocsr())

    def initial_state(self, v0, normalize=True):
        v0 = np.asarray(v0, dtype=complex).ravel()
        out = np.zeros(4 * self.block_size, dtype=complex)
        out[:self.block_size] = v0
        if normalize:
            out /= np.linalg.norm(out)
        return out

    def extract_v(self, state):
        return state[:self.block_size]
