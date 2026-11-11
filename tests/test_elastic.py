"""Core validation of the elastic wave operator."""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from lib import ElasticWaveEvolution2d
from lib.elastic import isotropic_voigt_sqrt


def test_voigt_sqrt():
    cp, cs = 1.0, 0.5
    C12 = isotropic_voigt_sqrt(cp, cs)
    mu = cs**2
    lam = cp**2 - 2 * mu
    C = np.array([[lam + 2 * mu, lam, 0], [lam, lam + 2 * mu, 0], [0, 0, 2 * mu]])
    assert np.allclose(C12 @ C12, C, atol=1e-14)
    with pytest.raises(ValueError):
        isotropic_voigt_sqrt(1.0, 0.9)  # lambda < 0


def test_hermitian_and_energy_conserving():
    op = ElasticWaveEvolution2d(3, 3, cp=1.0, cs=0.5)
    H = op.to_matrix(sparse=True)
    assert abs(H - H.conj().T).max() < 1e-13
    A0 = op.generator_undamped()
    assert abs(A0 + A0.T).max() < 1e-13  # exactly antisymmetric

    rng = np.random.default_rng(3)
    z0 = rng.standard_normal(8 * op.block_size)
    z0 /= np.linalg.norm(z0)
    z = expm_multiply(A0.tocsc() * 5.0, z0.astype(complex))
    assert np.linalg.norm(z) == pytest.approx(1.0, abs=1e-10)


def test_periodic_dispersion_exact():
    """On a periodic constant-coefficient grid the spectrum of H must be
    exactly the union over discrete k of the symbol eigenfrequencies
    {0, +-sqrt(eig M(k))} -- this pins the entire encoding including the
    sqrt(2) Voigt convention and the C^{1/2} factors."""
    nx = ny = 2
    op = ElasticWaveEvolution2d(nx, ny, cp=1.0, cs=0.5, periodic=True)
    H = op.to_matrix(sparse=False)
    eigs = np.sort(np.linalg.eigvalsh(H))

    Nx, Ny = 2**nx, 2**ny
    expected = [0.0] * (3 * Nx * Ny)  # padding blocks
    for ix in range(Nx):
        for iy in range(Ny):
            kx = 2 * np.pi * ix / Nx
            ky = 2 * np.pi * iy / Ny
            w2 = np.linalg.eigvalsh(op.symbol_matrix(kx, ky))
            w2 = np.clip(w2, 0.0, None)
            for w in np.sqrt(w2):
                expected += [w, -w]
            expected.append(0.0)  # 5 fields -> 2 propagating pairs + 1 null
    expected = np.sort(np.asarray(expected))
    assert eigs.shape == expected.shape
    assert np.abs(eigs - expected).max() < 1e-10


def test_continuum_speeds():
    """Smallest axis-aligned mode: omega/k -> cp (longitudinal) and cs
    (shear) within the O(kh) forward-difference error."""
    cp, cs = 1.0, 0.5
    op = ElasticWaveEvolution2d(4, 4, cp=cp, cs=cs, periodic=True)
    k = 2 * np.pi / 16
    w2 = np.sort(np.linalg.eigvalsh(op.symbol_matrix(k, 0.0)))
    kd = abs((np.exp(1j * k) - 1.0))  # |discrete wavenumber|
    # axis-aligned: the 2x2 symbol has exactly the shear and longitudinal
    # branches, omega = cs*kd and cp*kd
    speeds = np.sqrt(w2) / kd
    assert speeds[0] == pytest.approx(cs, rel=1e-10)
    assert speeds[1] == pytest.approx(cp, rel=1e-10)


def test_pwave_propagates_isotropically():
    """A radial velocity pulse spreads; energy is conserved and leaves the
    centre region (sanity of the assembled dynamics)."""
    op = ElasticWaveEvolution2d(4, 4, cp=1.0, cs=0.5)
    N = 16
    x = np.arange(N) - N // 2
    g = np.exp(-0.5 * (x[:, None]**2 + x[None, :]**2) / 1.5**2)
    z0 = op.initial_state(g, g)
    A0 = op.generator_undamped().tocsc()
    z = expm_multiply(A0 * 6.0, z0)
    assert np.linalg.norm(z) == pytest.approx(1.0, abs=1e-10)
    M = op.block_size
    centre = np.abs(z[:M].reshape(N, N))[N//2-2:N//2+2, N//2-2:N//2+2]
    assert centre.max() < 0.5 * np.abs(z0[:M]).max()
