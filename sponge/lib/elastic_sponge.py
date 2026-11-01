"""Graded sponge (complex absorbing potential) for the 2D isotropic
velocity--stress system.

Extracted verbatim from qwave-elastic/elib/elastic_cpml.py @ 3038d99
(class ElasticSponge2d only; the CPML classes belong to the CPML papers).

The undamped generator -iH of ElasticWaveEvolution2d is Hermitian-skew;
the sponge adds -diag(sigma) on the five physical blocks, so the
Hermitian part H1 = -diag(sigma) <= 0 is diagonal and negative
semidefinite and the Schrodingerised recovery has no threshold.
"""

import numpy as np
import scipy.sparse as sp

from .absorbing import pml_sigma_profile
from .elastic import ElasticWaveEvolution2d


class ElasticSponge2d:
    """Graded sponge (CAP): damp all five physical fields with
    sigma_x + sigma_y. 8 blocks, H1 = -diag(sigma) <= 0."""

    NUM_BLOCKS = 8

    def __init__(self, num_qubits_x, num_qubits_y=None, npml=8, cp=1.0,
                 cs=None, rho=1.0, h=1.0, m=2, R0=1e-3, sigma_max=None):
        self.wave = ElasticWaveEvolution2d(num_qubits_x, num_qubits_y,
                                           cp=cp, cs=cs, rho=rho, h=h)
        self.num_qubits = self.wave.num_qubits
        Nx = 2**self.wave.num_qubits_x
        Ny = 2**self.wave.num_qubits_y
        kw = dict(h=h, m=m, R0=R0, c=cp, sigma_max=sigma_max)
        s1x = pml_sigma_profile(Nx, npml, stagger=0.0, **kw)
        s1y = pml_sigma_profile(Ny, npml, stagger=0.0, **kw)
        st = np.kron(s1x, np.ones(Ny)) + np.kron(np.ones(Nx), s1y)
        M = self.wave.block_size
        self.sigma_blocks = np.concatenate([np.tile(st, 5), np.zeros(3 * M)])

    @property
    def block_size(self):
        return self.wave.block_size

    def generator(self):
        return (self.wave.generator_undamped()
                - sp.diags(self.sigma_blocks)).tocsr()

    def hamiltonian_parts(self):
        H = 1j * self.wave.generator_undamped()
        return (-sp.diags(self.sigma_blocks).tocsr(), (-H).tocsr())

    def initial_state(self, vx0, vy0=None, normalize=True):
        return self.wave.initial_state(vx0, vy0, normalize=normalize)
