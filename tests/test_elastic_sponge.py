"""Elastic sponge: diagonal, negative-semidefinite Hermitian part."""

import scipy.sparse as sp

from lib.elastic_sponge import ElasticSponge2d


def test_sponge_clean_h1():
    op = ElasticSponge2d(3, 3, npml=2, cp=1.0, cs=0.5)
    H1, _ = op.hamiltonian_parts()
    assert abs(H1 - sp.diags(H1.diagonal())).max() < 1e-14
    assert H1.diagonal().real.max() <= 1e-14
