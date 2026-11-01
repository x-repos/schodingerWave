"""Op-string form of the elastic wave Hamiltonian for circuit synthesis.

Encodes H = i A_0 of ElasticWaveEvolution2d as qubit operator strings so
that the (projector-extended) Bell-basis circuit can synthesise the
Trotter step from elementary gates. The 5 physical fields live on a
3-qubit block-label register; a coupling |b><b'| between blocks is the
3-character prefix built from the bit pairs of (b, b') with the library
alphabet (0,0)->'z', (1,1)->'o', (0,1)->'m', (1,0)->'p'. The spatial
factors are the forward/backward difference op-strings of
DifferentialOperator1d, weighted by the constant C^{1/2} entries.
"""

import numpy as np

from .op_library import OperatorList, DifferentialOperator1d
from .elastic import isotropic_voigt_sqrt

_SWAP = str.maketrans('mp', 'pm')


def _prefix(row_block, col_block, nbits=3):
    chars = []
    for k in reversed(range(nbits)):
        rb, cb = (row_block >> k) & 1, (col_block >> k) & 1
        chars.append({(0, 0): 'z', (1, 1): 'o', (0, 1): 'm', (1, 0): 'p'}[(rb, cb)])
    return ''.join(chars)


class ElasticSeparatedEvolution2d(OperatorList):
    """H = i A_0 of the staggered-pairing elastic system as op strings.

    Blocks 0..4 = [v~x, v~y, s_xx, s_yy, s_xy]; blocks 5..7 empty.
    num_qubits = nx + ny + 3. to_matrix() equals the padded Hermitian
    matrix of ElasticWaveEvolution2d exactly (pinned by a unit test).
    """

    def __init__(self, num_qubits_x, num_qubits_y=None, cp=1.0, cs=None,
                 rho=1.0, h=1.0):
        if num_qubits_y is None:
            num_qubits_y = num_qubits_x
        if cs is None:
            cs = cp / np.sqrt(3.0)
        self._num_qubits_x = num_qubits_x
        self._num_qubits_y = num_qubits_y
        self._num_qubits = num_qubits_x + num_qubits_y + 3
        C = isotropic_voigt_sqrt(cp, cs, rho)
        r = 1.0 / np.sqrt(rho)
        s2 = 1.0 / np.sqrt(2.0)

        dxf = DifferentialOperator1d(num_qubits_x, diff_type='forward', h=h, periodic=False)
        dxb = DifferentialOperator1d(num_qubits_x, diff_type='backward', h=h, periodic=False)
        dyf = DifferentialOperator1d(num_qubits_y, diff_type='forward', h=h, periodic=False)
        dyb = DifferentialOperator1d(num_qubits_y, diff_type='backward', h=h, periodic=False)
        Ix, Iy = 'I' * num_qubits_x, 'I' * num_qubits_y

        def terms(d_op, axis):
            if axis == 'x':
                return [(op + Iy, c) for op, c in zip(d_op.op_list, d_op.coeffs)]
            return [(Ix + op, c) for op, c in zip(d_op.op_list, d_op.coeffs)]

        # G blocks in op-string form (staggered pairing: backward diagonal
        # strains, forward shear), G[a][b] -> list of (string, coeff)
        G = {
            (0, 0): [(s, C[0, 0] * r * c) for s, c in terms(dxb, 'x')],
            (0, 1): [(s, C[0, 1] * r * c) for s, c in terms(dyb, 'y')],
            (1, 0): [(s, C[1, 0] * r * c) for s, c in terms(dxb, 'x')],
            (1, 1): [(s, C[1, 1] * r * c) for s, c in terms(dyb, 'y')],
            (2, 0): [(s, C[2, 2] * r * s2 * c) for s, c in terms(dyf, 'y')],
            (2, 1): [(s, C[2, 2] * r * s2 * c) for s, c in terms(dxf, 'x')],
        }

        op_list, coeffs = [], {}

        def add(key, c):
            coeffs[key] = coeffs.get(key, 0.0) + c
            if key not in op_list:
                op_list.append(key)

        for (a, b), tl in G.items():
            row, col = 2 + a, b               # ds_a/dt = +G v~_b -> H row
            pre = _prefix(row, col)
            pre_T = _prefix(col, row)
            for s_str, c in tl:
                # H[row,col] = i*G entry; Hermitian partner with -i and
                # the m<->p transposed string
                add(pre + s_str, 1j * c)
                add(pre_T + s_str.translate(_SWAP), -1j * c)

        order = np.argsort(op_list)
        self._op_list = [op_list[i] for i in order]
        self._coeffs = [coeffs[op_list[i]] for i in order]
