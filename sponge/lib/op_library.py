import re
import sympy
import numpy as np
import scipy


from typing import Optional


def get_op_list(operator):

    op_list = []
    coeffs = []
    for term in operator.args:
        term = str(term).replace('**', 'x').split('*')
        
        try:
            coeff = float(term[0])
            i = 1
        except:
            if term[0][0] == '-':
                coeff = -1.0
                term[0] = term[0][1:]
            else:
                coeff = 1.0
            i = 0
        coeffs.append(coeff)

        expr = []
        for j in range(i, len(term)):
            s = term[j]
            m = re.match('[I,p,m,z,o,X,Y,Z]x[0-9]+', s)
            if m is None:
                expr.append(s)
            else:
                expr += [ s[0] ] * int(s[2:])
        op_list.append(''.join(expr))
        #op_list.append(expr)
    
    return op_list, coeffs


class OperatorList:

    @property
    def num_qubits(self):
        return self._num_qubits
    
    @property
    def op_list(self):
        return self._op_list
    
    @property
    def coeffs(self):
        return self._coeffs
    
    def to_matrix(self, sparse=True):

        I = np.eye(2)
        s01 = np.array([[0., 1.],
                        [0., 0.]])
        s10 = np.array([[0., 0.],
                        [1., 0.]])
        s00 = np.array([[1., 0.],
                        [0., 0.]])
        s11 = np.array([[0., 0.],
                        [0., 1.]])
        X = np.array([[0., 1.],
                    [1., 0.]])
        Y = np.array([[0., -1j],
                    [1j, 0.]])
        Z = np.array([[1., 0.],
                    [0., -1.]])

        mat_dict = {'I': I, 'm': s01, 'p': s10, 'z': s00, 'o': s11}
        if sparse:
            for k, v in mat_dict.items():
                mat_dict[k] = scipy.sparse.csr_matrix(v)
            mat = scipy.sparse.csr_matrix((2**self._num_qubits,)*2, dtype=np.complex128)
        else:
            mat = np.zeros((2**self._num_qubits,)*2, dtype=np.complex128)

        for term, coeff in zip(self._op_list, self._coeffs):
            tmp = 1.
            for op in term:
                if sparse:
                    tmp = scipy.sparse.kron(tmp, mat_dict[op], format='csr')
                else:
                    tmp = np.kron(tmp, mat_dict[op])
            mat += coeff * tmp
        
        return mat


class LaplacianOperator1d(OperatorList):
    
    def __init__(self, num_qubits: int, is_pauli: bool=False, h: float=1.0, periodic: bool=False):
    
        self._num_qubits = num_qubits
        self._num_qubits_x = num_qubits
        self._is_pauli = is_pauli
        self._h = h
        self._periodic = periodic

        if is_pauli:
            X, Y, Z, I = sympy.symbols('X Y Z I', commutative=False)
            s10 = 0.5 * (X - 1j * Y)
            s01 = 0.5 * (X + 1j * Y)
            s00 = 0.5 * (I + Z)
            s11 = 0.5 * (I - Z)
        else:
            s10, s01, s00, s11, I = sympy.symbols('p m z o I', commutative=False)

        # core of QTT for gradients w.r.t. x- and y-coordinates
        Cx = sympy.Matrix([
            [I, s01, s10],
            [0, s10, 0,],
            [0, 0, s01],
        ])

        # boundary cores
        C_left = sympy.Matrix([1, 0, 0]).T
        C_right = sympy.Matrix([-2, 1, 1])

        operator = 1 / h**2 * C_left * Cx**num_qubits * C_right
        operator = sympy.expand(operator[0])

        op_list, coeffs = get_op_list(operator)

        if self._periodic:
            operator += 1 / h**2 * s10 ** num_qubits + 1 / h**2 * s01 ** num_qubits
        
        self._op_list = [op_list[i] for i in np.argsort(op_list)]
        self._coeffs = [coeffs[i] for i in np.argsort(op_list)]


class LaplacianOperator2d(OperatorList):

    @property
    def num_qubits(self):
        return self._num_qubits

    @property
    def num_qubits_x(self):
        return self._num_qubits_x
    
    @property
    def num_qubits_y(self):
        return self._num_qubits_y
    
    def __init__(self, num_qubits_x: int, num_qubits_y: int=None, is_scalewise: bool=False, \
                is_pauli: bool=False, h: float=1.0, periodic: bool=False):
    
        if num_qubits_y is None:
            num_qubits_y = num_qubits_x

        num_qubits = num_qubits_x + num_qubits_y
        self._num_qubits = num_qubits
        self._num_qubits_x = num_qubits_x
        self._num_qubits_y = num_qubits_y
        self._is_scalewise = is_scalewise
        self._is_pauli = is_pauli
        self._h = h
        self._periodic = periodic

        if is_pauli:
            X, Y, Z, I = sympy.symbols('X Y Z I', commutative=False)
            s10 = 0.5 * (X - 1j * Y) 
            s01 = 0.5 * (X + 1j * Y)
            s00 = 0.5 * (I + Z)
            s11 = 0.5 * (I - Z)   
        else:
            s10, s01, s00, s11, I = sympy.symbols('p m z o I', commutative=False)

        # core of QTT for gradients w.r.t. x- and y-coordinates
        Cx = sympy.Matrix([
            [I, s01, s10, 0, 0, 0],
            [0, s10, 0, 0, 0, 0],
            [0, 0, s01, 0, 0, 0],
            [0, 0, 0, I, 0, 0],
            [0, 0, 0, 0, I, 0],
            [0, 0, 0, 0, 0, I]
        ])
        Cy = sympy.Matrix([
            [I, 0, 0, 0, 0, 0],
            [0, I, 0, 0, 0, 0],
            [0, 0, I, 0, 0, 0],
            [0, 0, 0, I, s01, s10],
            [0, 0, 0, 0, s10, 0],
            [0, 0, 0, 0, 0, s01]
            ])

        # boundary cores
        C_left = sympy.Matrix([1, 0, 0, 1, 0, 0]).T
        C_right = sympy.Matrix([-2, 1, 1, -2, 1, 1])

        # contraction of QTT to generate qubit operator
        if is_scalewise:
            operator = 1 / h**2 * C_left * Cx**max(num_qubits_x - num_qubits_y, 0) * Cy**max(num_qubits_y - num_qubits_x, 0) * (Cx*Cy)**min(num_qubits_x, num_qubits_y) * C_right
        else:
            operator = 1 / h**2 * C_left * Cx**num_qubits_x * Cy**num_qubits_y * C_right
        operator = sympy.expand(operator[0])

        if self._periodic:
            if is_scalewise:
                operator += 1 / h**2 * s10**max(num_qubits_x - num_qubits_y, 0) * I**max(num_qubits_y - num_qubits_x, 0) * (s10*I)**min(num_qubits_x, num_qubits_y) \
                            + 1 / h**2 * s01**max(num_qubits_x - num_qubits_y, 0) * I**max(num_qubits_y - num_qubits_x, 0) * (s01*I)**min(num_qubits_x, num_qubits_y) \
                            + 1 / h**2 * I**max(num_qubits_x - num_qubits_y, 0) * s10**max(num_qubits_y - num_qubits_x, 0) * (I*s10)**min(num_qubits_x, num_qubits_y) \
                            + 1 / h**2 * I**max(num_qubits_x - num_qubits_y, 0) * s01**max(num_qubits_y - num_qubits_x, 0) * (I*s01)**min(num_qubits_x, num_qubits_y)
            else:
                operator += 1 / h**2 * s10**num_qubits_x * I**num_qubits_y + 1 / h**2 * s01**num_qubits_x * I**num_qubits_y \
                            + 1 / h**2 * I**num_qubits_x * s10**num_qubits_y + 1 / h**2 * I**num_qubits_x * s01**num_qubits_y

        op_list, coeffs = get_op_list(operator)
        self._op_list = [op_list[i] for i in np.argsort(op_list)]
        self._coeffs = [coeffs[i] for i in np.argsort(op_list)]


class DifferentialOperator1d(OperatorList):
    
    def __init__(self, num_qubits: int, is_pauli=False, diff_type: str='central', h: float=1.0, periodic: bool=False):
    
        self._num_qubits = num_qubits
        self._num_qubits_x = num_qubits
        self._is_pauli = is_pauli
        self._diff_type = diff_type
        self._h = h
        self._periodic = periodic

        if is_pauli:
            X, Y, Z, I = sympy.symbols('X Y Z I', commutative=False)
            s10 = 0.5 * (X - 1j * Y)
            s01 = 0.5 * (X + 1j * Y)
            s00 = 0.5 * (I + Z)
            s11 = 0.5 * (I - Z)
        else:
            s10, s01, s00, s11, I = sympy.symbols('p m z o I', commutative=False)

        if diff_type == 'central':
            # core of QTT for gradients w.r.t. x--coordinates
            Cx = sympy.Matrix([
                [I, s01, s10],
                [0, s10, 0,],
                [0, 0, s01],
            ])

            # boundary cores
            C_left = sympy.Matrix([0.5, 0, 0]).T
            C_right = sympy.Matrix([0, 1, -1])
        
        elif diff_type == 'forward':
            # core of QTT for gradients w.r.t. x-coordinates
            Cx = sympy.Matrix([
                [I, s01],
                [0, s10,]
            ])

            # boundary cores
            C_left = sympy.Matrix([1, 0]).T
            C_right = sympy.Matrix([-1, 1])
        
        elif diff_type == 'backward':
            # core of QTT for gradients w.r.t. x-coordinates
            Cx = sympy.Matrix([
                [I, s10],
                [0, s01],
            ])

            # boundary cores
            C_left = sympy.Matrix([1, 0]).T
            C_right = sympy.Matrix([1, -1])
        
        else:
            raise NotImplementedError()

        operator = 1 / h * C_left * Cx**num_qubits * C_right
        operator = sympy.expand(operator[0])

        if self._periodic:
            if diff_type == 'central':
                operator += 0.5 / h * s10 ** num_qubits - 0.5 / h * s01 ** num_qubits

            elif diff_type == 'forward':
                operator += 1 / h * s10 ** num_qubits
            
            elif diff_type == 'backward':
                operator -= 1 / h * s01 ** num_qubits

        op_list, coeffs = get_op_list(operator)
        self._op_list = [op_list[i] for i in np.argsort(op_list)]
        self._coeffs = [coeffs[i] for i in np.argsort(op_list)]


class WaveEquationEvolution(OperatorList):

    @property
    def num_qubits(self):
        return self._num_qubits
    
    @property
    def dim(self):
        return self._dim
    
    @property
    def diff_type(self):
        return self._diff_type
    
    @property
    def h(self):
        return self._h

    def __init__(self, num_qubits_x: int, num_qubits_y: Optional[int]=None, \
                dim: int=1, diff_type: str='central', h: float=1.0, periodic: bool=False):

        self._dim = dim
        self._h = h
        self._periodic = periodic

        if dim == 1:
            self._num_qubits_x = num_qubits_x
            self._num_qubits = self._num_qubits_x + 1
        
        elif dim == 2:
            self._num_qubits_x = num_qubits_x
            self._num_qubits_y = num_qubits_y if num_qubits_y is not None else num_qubits_x
            self._num_qubits = self._num_qubits_x + self._num_qubits_y + 1
        
        else:
            raise NotImplementedError()
        
        if diff_type not in ['central', 'forward', 'backward']:
            raise NotImplementedError()            

        self._diff_type = diff_type
        
        op_list = []
        coeffs = {}
        
        if dim == 1:
            if diff_type == 'central':

                diff_op = DifferentialOperator1d(self._num_qubits_x, is_pauli=False, diff_type='central', h=h, periodic=periodic)

                for op, coeff, in zip(diff_op._op_list, diff_op._coeffs):
                    if 'm' + op not in op_list:
                        op_list.append('m' + op)
                        coeffs['m' + op] = coeff
                    else:
                        coeffs['m' + op] += coeff
                for op, coeff, in zip(diff_op._op_list, diff_op._coeffs):
                    if 'p' + op not in op_list:
                        op_list.append('p' + op)
                        coeffs['p' + op] = -coeff
                    else:
                        coeffs['p' + op] -= coeff
            
            elif diff_type == 'forward':

                diff_op = DifferentialOperator1d(self._num_qubits_x, is_pauli=False, diff_type='forward', h=h, periodic=periodic)

                for op, coeff, in zip(diff_op._op_list, diff_op._coeffs):
                    if 'm' + op not in op_list:
                        op_list.append('m' + op)
                        coeffs['m' + op] = coeff
                    else:
                        coeffs['m' + op] += coeff
                
                diff_op = DifferentialOperator1d(self._num_qubits_x, is_pauli=False, diff_type='backward', h=h, periodic=periodic)

                for op, coeff, in zip(diff_op._op_list, diff_op._coeffs):
                    if 'p' + op not in op_list:
                        op_list.append('p' + op)
                        coeffs['p' + op] = -coeff
                    else:
                        coeffs['p' + op] -= coeff
            
            elif diff_type == 'backward':

                diff_op = DifferentialOperator1d(self._num_qubits_x, is_pauli=False, diff_type='backward', h=h, periodic=periodic)

                for op, coeff, in zip(diff_op._op_list, diff_op._coeffs):
                    if 'm' + op not in op_list:
                        op_list.append('m' + op)
                        coeffs['m' + op] = coeff
                    else:
                        coeffs['m' + op] += coeff
                
                diff_op = DifferentialOperator1d(self._num_qubits_x, is_pauli=False, diff_type='forward', h=h, periodic=periodic)

                for op, coeff, in zip(diff_op._op_list, diff_op._coeffs):
                    if 'p' + op not in op_list:
                        op_list.append('p' + op)
                        coeffs['p' + op] = -coeff
                    else:
                        coeffs['p' + op] -= coeff
                        
        elif dim == 2:
            if diff_type == 'central':

                diff_op_x = DifferentialOperator1d(self._num_qubits_x, is_pauli=False, diff_type='central', h=h, periodic=periodic)
                diff_op_y = DifferentialOperator1d(self._num_qubits_y, is_pauli=False, diff_type='central', h=h, periodic=periodic)

                diff_op_list = [op + 'I'*self._num_qubits_y for op in diff_op_x._op_list] \
                            + ['I'*self._num_qubits_x + op for op in diff_op_y._op_list]
                diff_op_coeffs = diff_op_x._coeffs + [-1j*coeff for coeff in diff_op_y._coeffs]
                
                for op, coeff, in zip(diff_op_list, diff_op_coeffs):
                    if 'm' + op not in op_list:
                        op_list.append('m' + op)
                        coeffs['m' + op] = coeff
                    else:
                        coeffs['m' + op] += coeff
                
                diff_op_list = [op + 'I'*self._num_qubits_y for op in diff_op_x._op_list] \
                            + ['I'*self._num_qubits_x + op for op in diff_op_y._op_list]
                diff_op_coeffs = diff_op_x._coeffs + [1j*coeff for coeff in diff_op_y._coeffs]

                for op, coeff, in zip(diff_op_list, diff_op_coeffs):
                    if 'p' + op not in op_list:
                        op_list.append('p' + op)
                        coeffs['p' + op] = -coeff
                    else:
                        coeffs['p' + op] -= coeff

            elif diff_type == 'forward':

                diff_op_x = DifferentialOperator1d(self._num_qubits_x, is_pauli=False, diff_type='forward', h=h, periodic=periodic)
                diff_op_y = DifferentialOperator1d(self._num_qubits_y, is_pauli=False, diff_type='forward', h=h, periodic=periodic)

                diff_op_list = [op + 'I'*self._num_qubits_y for op in diff_op_x._op_list] \
                            + ['I'*self._num_qubits_x + op for op in diff_op_y._op_list]
                diff_op_coeffs = diff_op_x._coeffs + [-1j*coeff for coeff in diff_op_y._coeffs]
                
                for op, coeff, in zip(diff_op_list, diff_op_coeffs):
                    if 'm' + op not in op_list:
                        op_list.append('m' + op)
                        coeffs['m' + op] = coeff
                    else:
                        coeffs['m' + op] += coeff

                diff_op_x = DifferentialOperator1d(self._num_qubits_x, is_pauli=False, diff_type='backward', h=h, periodic=periodic)
                diff_op_y = DifferentialOperator1d(self._num_qubits_y, is_pauli=False, diff_type='backward', h=h, periodic=periodic)

                diff_op_list = [op + 'I'*self._num_qubits_y for op in diff_op_x._op_list] \
                            + ['I'*self._num_qubits_x + op for op in diff_op_y._op_list]
                diff_op_coeffs = diff_op_x._coeffs + [1j*coeff for coeff in diff_op_y._coeffs]

                for op, coeff, in zip(diff_op_list, diff_op_coeffs):
                    if 'p' + op not in op_list:
                        op_list.append('p' + op)
                        coeffs['p' + op] = -coeff
                    else:
                        coeffs['p' + op] -= coeff

            elif diff_type == 'backward':

                diff_op_x = DifferentialOperator1d(self._num_qubits_x, is_pauli=False, diff_type='backward', h=h, periodic=periodic)
                diff_op_y = DifferentialOperator1d(self._num_qubits_y, is_pauli=False, diff_type='backward', h=h, periodic=periodic)

                diff_op_list = [op + 'I'*self._num_qubits_y for op in diff_op_x._op_list] \
                            + ['I'*self._num_qubits_x + op for op in diff_op_y._op_list]
                diff_op_coeffs = diff_op_x._coeffs + [-1j*coeff for coeff in diff_op_y._coeffs]
                
                for op, coeff, in zip(diff_op_list, diff_op_coeffs):
                    if 'm' + op not in op_list:
                        op_list.append('m' + op)
                        coeffs['m' + op] = coeff
                    else:
                        coeffs['m' + op] += coeff


                diff_op_x = DifferentialOperator1d(self._num_qubits_x, is_pauli=False, diff_type='forward', h=h, periodic=periodic)
                diff_op_y = DifferentialOperator1d(self._num_qubits_y, is_pauli=False, diff_type='forward', h=h, periodic=periodic)

                diff_op_list = [op + 'I'*self._num_qubits_y for op in diff_op_x._op_list] \
                            + ['I'*self._num_qubits_x + op for op in diff_op_y._op_list]
                diff_op_coeffs = diff_op_x._coeffs + [1j*coeff for coeff in diff_op_y._coeffs]

                for op, coeff, in zip(diff_op_list, diff_op_coeffs):
                    if 'p' + op not in op_list:
                        op_list.append('p' + op)
                        coeffs['p' + op] = -coeff
                    else:
                        coeffs['p' + op] -= coeff
        
        self._op_list = [op_list[i] for i in np.argsort(op_list)]
        self._coeffs = [coeffs[op] for op in np.sort(op_list)]


class AdvectionEquationEvolution(OperatorList):

    @property
    def num_qubits(self):
        return self._num_qubits
    
    @property
    def dim(self):
        return self._dim
    
    @property
    def diff_type(self):
        return self._diff_type
    
    @property
    def h(self):
        return self._h
    
    def __init__(self, num_qubits_x: int, num_qubits_y: Optional[int]=None, \
                dim: int=1, diff_type: str='central', h: float=1.0, periodic: bool=False,  \
                vx: float=1.0, vy: Optional[float]=None):

        self._dim = dim
        self._h = h
        self._periodic = periodic

        if dim == 1:
            self._num_qubits_x = num_qubits_x
            self._num_qubits = self._num_qubits_x
            self._vx = vx
        
        elif dim == 2:
            self._num_qubits_x = num_qubits_x
            self._num_qubits_y = num_qubits_y if num_qubits_y is not None else num_qubits_x
            self._num_qubits = self._num_qubits_x + self._num_qubits_y
            self._vx = vx
            self._vy = vy if vy is not None else 0.0
        
        else:
            raise NotImplementedError()
        
        if diff_type not in ['central']:
            raise NotImplementedError()            

        self._diff_type = diff_type
        
        op_list = []
        coeffs = {}
        
        if dim == 1:
            if diff_type == 'central':

                diff_op = DifferentialOperator1d(self._num_qubits_x, is_pauli=False, diff_type='central', h=h, periodic=periodic)

                for op, coeff, in zip(diff_op._op_list, diff_op._coeffs):
                    if op not in op_list:
                        op_list.append(op)
                        coeffs[op] = -1j * self._vx * coeff
                    else:
                        coeffs[op] += -1j * self._vx * coeff
            
        elif dim == 2:
            if diff_type == 'central':

                diff_op_x = DifferentialOperator1d(self._num_qubits_x, is_pauli=False, diff_type='central', h=h, periodic=periodic)
                diff_op_y = DifferentialOperator1d(self._num_qubits_y, is_pauli=False, diff_type='central', h=h, periodic=periodic)
                
                for op, coeff, in zip(diff_op_x._op_list, diff_op_x._coeffs):
                    if op + 'I'*self._num_qubits_y not in op_list:
                        op_list.append(op + 'I'*self._num_qubits_y)
                        coeffs[op + 'I'*self._num_qubits_y] = -1j * self._vx * coeff
                    else:
                        coeffs[op + 'I'*self._num_qubits_y] += -1j * self._vx * coeff
                
                for op, coeff, in zip(diff_op_y._op_list, diff_op_y._coeffs):
                    if 'I'*self._num_qubits_x + op not in op_list:
                        op_list.append('I'*self._num_qubits_x + op)
                        coeffs['I'*self._num_qubits_x + op] = -1j * self._vy * coeff
                    else:
                        coeffs['I'*self._num_qubits_x + op] += -1j * self._vy * coeff
                    
        self._op_list = [op_list[i] for i in np.argsort(op_list)]
        self._coeffs = [coeffs[op] for op in np.sort(op_list)]
