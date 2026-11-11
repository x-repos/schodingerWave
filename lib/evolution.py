# Based on lib/evolution.py of "hamiltonian-simulation-for-hyperbolic-pde"
# by Y. Sato, R. Kondo, I. Hamamura, T. Onodera and N. Yamamoto (Apache
# License 2.0; Phys. Rev. Research 6, 033246 (2024)).  Modified here:
# projector characters (s00/s11) are supported as 0/1-controls of the
# rotation, and get_bell_evolution_circ_strang (palindromic second-order
# product) was added.
import cmath


from qiskit import QuantumCircuit


class _ReversedTerms:
    """View of an operator with the term order reversed (for Strang)."""

    def __init__(self, operator):
        self.num_qubits = operator.num_qubits
        self.op_list = list(reversed(operator.op_list))
        self.coeffs = list(reversed(operator.coeffs))


def get_bell_evolution_circ_strang(operator, dt, *, barrier=False):
    """Second-order (Strang) variant of get_bell_evolution_circ.

    Palindromic product of the term exponentials with half time step each,
    exp(-i dt H) + O(dt^3) per step instead of O(dt^2), at twice the gate
    count of the first-order circuit.
    """
    circ = get_bell_evolution_circ(operator, dt / 2, barrier=barrier)
    circ.compose(get_bell_evolution_circ(_ReversedTerms(operator), dt / 2,
                                         barrier=barrier), inplace=True)
    return circ


def get_bell_evolution_circ(operator, dt, *, barrier=False):
    meas_circ_dict = {}
    coeffs_dict = {}
    num_qubits = operator.num_qubits
    circ = QuantumCircuit(num_qubits)

    for term_op, coeff in zip(operator.op_list, operator.coeffs):
        # Listing qubits where term_op acts on non-trivially.
        qubits_s01 = []
        qubits_s10 = []
        qubits_s00 = []
        qubits_s11 = []

        for i, op in enumerate(reversed(term_op)): # The "term op" is "reversed" to fit the endian to that of qiskit.
            if op == 'm': 
                qubits_s01.append(i)
            elif op == 'p':
                qubits_s10.append(i)
            elif op == 'z':
                qubits_s00.append(i)
            elif op == 'o':
                qubits_s11.append(i)
        
        # Since we assume that the operator is symmetric matrix,
        # we only focus on the term where the left most non-trivial operator is s01
        # to construct the circuit.
        if len(qubits_s01) > 0:
            if len(qubits_s10) == 0 or qubits_s01[-1] > qubits_s10[-1]:

                q_controls = []

                # Rotating basis to the Bell basis
                for i in reversed(qubits_s01[:-1]):
                    circ.cx(qubits_s01[-1], i)
                    circ.x(i) # flipping the qubit for applying the mcrz gate controlled by the 0 state.
                    q_controls.append(circ.qubits[i])
                for i in reversed(qubits_s10):
                    circ.cx(qubits_s01[-1], i)
                    q_controls.append(circ.qubits[i])
                # Projector qubits (s00/s11) restrict the rotation to their
                # subspace: plain controls, with s00 controlled by the 0 state.
                for i in qubits_s11:
                    q_controls.append(circ.qubits[i])
                for i in qubits_s00:
                    circ.x(i)
                    q_controls.append(circ.qubits[i])

                lam = cmath.phase(coeff)
                gam = abs(coeff)

                circ.p(lam, qubits_s01[-1])
                circ.h(qubits_s01[-1])

                if barrier:
                    circ.barrier()
                
                # rotation
                if len(q_controls) > 0:
                    circ.mcrz(2*dt*gam, q_controls, circ.qubits[qubits_s01[-1]])
                else:
                    circ.rz(2*dt*gam, circ.qubits[qubits_s01[-1]])

                if barrier:
                    circ.barrier()

                # uncomputation
                circ.h(qubits_s01[-1])
                circ.p(-lam, qubits_s01[-1])
                for i in qubits_s00:
                    circ.x(i)
                for i in qubits_s10:
                    circ.cx(qubits_s01[-1], i)
                for i in qubits_s01[:-1]:
                    circ.x(i) # flipping the qubit for applying the mcrz gate controlled by the 0 state.
                    circ.cx(qubits_s01[-1], i)
                
                if barrier:
                    circ.barrier()
                    circ.barrier()

    return circ