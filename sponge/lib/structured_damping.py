"""Structured (poly(n)) circuit for the Schrodingerised damping factor.

In the Trotterised Schrodingerised sponge evolution the dissipative factor is

    U_damp(tau) = exp(+i tau D_eta (x) Sigma),

a diagonal unitary on (p register) (x) (system register). Synthesised as a
generic DiagonalGate it costs Theta(2^n) CX gates (n = n_p + n_sys). Two
exact structural facts make it polynomial:

1. eta is LINEAR in the p-register bits.  With N_p = 2^{n_p} points on
   [-p_max, p_max) the discrete Fourier frequencies are
       eta_k = alpha * kbar,  alpha = pi / p_max,
   where kbar is the two's-complement reading of the index bits,
       kbar = sum_{b < n_p-1} 2^b k_b  -  2^{n_p-1} k_{n_p-1}.
   Hence  exp(i tau D_eta (x) Sigma) = prod_b  C_{k_b}[ exp(i tau alpha w_b Sigma) ]
   with w_b = +-2^b: n_p commuting, singly-controlled diagonal factors.

2. Sigma is a SUM of one-dimensional profiles, block by block,
       Sigma = sum_blocks |b><b| (x) [ sigma_x^{(b)}(i) + sigma_y^{(b)}(j) ],
   and each 1D profile is piecewise polynomial of degree m on dyadic
   intervals (the layer strips).  A degree-m polynomial in the integer
   index j = sum 2^c j_c is a multilinear polynomial of degree <= m in the
   bits (j_c^2 = j_c), whose coefficients follow from the Moebius
   transform.  Each monomial a_S prod_{c in S} j_c, restricted to a dyadic
   strip (fixed high bits) and to a block-label sub-cube, and multiplied by
   one p bit, is a single multi-controlled phase gate.

The decomposition (`dyadic_poly_pieces`) is adaptive and EXACT for any
profile: it fits a degree-<=m multilinear polynomial on a dyadic interval
and bisects where the fit fails (clipped nodes, stagger offsets), down to
single nodes if needed.  The resulting circuit equals the DiagonalGate to
machine precision (tests/test_structured_damping.py).

CAUTION (qiskit 1.4.5): transpile the flag-ancilla variant with
optimization_level=0.  At levels >= 1 the transpiler changes the unitary of
the full Schrodingerised circuit (measured: 2e-2 -- 1.7e-1 state error on a
1D sponge), although the circuit itself is exact (Statevector evolution
and level-0 transpilation agree with the dense diagonal to 1e-15).  The
ancilla-free variant is unaffected.

Qubit conventions follow Schrodingerisation.get_circuit: system qubits
0..n_sys-1 little-endian with system index s = block*M + ix*Ny + iy
(y bits lowest, then x bits, then block-label bits), p qubits
n_sys..n_sys+n_p-1 with index k = sum 2^b k_b.
"""

from itertools import combinations

import numpy as np

# --------------------------------------------------------------------- #
# 1D profile -> dyadic multilinear pieces                                #
# --------------------------------------------------------------------- #


def _moebius_coeffs(block, q, m):
    """Multilinear coefficients a_S (|S| <= m) of the block values, read as
    a function of the q low bits, via the Moebius transform."""
    coeffs = {}
    for r in range(m + 1):
        for S in combinations(range(q), r):
            a = 0.0
            for t in range(r + 1):
                for T in combinations(S, t):
                    idx = sum(1 << c for c in T)
                    a += (-1) ** (r - t) * block[idx]
            coeffs[S] = a
    return coeffs


def _eval_multilinear(coeffs, q):
    x = np.arange(2**q)
    out = np.zeros(2**q)
    for S, a in coeffs.items():
        mask = np.ones(2**q, dtype=bool)
        for c in S:
            mask &= ((x >> c) & 1).astype(bool)
        out[mask] += a
    return out


def dyadic_poly_pieces(f, m=2, tol=1e-12):
    """Exact decomposition of a length-2^n profile into dyadic pieces.

    Returns a list of (lo, q, coeffs): on indices lo .. lo+2^q-1 (lo a
    multiple of 2^q, i.e. the high n-q bits fixed to lo >> q) the profile
    equals sum_S coeffs[S] prod_{c in S} bit_c(index - lo); zero pieces are
    omitted, and terms with |a_S| <= tol are dropped.
    """
    f = np.asarray(f, dtype=float)
    n = int(round(np.log2(len(f))))
    if 2**n != len(f):
        raise ValueError("profile length must be a power of two")
    scale = max(1.0, np.abs(f).max())
    pieces = []

    def rec(lo, q):
        block = f[lo:lo + 2**q]
        if np.abs(block).max() <= tol * scale:
            return
        co = _moebius_coeffs(block, q, min(m, q))
        if np.abs(_eval_multilinear(co, q) - block).max() <= tol * scale:
            co = {S: a for S, a in co.items() if abs(a) > tol * scale}
            pieces.append((lo, q, co))
            return
        rec(lo, q - 1)
        rec(lo + 2**(q - 1), q - 1)

    rec(0, n)
    return pieces


# --------------------------------------------------------------------- #
# Sigma (block diagonal) -> separable components                         #
# --------------------------------------------------------------------- #


def separable_components(sigma_blocks, num_blocks, num_qubits_axes):
    """Split the system diagonal Sigma into per-block, per-axis 1D profiles.

    sigma_blocks : length num_blocks * prod(2^n_a) diagonal of Sigma.
    num_qubits_axes : (n_x, n_y) or (n_x, n_y, n_z), most significant first.

    Returns a list of (axis, profile, sorted block tuple) with identical
    profiles grouped over blocks; raises if Sigma is not exactly a sum of
    1D profiles on every block (it always is for the sponge layers).
    """
    sizes = [2**n for n in num_qubits_axes]
    M = int(np.prod(sizes))
    sb = np.asarray(sigma_blocks, dtype=float).reshape(num_blocks, *sizes)
    centre = tuple(s // 2 for s in sizes)
    groups = {}
    for b in range(num_blocks):
        T = sb[b]
        if not T.any():
            continue
        if abs(T[centre]) > 0:
            raise ValueError("Sigma must vanish at the grid centre")
        recon = np.zeros_like(T)
        profs = []
        for a, s in enumerate(sizes):
            idx = list(centre)
            idx[a] = slice(None)
            prof = T[tuple(idx)].copy()
            profs.append(prof)
            shape = [1] * len(sizes)
            shape[a] = s
            recon = recon + prof.reshape(shape)
        if np.abs(recon - T).max() > 1e-12 * max(1.0, np.abs(T).max()):
            raise ValueError(f"block {b}: Sigma is not a sum of 1D profiles")
        for a, prof in enumerate(profs):
            if not prof.any():
                continue
            key = (a, prof.tobytes())
            groups.setdefault(key, (a, prof, []))[2].append(b)
    return [(a, prof, tuple(sorted(bl))) for a, prof, bl in groups.values()]


def _dyadic_cover(blocks, num_label_bits):
    """Cover a set of block labels by aligned dyadic ranges (sub-cubes)."""
    s = set(blocks)
    out = []

    def rec(lo, q):
        rng = set(range(lo, lo + 2**q))
        if rng <= s:
            out.append((lo, q))
        elif rng & s:
            rec(lo, q - 1)
            rec(lo + 2**(q - 1), q - 1)

    rec(0, num_label_bits)
    return out


# --------------------------------------------------------------------- #
# circuit                                                                #
# --------------------------------------------------------------------- #


def p_bit_weights(num_qubits_p):
    w = [float(2**b) for b in range(num_qubits_p)]
    w[-1] = -float(2**(num_qubits_p - 1))
    return np.array(w)


def damping_terms(sch, sigma_blocks, num_blocks, num_qubits_axes, m=2):
    """Abstract term list of exp(i tau D_eta (x) Sigma) / tau.

    Each term is (angle_per_tau, fixed) with `fixed` a dict
    {qubit: required bit}; the gate multiplies the phase
    exp(i tau * angle_per_tau) onto basis states matching every fixed bit.
    """
    n_axes = list(num_qubits_axes)
    n_lab = int(round(np.log2(num_blocks)))
    n_sys = n_lab + sum(n_axes)
    # axis a's lowest qubit: axes stored most-significant first
    axis_q0 = [sum(n_axes[a + 1:]) for a in range(len(n_axes))]
    lab_q0 = sum(n_axes)
    alpha = np.pi / sch.p_max
    kbar = np.round(sch.eta / alpha).astype(int)
    k = np.arange(sch.Np)
    w = p_bit_weights(sch.num_qubits_p)
    assert np.array_equal(kbar, ((k[:, None] >> np.arange(sch.num_qubits_p)) & 1) @ w), \
        "eta is not the two's-complement-linear grid"

    base = []      # (coefficient, fixed system bits)
    for a, prof, blocks in separable_components(sigma_blocks, num_blocks, n_axes):
        na = n_axes[a]
        for llo, lq in _dyadic_cover(blocks, n_lab):
            lab_fixed = {lab_q0 + c: (llo >> c) & 1 for c in range(lq, n_lab)}
            for lo, q, coeffs in dyadic_poly_pieces(prof, m=m):
                ax_fixed = {axis_q0[a] + c: (lo >> c) & 1 for c in range(q, na)}
                for S, coef in coeffs.items():
                    fixed = dict(lab_fixed)
                    fixed.update(ax_fixed)
                    fixed.update({axis_q0[a] + c: 1 for c in S})
                    base.append((coef, fixed))
    terms = []
    for b in range(sch.num_qubits_p):
        for coef, fixed in base:
            f = dict(fixed)
            f[n_sys + b] = 1
            terms.append((alpha * w[b] * coef, f))
    return terms, n_sys


def damping_circuit(sch, sigma_blocks, num_blocks, num_qubits_axes, tau,
                    m=2, flag_ancilla=False):
    """QuantumCircuit for exp(+i tau D_eta (x) Sigma), exact.

    Layout: system qubits 0..n_sys-1, p qubits n_sys.., and, if
    flag_ancilla, one extra clean ancilla (last qubit) used to compute the
    strip/label predicate once per piece instead of re-controlling every
    monomial on it.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PhaseGate, MCXGate

    terms, n_sys = damping_terms(sch, sigma_blocks, num_blocks,
                                 num_qubits_axes, m=m)
    n = n_sys + sch.num_qubits_p
    circ = QuantumCircuit(n + (1 if flag_ancilla else 0))

    def mcphase(theta, fixed):
        qs = sorted(fixed)
        target = qs[-1]            # p qubit (always fixed to 1, highest index)
        ctrls = qs[:-1]
        if not ctrls:
            circ.p(theta, target)
            return
        state = sum(fixed[q] << i for i, q in enumerate(ctrls))
        circ.append(PhaseGate(theta).control(len(ctrls), ctrl_state=state),
                    ctrls + [target])

    if not flag_ancilla:
        for ang, fixed in terms:
            mcphase(tau * ang, fixed)
        return circ

    # Flag variant: per piece, compute the predicate (label sub-cube AND
    # strip prefix) into the ancilla once, control the monomials on it.
    anc = n
    n_axes = list(num_qubits_axes)
    n_lab = int(round(np.log2(num_blocks)))
    axis_q0 = [sum(n_axes[a + 1:]) for a in range(len(n_axes))]
    lab_q0 = sum(n_axes)
    alpha = np.pi / sch.p_max
    w = p_bit_weights(sch.num_qubits_p)
    for a, prof, blocks in separable_components(sigma_blocks, num_blocks, n_axes):
        na = n_axes[a]
        for llo, lq in _dyadic_cover(blocks, n_lab):
            for lo, q, coeffs in dyadic_poly_pieces(prof, m=m):
                pred = {lab_q0 + c: (llo >> c) & 1 for c in range(lq, n_lab)}
                pred.update({axis_q0[a] + c: (lo >> c) & 1 for c in range(q, na)})
                pq = sorted(pred)
                use_flag = len(pq) > 1
                if use_flag:
                    st = sum(pred[qq] << i for i, qq in enumerate(pq))
                    circ.append(MCXGate(len(pq), ctrl_state=st), pq + [anc])
                for b in range(sch.num_qubits_p):
                    for S, coef in coeffs.items():
                        fixed = {axis_q0[a] + c: 1 for c in S}
                        if use_flag:
                            fixed[anc] = 1
                        else:
                            fixed.update(pred)
                        fixed[n_sys + b] = 1
                        theta = tau * alpha * w[b] * coef
                        qs = [qq for qq in sorted(fixed) if qq != n_sys + b]
                        if not qs:
                            circ.p(theta, n_sys + b)
                        else:
                            stt = sum(fixed[qq] << i for i, qq in enumerate(qs))
                            circ.append(PhaseGate(theta).control(len(qs), ctrl_state=stt),
                                        qs + [n_sys + b])
                if use_flag:
                    circ.append(MCXGate(len(pq), ctrl_state=st), pq + [anc])
    return circ


def get_circuit_structured(sch, z0, T, num_steps, system_circuit_factory,
                           sigma_blocks, num_blocks, num_qubits_axes,
                           m=2, flag_ancilla=False):
    """Schrodingerisation.get_circuit(trotter_order=2) with the generic
    DiagonalGate damping replaced by the structured damping circuit.
    Same layout and metadata; with flag_ancilla the ancilla is the last
    qubit and returns to |0> (the statevector is then sliced on it)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import QFTGate

    n_sys = int(np.log2(sch.H1.shape[0]))
    n = n_sys + sch.num_qubits_p
    nq = n + (1 if flag_ancilla else 0)
    sys_q = list(range(n_sys))
    p_q = list(range(n_sys, n))
    W0 = sch.initial_w(np.asarray(z0, dtype=complex)).ravel()
    norm0 = np.linalg.norm(W0)
    dt = T / num_steps

    circ = QuantumCircuit(nq)
    circ.initialize(W0 / norm0, list(range(n)))
    circ.append(QFTGate(sch.num_qubits_p).inverse(), p_q)
    full = damping_circuit(sch, sigma_blocks, num_blocks, num_qubits_axes,
                           dt, m=m, flag_ancilla=flag_ancilla)
    half = damping_circuit(sch, sigma_blocks, num_blocks, num_qubits_axes,
                           dt / 2, m=m, flag_ancilla=flag_ancilla)
    wave = system_circuit_factory(dt)
    circ.compose(half, inplace=True)
    for s in range(num_steps):
        circ.compose(wave, qubits=sys_q, inplace=True)
        circ.compose(full if s < num_steps - 1 else half, inplace=True)
    circ.append(QFTGate(sch.num_qubits_p), p_q)
    circ.metadata = {'norm0': norm0, 'n_sys': n_sys}
    return circ
