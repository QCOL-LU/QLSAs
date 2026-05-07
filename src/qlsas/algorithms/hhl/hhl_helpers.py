from typing import Optional

import numpy as np
from qiskit.circuit.library import RYGate
from qiskit.circuit.library import ExactReciprocalGate
from qiskit import QuantumCircuit, QuantumRegister

# ==============================================================================
# Eigenvalue inversion oracles
# ==============================================================================

def mcry_eig_inversion(
    circ: QuantumCircuit,
    qpe_register: QuantumRegister,
    ancilla_qubit: QuantumRegister,
    A: np.ndarray,
    t0: float,
    C: float,
    *,
    has_negative_eigenvalues: Optional[bool] = None,
):
    """
    Multi-controlled RY (MCRY) eigenvalue-inversion routine.  Appends one
    ``m``-controlled RY per QPE basis state ``k ∈ [1, 2^m)``, with the
    angle derived directly from the phase implied by ``k`` (no snapping to
    true eigenvalues).  Two's-complement phase unwrapping is applied only
    when ``A`` has negative eigenvalues; for SPD matrices unwrap would
    misinterpret QPE leakage on ``k >= 2**(m-1)`` as spurious negative
    eigenvalues and flip the sign of the rotation.

    Pass ``has_negative_eigenvalues`` to declare A's sign profile and skip
    the ``np.linalg.eigvalsh(A)`` fallback used when the flag is ``None``.
    """
    m = len(qpe_register)
    if has_negative_eigenvalues is None:
        has_negative_eigs = bool(np.any(np.linalg.eigvalsh(A) < -1e-12))
    else:
        has_negative_eigs = bool(has_negative_eigenvalues)

    for k in range(1, 2**m):  # k=0 corresponds to phase=0; no inversion
        phi = k / (2**m)
        if has_negative_eigs and phi >= 0.5:
            phi -= 1.0

        lam = (2 * np.pi * phi) / t0
        ratio = max(min(C / lam, 1.0), -1.0)
        theta = 2 * np.arcsin(ratio)

        ctrl_state = format(k, f"0{m}b")
        mc_ry = RYGate(theta).control(m, ctrl_state=ctrl_state)
        circ.append(mc_ry, list(qpe_register) + [ancilla_qubit])


def exact_reciprocal_eig_inversion(
    circ: QuantumCircuit,
    qpe_register: QuantumRegister,
    ancilla_qubit: QuantumRegister,
    A: np.ndarray,
    t0: float,
    C: float,
    *,
    has_negative_eigenvalues: Optional[bool] = None,
):
    """
    Eigenvalue inversion via Qiskit's :class:`ExactReciprocalGate`.

    Sets ``neg_vals`` based on whether A has any negative eigenvalues,
    because under ``neg_vals=True`` Qiskit treats ``k >= 2**(n-1)`` as
    negative phases — which would corrupt SPD problems where QPE leakage
    leaves amplitude in that range.  The gate uses
    ``nl = 2**n`` when ``neg_vals=False`` and ``nl = 2**(n-1)`` when
    ``neg_vals=True``, so the physical-to-gate scaling depends on the
    flag: ``S = C*t0/(2*pi)`` for the former, ``S = C*t0/pi`` for the
    latter.

    Pass ``has_negative_eigenvalues`` to declare A's sign profile and skip
    the ``np.linalg.eigvalsh(A)`` fallback used when the flag is ``None``.
    """
    num_qpe_qubits = len(qpe_register)
    if has_negative_eigenvalues is None:
        has_negative_eigs = bool(np.any(np.linalg.eigvalsh(A) < -1e-12))
    else:
        has_negative_eigs = bool(has_negative_eigenvalues)

    if has_negative_eigs:
        S = (C * t0) / np.pi
    else:
        S = (C * t0) / (2 * np.pi)

    recip_gate = ExactReciprocalGate(
        num_state_qubits=num_qpe_qubits,
        scaling=S,
        neg_vals=has_negative_eigs,
    )

    circ.append(recip_gate, list(qpe_register) + [ancilla_qubit])

def ucry_eig_inversion(
    circ: QuantumCircuit,
    qpe_register: QuantumRegister,
    ancilla_qubit,
    A: np.ndarray,
    t0: float,
    C: float,
    *,
    has_negative_eigenvalues: Optional[bool] = None,
):
    """
    Uniformly-controlled RY (UCRY) eigenvalue-inversion routine.
    Functionally equivalent to :func:`mcry_eig_inversion` (same rotation
    angle table, same unitary on every QPE basis state) but realises it
    as a recursive CNOT + RY tree (Möttönen et al., 2004), achieving
    ``O(2^m)`` circuit depth instead of ``O(m · 2^m)`` and using no
    ancilla qubits.

    Pass ``has_negative_eigenvalues`` to declare A's sign profile and skip
    the ``np.linalg.eigvalsh(A)`` fallback used when the flag is ``None``.
    """
    m = len(qpe_register)
    if has_negative_eigenvalues is None:
        has_negative_eigs = bool(np.any(np.linalg.eigvalsh(A) < -1e-12))
    else:
        has_negative_eigs = bool(has_negative_eigenvalues)

    thetas = np.zeros(2**m)  # thetas[0] = 0: phase=0 ⇒ no inversion
    for k in range(1, 2**m):
        phi = k / (2**m)
        if has_negative_eigs and phi >= 0.5:
            phi -= 1.0

        lam = (2 * np.pi * phi) / t0
        ratio = max(min(C / lam, 1.0), -1.0)
        thetas[k] = 2 * np.arcsin(ratio)

    _apply_ucry(circ, list(qpe_register), ancilla_qubit, thetas)


def _apply_ucry(circ, controls, target, thetas):
    """
    Recursively decompose a uniformly controlled RY into CNOT + RY pairs.

    When controls are in computational basis state |k>, applies RY(thetas[k])
    to *target*.  The recursion splits the angle table in half at each level,
    branching on the last qubit in *controls*:

        UCR_n  =  UCR_{n-1}(alpha)  ·  CX(ctrl[-1], tgt)
                · UCR_{n-1}(beta)   ·  CX(ctrl[-1], tgt)

    where  alpha_k = (theta_k + theta_{k+half}) / 2
           beta_k  = (theta_k - theta_{k+half}) / 2

    Gate count: 2^n RY  +  2^{n+1} - 2 CNOT   (n = len(controls))
    Depth:      O(2^n)
    Ancillas:   0
    """
    n = len(controls)

    if n == 0:
        if abs(thetas[0]) > 1e-15:
            circ.ry(float(thetas[0]), target)
        return

    half = len(thetas) // 2
    alphas = (thetas[:half] + thetas[half:]) / 2
    betas  = (thetas[:half] - thetas[half:]) / 2

    _apply_ucry(circ, controls[:-1], target, alphas)
    circ.cx(controls[-1], target)
    _apply_ucry(circ, controls[:-1], target, betas)
    circ.cx(controls[-1], target)


# ==============================================================================

def dynamic_t0(A: np.ndarray, buffer: float = 0.05) -> float:
    """
    Calculate the optimal time evolution parameter t0.
    Ensures that the largest eigenvalue does not cause the QPE phase 
    to wrap past pi or -pi (which would corrupt the sign bit).
    """
    eigs = np.linalg.eigvalsh(A)
    max_eig = np.max(np.abs(eigs))
    
    # Map to [-pi, pi] with a safety buffer so max eigenvalue doesn't alias
    t0 = (np.pi / max_eig) * (1 - buffer)
    return float(t0)

def C_factor(A: np.ndarray, scale: float = 0.9, zero_tol: float = 1e-12) -> float:
    """
    Calculate the optimal ancilla rotation scaling factor C.
    Must be strictly less than the absolute value of the smallest non-zero eigenvalue.
    """
    eigs = np.linalg.eigvalsh(A)
    abs_eigs = np.abs(eigs)
    
    # Filter out values extremely close to 0 (null space)
    min_eig = np.min(abs_eigs[abs_eigs > zero_tol])
    
    C = scale * min_eig
    return float(C)