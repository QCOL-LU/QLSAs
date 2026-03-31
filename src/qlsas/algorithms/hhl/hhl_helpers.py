import numpy as np
from qiskit.circuit.library import RYGate
from qiskit.circuit.library import ExactReciprocalGate
from qiskit import QuantumCircuit, QuantumRegister

# ==============================================================================
# Eigenvalue inversion oracles
# ==============================================================================

def classical_eig_inversion_oracle(
    circ: QuantumCircuit,
    qpe_register: QuantumRegister,
    ancilla_qubit: QuantumRegister,
    A: np.ndarray,
    t0: float,
    C: float,
    lam_floor: float = 1e-12,
):
    """
    Classical eigenvalue inversion oracle for the HHL algorithm.
    This oracle is used to invert the eigenvalues of the matrix A using 
    classically calculated eigenvalues. Automatically detects if A is 
    indefinite to apply two's complement phase unwrapping.
    """
    m = len(qpe_register)
    
    eigs = np.linalg.eigvalsh(A)
    eigs = np.real_if_close(eigs)
    eigs = np.real(eigs)
    
    # 1. Automatically detect if phase unwrapping is needed
    has_negative_eigs = bool(np.any(eigs < -1e-12))

    for k in range(2**m):
        phi = k / (2**m)
        
        # 2. Apply two's complement if matrix is indefinite
        if has_negative_eigs and phi >= 0.5:
            phi -= 1.0
            
        # 3. Calculate signed eigenvalue estimate from phase
        lam_est = (2 * np.pi * phi) / t0
        
        # Apply the floor while preserving the sign
        if abs(lam_est) < lam_floor:
            lam_est = lam_floor if lam_est >= 0 else -lam_floor

        # Find the nearest true eigenvalue (signed distance)
        lam = eigs[np.argmin(np.abs(eigs - lam_est))]
        #lam = lam_est

        ratio = C / lam
        # Clamp between -1.0 and 1.0 to ensure arcsin is valid
        ratio = max(min(ratio, 1.0), -1.0) 
        theta = 2 * np.arcsin(ratio)

        ctrl_state = format(k, f"0{m}b")
        mc_ry = RYGate(theta).control(m, ctrl_state=ctrl_state)
        circ.append(mc_ry, list(qpe_register) + [ancilla_qubit])


def quantum_eig_inversion_oracle(
    circ: QuantumCircuit,
    qpe_register: QuantumRegister,
    ancilla_qubit: QuantumRegister, 
    A: np.ndarray,
    t0: float,
    C: float,
):
    """
    Quantum eigenvalue inversion oracle for the HHL algorithm.
    This oracle is used to invert the eigenvalues of the matrix A using 
    the qiskit ExactReciprocalGate method.
    """
    num_qpe_qubits = len(qpe_register)

    # Check if A is indefinite/negative definite (has negative eigenvalues)
    eigs = np.linalg.eigvalsh(A)
    # Use a small tolerance to prevent floating-point noise from triggering True
    has_negative_eigs = bool(np.any(eigs < -1e-12))
    
    # Translate physical C into Qiskit's integer-based scaling factor S
    S = (C * t0) / (2 * np.pi)
    
    recip_gate = ExactReciprocalGate(
        num_state_qubits=num_qpe_qubits,
        scaling=S,
        neg_vals=has_negative_eigs
    )
    
    # Append the gate. ExactReciprocalGate expects the state qubits first, 
    # followed by the target ancilla qubit.
    circ.append(recip_gate, list(qpe_register) + [ancilla_qubit])

def unary_iteration_eig_inversion_oracle(
    circ: QuantumCircuit,
    qpe_register: QuantumRegister,
    ancilla_qubit,
    A: np.ndarray,
    t0: float,
    C: float,
    lam_floor: float = 1e-12,
):
    """
    Eigenvalue inversion oracle using a uniformly controlled RY (UCRy)
    decomposition.  Functionally equivalent to classical_eig_inversion_oracle
    but replaces the 2^m  m-controlled RY gates with a recursive CNOT + RY
    tree, achieving O(2^m) circuit depth instead of O(m · 2^m).

    Based on Möttönen et al., "Quantum Circuits for General Multiqubit
    Gates" (2004).  The technique is closely related to the unary-iteration
    scheme of Babbush et al. (arXiv:1805.03662) but requires no ancilla
    qubits.
    """
    m = len(qpe_register)

    eigs = np.linalg.eigvalsh(A)
    eigs = np.real_if_close(eigs)
    eigs = np.real(eigs)

    has_negative_eigs = bool(np.any(eigs < -1e-12))

    thetas = np.empty(2**m)
    for k in range(2**m):
        phi = k / (2**m)

        if has_negative_eigs and phi >= 0.5:
            phi -= 1.0

        lam_est = (2 * np.pi * phi) / t0

        if abs(lam_est) < lam_floor:
            lam_est = lam_floor if lam_est >= 0 else -lam_floor

        lam = eigs[np.argmin(np.abs(eigs - lam_est))]

        ratio = C / lam
        ratio = max(min(ratio, 1.0), -1.0)
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