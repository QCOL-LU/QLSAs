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
        #lam = eigs[np.argmin(np.abs(eigs - lam_est))]
        lam = lam_est

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