import numpy as np
from qiskit.circuit.library import RYGate
from qiskit.circuit.library import ExactReciprocalGate

# ==============================================================================
# Eigenvalue inversion oracles
# ==============================================================================

def classical_eig_inversion_oracle(
    circ,
    qpe_register,
    ancilla_qubit,
    t0: float,
    eigs: np.ndarray,
    C: float,
    unwrap_phase: bool = False,
    lam_floor: float = 1e-12,
):
    """
    Classical eigenvalue inversion oracle for the HHL algorithm.
    This oracle is used to invert the eigenvalues of the matrix A using 
    classically calculated eigenvalues.
    
    The oracle is appended to the circuit as a sequence of controlled-RY gates.
    The controlled-RY gates are controlled on the qubits in the qpe_register,
    and the target is the ancilla_qubit.
    The angle of the RY gate is calculated based on the eigenvalue of the matrix A.
    """
    m = len(qpe_register)
    eigs = np.real_if_close(eigs)
    eigs = np.real(eigs)

    for k in range(2**m):
        phi = k / (2**m)
        if unwrap_phase and phi >= 0.5:
            phi -= 1.0
        lam_est = abs((2*np.pi*phi) / t0)
        lam_est = max(lam_est, lam_floor)

        # Find the nearest true eigenvalue (keeping its sign)
        lam = eigs[np.argmin(np.abs(eigs - lam_est))]

        ratio = C / lam
        # Clamp between -1.0 and 1.0 to ensure arcsin is valid
        ratio = max(min(ratio, 1.0), -1.0) 
        theta = 2 * np.arcsin(ratio)

        ctrl_state = format(k, f"0{m}b")
        mc_ry = RYGate(theta).control(m, ctrl_state=ctrl_state)
        circ.append(mc_ry, list(qpe_register) + [ancilla_qubit])


def quantum_eig_inversion_oracle(
):
    """
    Quantum eigenvalue inversion oracle for the HHL algorithm.
    This oracle is used to invert the eigenvalues of the matrix A using 
    the qiskit ExactReciprocalGate method.
    """
    pass

# ==============================================================================

def dynamic_t0():
    pass

def C_factor():
    pass