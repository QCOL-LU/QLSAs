"""Eigenvalue-inversion oracle strategies for the HHL algorithm.

Each oracle is a lightweight strategy object that appends the appropriate
rotation circuit to a partially-built HHL circuit.  Pass one to
:class:`~qlsas.algorithms.hhl.hhl.HHL` at construction time::

    from qlsas.algorithms.hhl.eig_oracles import ClassicalEigOracle
    hhl = HHL(num_qpe_qubits=4, eig_oracle=ClassicalEigOracle())

Available oracles
-----------------
:class:`ClassicalEigOracle`
    Classically-computed, multi-controlled RY gates.  Accurate but deep.
:class:`QuantumEigOracle`
    Qiskit's ``ExactReciprocalGate``; compact for positive-definite matrices.
:class:`UnaryEigOracle`
    Uniformly-controlled RY (UCRy) decomposition.  O(2^m) depth, no ancilla.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

from qlsas.algorithms.hhl.hhl_helpers import (
    classical_eig_inversion_oracle,
    quantum_eig_inversion_oracle,
    unary_iteration_eig_inversion_oracle,
)


class EigOracle(ABC):
    """Abstract strategy for eigenvalue inversion in HHL.

    Implementations receive the partially-built circuit, the QPE and ancilla
    registers, and the problem parameters (*A*, *t0*, *C*), then append the
    appropriate rotation gates in-place.
    """

    @abstractmethod
    def apply(
        self,
        circ: QuantumCircuit,
        qpe_register: QuantumRegister,
        ancilla_qubit,
        A: np.ndarray,
        t0: float,
        C: float,
    ) -> None:
        """Append eigenvalue-inversion gates to *circ*.

        Parameters
        ----------
        circ : QuantumCircuit
            The circuit being built (mutated in place).
        qpe_register : QuantumRegister
            The QPE register whose state encodes the eigenvalue estimate.
        ancilla_qubit :
            The single ancilla qubit to rotate.
        A : np.ndarray
            The Hermitian coefficient matrix.
        t0 : float
            Hamiltonian evolution time (determines the QPE phase scaling).
        C : float
            Ancilla rotation scaling factor.
        """
        ...


class ClassicalEigOracle(EigOracle):
    """Classical eigenvalue inversion via multi-controlled RY gates.

    Computes exact eigenvalues with ``numpy`` and generates one
    multi-controlled rotation per QPE basis state.  Faithful for any
    Hermitian matrix (positive-definite, indefinite, or negative-definite)
    because it handles two's-complement phase unwrapping automatically.
    """

    def apply(
        self,
        circ: QuantumCircuit,
        qpe_register: QuantumRegister,
        ancilla_qubit,
        A: np.ndarray,
        t0: float,
        C: float,
    ) -> None:
        classical_eig_inversion_oracle(
            circ, qpe_register, ancilla_qubit, A=A, t0=t0, C=C
        )


class QuantumEigOracle(EigOracle):
    """Quantum eigenvalue inversion via Qiskit's ``ExactReciprocalGate``.

    Uses Qiskit's built-in reciprocal gate, which is compact and efficient
    for positive-definite matrices but requires the ``neg_vals`` flag for
    indefinite systems.
    """

    def apply(
        self,
        circ: QuantumCircuit,
        qpe_register: QuantumRegister,
        ancilla_qubit,
        A: np.ndarray,
        t0: float,
        C: float,
    ) -> None:
        quantum_eig_inversion_oracle(
            circ, qpe_register, ancilla_qubit, A=A, t0=t0, C=C
        )


class UnaryEigOracle(EigOracle):
    """Uniformly-controlled RY (UCRy) eigenvalue inversion oracle.

    Achieves the same rotation angles as :class:`ClassicalEigOracle` but
    uses a recursive CNOT + RY tree (Möttönen et al., 2004) that gives
    O(2^m) depth without additional ancilla qubits.
    """

    def apply(
        self,
        circ: QuantumCircuit,
        qpe_register: QuantumRegister,
        ancilla_qubit,
        A: np.ndarray,
        t0: float,
        C: float,
    ) -> None:
        unary_iteration_eig_inversion_oracle(
            circ, qpe_register, ancilla_qubit, A=A, t0=t0, C=C
        )
