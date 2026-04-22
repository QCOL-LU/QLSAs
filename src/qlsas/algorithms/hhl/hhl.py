from __future__ import annotations

from typing import Optional
import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import HamiltonianGate
from qiskit.synthesis.qft import synth_qft_full

from qlsas.algorithms.base import QLSA
from qlsas.readout.base import QLSACircuit
from qlsas.algorithms.hhl.eig_oracles import EigOracle, ClassicalEigOracle
from qlsas.algorithms.hhl.hhl_helpers import dynamic_t0, C_factor


class HHL(QLSA):
    """HHL (Harrow-Hassidim-Lloyd) quantum linear systems algorithm.

    This class builds the *core* HHL circuit: state preparation, forward QPE,
    eigenvalue inversion, ancilla measurement, and inverse QPE.  It does **not**
    include readout — that is handled by a :class:`~qlsas.readout.base.Readout`
    strategy passed to the solver.

    Parameters
    ----------
    num_qpe_qubits : int
        Number of qubits for the phase-estimation register.
    eig_oracle : EigOracle, optional
        Eigenvalue inversion strategy.  Defaults to
        :class:`~qlsas.algorithms.hhl.eig_oracles.ClassicalEigOracle`.

    Examples
    --------
    ::

        from qlsas.algorithms.hhl import HHL, ClassicalEigOracle, UnaryEigOracle

        hhl = HHL(num_qpe_qubits=4)                          # default oracle
        hhl = HHL(num_qpe_qubits=4, eig_oracle=UnaryEigOracle())  # UCRy oracle
    """

    def __init__(
        self,
        num_qpe_qubits: int,
        eig_oracle: Optional[EigOracle] = None,
    ) -> None:
        super().__init__()
        self.num_qpe_qubits = num_qpe_qubits
        if eig_oracle is not None and not isinstance(eig_oracle, EigOracle):
            raise TypeError(
                f"eig_oracle must be an EigOracle instance, got {type(eig_oracle).__name__!r}. "
                "Use ClassicalEigOracle(), QuantumEigOracle(), or UnaryEigOracle()."
            )
        self.eig_oracle: EigOracle = eig_oracle or ClassicalEigOracle()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_circuit(
        self,
        A: np.ndarray,
        b: np.ndarray,
        state_prep,
        *,
        t0: Optional[float] = None,
        C: Optional[float] = None,
    ) -> QLSACircuit:
        """Build the core HHL circuit (no readout measurements).

        Parameters
        ----------
        A : np.ndarray
            Hermitian coefficient matrix.
        b : np.ndarray
            Unit-norm right-hand-side vector.
        state_prep : StatePrep
            Strategy for loading |b⟩ into the circuit.
        t0 : float, optional
            Time parameter for the controlled Hamiltonian.  Auto-computed if *None*.
        C : float, optional
            Scaling factor for eigenvalue inversion.  Auto-computed if *None*.

        Returns
        -------
        QLSACircuit
            The core circuit plus register metadata.  The ``params`` field
            contains ``{"t0": t0, "C": C}`` for inspection by downstream
            components.
        """
        self._validate_inputs(A, b)

        t0_val = dynamic_t0(A) if t0 is None else t0
        C_val = C_factor(A) if C is None else C

        return self._build_core_circuit(A, b, state_prep, t0_val, C_val)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(A: np.ndarray, b: np.ndarray) -> None:
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be a square matrix, got shape {A.shape}")
        if b.ndim != 1:
            raise ValueError(f"b must be a vector, got shape {b.shape}")
        if A.shape[0] != b.shape[0]:
            raise ValueError(
                f"Dimension mismatch: A is {A.shape[0]}x{A.shape[1]}, "
                f"but b has shape {b.shape}"
            )
        if not np.allclose(A, A.T.conjugate()):
            raise ValueError("A must be Hermitian")
        if not np.isclose(np.linalg.norm(b), 1):
            raise ValueError(
                f"b should have unit norm, instead has norm: {np.linalg.norm(b)}"
            )

    # ------------------------------------------------------------------
    # Core circuit construction
    # ------------------------------------------------------------------

    def _build_core_circuit(
        self,
        A: np.ndarray,
        b: np.ndarray,
        state_prep,
        t0: float,
        C: float,
    ) -> QLSACircuit:
        """Build the core HHL circuit shared among all readout methods.

        Includes state preparation, forward QPE, eigenvalue inversion,
        ancilla measurement, and inverse QPE (uncomputation).
        """
        data_register_size = int(math.log2(len(b)))

        # Quantum registers
        ancilla_flag_register = QuantumRegister(1, name="ancilla_flag_register")
        qpe_register = QuantumRegister(self.num_qpe_qubits, name="qpe_register")
        b_to_x_register = QuantumRegister(data_register_size, name="b_to_x_register")

        # Classical register for ancilla measurement
        ancilla_flag_result = ClassicalRegister(1, name="ancilla_flag_result")

        circ = QuantumCircuit(
            ancilla_flag_register,
            qpe_register,
            b_to_x_register,
            ancilla_flag_result,
        )
        circ.name = f"HHL {len(b)} by {len(b)}"

        # 1. State Preparation
        circ.compose(state_prep.load_state(b), b_to_x_register, inplace=True)
        circ.barrier()

        # 2. Forward QPE
        self._apply_qpe(circ, A, qpe_register, b_to_x_register, t0, inverse=False)
        circ.barrier()

        # 3. Eigenvalue-based rotation
        self._apply_eig_oracle(circ, qpe_register, ancilla_flag_register[0], A, t0, C)
        circ.barrier()

        # 4. Measure ancilla flag
        circ.measure(ancilla_flag_register, ancilla_flag_result)
        circ.barrier()

        # 5. Uncomputation (inverse QPE)
        self._apply_qpe(circ, A, qpe_register, b_to_x_register, t0, inverse=True)
        circ.barrier()

        return QLSACircuit(
            circuit=circ,
            solution_register=b_to_x_register,
            ancilla_register=ancilla_flag_register,
            ancilla_creg=ancilla_flag_result,
            params={"t0": t0, "C": C},
        )

    # ------------------------------------------------------------------
    # QPE
    # ------------------------------------------------------------------

    def _apply_qpe(
        self,
        circ: QuantumCircuit,
        A: np.ndarray,
        qpe_register: QuantumRegister,
        target_register: QuantumRegister,
        t0: float,
        inverse: bool = False,
    ) -> None:
        """Apply Phase Estimation (or its inverse) for HHL."""
        if not inverse:
            circ.h(qpe_register)
            circ.barrier()

            for i in range(len(qpe_register)):
                time = t0 * (2**i)
                U = HamiltonianGate(A, -time, label=f"H_{i}")
                G = U.control(1)
                qubits = [qpe_register[i]] + target_register[:]
                circ.append(G, qubits)

            circ.barrier()
            iqft_circ = synth_qft_full(
                len(qpe_register),
                approximation_degree=0,
                do_swaps=True,
                inverse=True,
                name="IQFT",
            )
            circ.compose(iqft_circ, qpe_register, inplace=True)
        else:
            qft_circ = synth_qft_full(
                len(qpe_register),
                approximation_degree=0,
                do_swaps=True,
                inverse=False,
                name="QFT",
            )
            circ.compose(qft_circ, qpe_register, inplace=True)
            circ.barrier()

            for i in reversed(range(len(qpe_register))):
                time = t0 * (2**i)
                U = HamiltonianGate(A, time, label=f"H_{i}")
                G = U.control(1)
                qubits = [qpe_register[i]] + target_register[:]
                circ.append(G, qubits)

            circ.barrier()
            circ.h(qpe_register)

    # ------------------------------------------------------------------
    # Eigenvalue inversion
    # ------------------------------------------------------------------

    def _apply_eig_oracle(
        self,
        circ: QuantumCircuit,
        qpe_register: QuantumRegister,
        ancilla_qubit,
        A: np.ndarray,
        t0: float,
        C: float,
    ) -> None:
        """Delegate to the selected :class:`~qlsas.algorithms.hhl.eig_oracles.EigOracle`."""
        self.eig_oracle.apply(circ, qpe_register, ancilla_qubit, A, t0, C)
