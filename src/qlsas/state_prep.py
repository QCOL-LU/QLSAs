"""State preparation strategies for loading classical vectors into quantum registers."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import StatePreparation as QiskitStatePreparation


class StatePrep(ABC):
    """Abstract base for state preparation strategies.

    Implementations encode a classical unit-norm vector into a quantum
    register and return the corresponding ``QuantumCircuit``.

    To add a new strategy, subclass ``StatePrep`` and implement
    :meth:`load_state`.  Example::

        class MyStatePrep(StatePrep):
            def load_state(self, state: np.ndarray) -> QuantumCircuit:
                ...
    """

    @abstractmethod
    def load_state(self, state: np.ndarray) -> QuantumCircuit:
        """Return a circuit that prepares ``|state>`` in a fresh register.

        Parameters
        ----------
        state : np.ndarray
            Unit-norm state vector whose length must be a power of two.

        Returns
        -------
        QuantumCircuit
            A circuit acting on ``log₂(len(state))`` qubits.
        """
        ...


class DefaultStatePrep(StatePrep):
    """Standard amplitude-encoding using Qiskit's ``StatePreparation`` gate.

    Uses a unitary (reset-free) decomposition, which is required by backends
    that do not support mid-circuit resets (e.g. IBM Nighthawk).
    """

    def load_state(self, state: np.ndarray) -> QuantumCircuit:
        if not math.log2(len(state)).is_integer():
            raise ValueError(f"State must be a power of two: {len(state)}")
        if not np.isclose(np.linalg.norm(state), 1):
            raise ValueError(
                f"State must have unit norm, instead has norm: {np.linalg.norm(state)}"
            )

        register_size = int(math.log2(len(state)))
        b_register = QuantumRegister(register_size)
        circuit = QuantumCircuit(b_register)
        sp = QiskitStatePreparation(list(state), normalize=True)
        circuit.append(sp, b_register)
        return circuit
