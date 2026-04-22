"""Base abstractions for QLSA circuit metadata and readout strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


@dataclass
class QLSACircuit:
    """Rich return type from QLSA.build_circuit().

    Carries the core quantum circuit together with the register metadata that
    downstream components (readout, post-processing) need to do their job
    without hard-coding register names.
    """

    circuit: QuantumCircuit
    """The quantum circuit *before* any readout measurements are added."""

    solution_register: QuantumRegister
    """Register that will contain the solution state |x>."""

    ancilla_register: QuantumRegister
    """Single-qubit flag register (ancilla = 1 signals success)."""

    ancilla_creg: ClassicalRegister
    """Classical register already wired to measure the ancilla flag."""

    params: dict = field(default_factory=dict)
    """Algorithm-specific parameters computed during circuit construction.

    For HHL this contains ``{"t0": float, "C": float}`` so that downstream
    components (e.g. resource estimators) can inspect the values that were
    used without reaching into the algorithm object.
    """


class Readout(ABC):
    """Strategy that appends readout operations to a QLSA circuit and
    knows how to interpret the resulting measurement outcomes.
    """

    @property
    @abstractmethod
    def register_names(self) -> list[str]:
        """Ordered list of classical register names used by this readout.

        The order matches the bit ordering expected by :meth:`process`:
        the *last* name's bits appear as the LSBs (rightmost characters) of
        each bitstring returned by
        :meth:`~qlsas.measurement_result.MeasurementResult.get_counts`.
        """
        ...

    @abstractmethod
    def apply(
        self,
        qlsa_circuit: QLSACircuit,
        *,
        state_prep=None,
    ) -> QuantumCircuit:
        """Append measurement / readout gates to *qlsa_circuit* and return
        the complete ``QuantumCircuit`` ready for transpilation.

        Parameters
        ----------
        qlsa_circuit : QLSACircuit
            The core circuit produced by a QLSA algorithm.
        state_prep : StatePrep, optional
            State preparation strategy.  Required by readouts (e.g.
            :class:`~qlsas.readout.swap_test.SwapTestReadout`) that need to
            load a reference vector into an auxiliary register.
        """
        ...

    @abstractmethod
    def process(
        self,
        result: Any,
        A: np.ndarray,
        b: np.ndarray,
        verbose: bool = True,
    ) -> tuple:
        """Post-process execution results and return the solution.

        Parameters
        ----------
        result : MeasurementResult
            Wrapped measurement result from :class:`~qlsas.executer.Executer`.
        A : np.ndarray
            Coefficient matrix (needed for residual / norm computation).
        b : np.ndarray
            Right-hand-side vector (unit norm).
        verbose : bool
            Whether to print diagnostic information.

        The concrete return type depends on the readout strategy
        (e.g. a solution vector for tomography, a scalar for swap test).
        """
        ...
