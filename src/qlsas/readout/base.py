"""Base abstractions for QLSA circuit metadata and readout strategies."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


@dataclass
class SuccessCriterion:
    """Defines which classical-register values mark a shot as 'successful'.

    Each entry pairs a classical register with the bit pattern that must hold
    for the shot to count as success. HHL uses one register with required
    value ``"1"``; QSVT-based solvers can use multiple registers with
    arbitrary required patterns (e.g. ``["1", "11"]`` for two success
    registers of widths 1 and 2).

    Convention: ``registers[0]`` is assumed to occupy the **rightmost**
    (LSB) characters of the joined measurement bitstring; ``registers[1]``
    sits to its left, and so on. Readouts must construct their
    ``register_names`` such that the success registers come first (i.e.
    end up rightmost) in the join order.
    """

    registers: list[ClassicalRegister]
    required_values: list[str]

    def __post_init__(self) -> None:
        if len(self.registers) != len(self.required_values):
            raise ValueError(
                f"registers ({len(self.registers)}) and required_values "
                f"({len(self.required_values)}) length mismatch."
            )
        for reg, val in zip(self.registers, self.required_values):
            if len(val) != len(reg):
                raise ValueError(
                    f"required_value {val!r} length {len(val)} does not match "
                    f"register {reg.name!r} width {len(reg)}."
                )

    @property
    def register_names(self) -> list[str]:
        return [r.name for r in self.registers]

    @property
    def total_width(self) -> int:
        return sum(len(r) for r in self.registers)

    def matches(self, bitstring: str) -> bool:
        """Check whether *bitstring* satisfies all success conditions.

        Assumes the success registers occupy the rightmost characters of the
        bitstring, with ``registers[0]`` at the very right.
        """
        pos = 0
        for reg, required in zip(self.registers, self.required_values):
            width = len(reg)
            slice_val = (
                bitstring[-(pos + width):] if pos == 0
                else bitstring[-(pos + width):-pos]
            )
            if slice_val != required:
                return False
            pos += width
        return True


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
    """Single-qubit flag register (ancilla = 1 signals success).

    Retained for HHL-era circuit-construction metadata. New code should
    prefer :attr:`success_criterion` for post-selection logic.
    """

    ancilla_creg: ClassicalRegister
    """Classical register already wired to measure the ancilla flag.

    Retained for backward compatibility. The same register also appears in
    :attr:`success_criterion`.
    """

    success_criterion: Optional[SuccessCriterion] = None
    """Defines what bitstring pattern marks a successful shot.

    If ``None``, downstream post-selection helpers fall back to the legacy
    HHL convention (``key[-1] == "1"``). Algorithms should populate this so
    that future multi-ancilla solvers (e.g. QSVT) can plug in cleanly.
    """

    params: dict = field(default_factory=dict)
    """Algorithm-specific parameters computed during circuit construction.

    For HHL this contains ``{"t0": float, "C": float}`` so that downstream
    components (e.g. resource estimators) can inspect the values that were
    used without reaching into the algorithm object.
    """


@dataclass
class TomographyResult:
    """Uniform return type for tomography readouts.

    Attributes
    ----------
    direction : np.ndarray
        Unit-norm reconstructed solution direction.
    alpha : float
        Least-squares scale that fits ``A·(α·direction) ≈ b`` (computed via
        :func:`~qlsas.post_processor.norm_estimation`). May be ``nan`` when
        the readout's caller provides A and b that are not the original
        problem (e.g. inside iterative refinement, where the IR loop computes
        its own scale).
    success_rate : float
        Fraction of shots accepted by the success criterion.
    residual : float
        ``‖b − A·(α·direction)‖`` — the residual that ``α·direction``
        achieves on the (A, b) the readout was given.
    metadata : dict
        Strategy-specific diagnostics.

    For backward compatibility with code that does
    ``solution, success_rate, residual = readout.process(...)``, this
    dataclass is iterable as a 3-tuple yielding
    ``(direction, success_rate, residual)``. New code should access fields
    by name and use :attr:`scaled` when the physically-scaled vector is
    needed.
    """

    direction: np.ndarray
    alpha: float
    success_rate: float
    residual: float
    metadata: dict = field(default_factory=dict)

    @property
    def scaled(self) -> np.ndarray:
        if math.isnan(self.alpha):
            return self.direction
        return self.alpha * self.direction

    def __iter__(self) -> Iterator[Any]:
        yield self.direction
        yield self.success_rate
        yield self.residual

    def __len__(self) -> int:
        # Matches the 3-element __iter__ above so callers using
        # ``len(result)`` see the same shape they would from tuple unpacking.
        return 3


class Readout(ABC):
    """Strategy that appends readout operations to a QLSA circuit and
    knows how to interpret the resulting measurement outcomes.
    """

    @property
    @abstractmethod
    def register_names(self) -> list[str]:
        """Ordered list of classical register names used by this readout.

        The order matches the bit ordering expected by :meth:`process`:
        the *first* name's bits appear as the LSBs (rightmost characters) of
        each bitstring returned by
        :meth:`~qlsas.measurement_result.MeasurementResult.get_counts`.
        Concretely, place success-criterion registers first so they end up at
        the rightmost positions where :meth:`SuccessCriterion.matches` reads
        them.
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
    ) -> Any:
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

        Tomography readouts return a :class:`TomographyResult`; non-tomography
        readouts (e.g. swap test) return a strategy-specific tuple.
        """
        ...


class MultiCircuitReadout(Readout):
    """Strategy that submits multiple circuits per solve.

    Concrete subclasses implement :meth:`build_circuits` (returning all
    circuits in execution order) and :meth:`combine_results` (collapsing
    the per-circuit measurement outcomes into a :class:`TomographyResult`).

    The single-circuit ``apply`` / ``process`` methods inherited from
    :class:`Readout` are not meaningful for multi-circuit strategies and
    are provided here as concrete implementations that raise
    :class:`NotImplementedError`. The solver dispatches on the
    ``MultiCircuitReadout`` marker so the single-circuit code paths are
    never reached for these readouts.
    """

    @abstractmethod
    def build_circuits(self, qlsa_circuit: "QLSACircuit") -> list[QuantumCircuit]:
        """Return all circuits this readout needs, in execution order.

        Conventionally the first entry is the base / Z-basis circuit and the
        rest are basis variants or auxiliary measurements.
        """
        ...

    @abstractmethod
    def combine_results(
        self,
        results: list[Any],
        A: np.ndarray,
        b: np.ndarray,
        success_criterion: Optional[SuccessCriterion] = None,
        verbose: bool = True,
    ) -> TomographyResult:
        """Reconstruct the solution from per-circuit measurement results."""
        ...

    def apply(self, qlsa_circuit: "QLSACircuit", *, state_prep=None) -> QuantumCircuit:
        raise NotImplementedError(
            f"{type(self).__name__} is a multi-circuit readout. Use "
            f"build_circuits(qlsa_circuit) instead, or drive the solve via "
            f"QuantumLinearSolver which dispatches automatically."
        )

    def process(self, result: Any, A: np.ndarray, b: np.ndarray, verbose: bool = True) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} is a multi-circuit readout. Use "
            f"combine_results(results, A, b, success_criterion) instead, or "
            f"drive the solve via QuantumLinearSolver which dispatches automatically."
        )
