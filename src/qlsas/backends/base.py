"""Backend protocol and shared dataclasses.

A :class:`Backend` is a thin polymorphic wrapper around a concrete execution
target (Aer, IBM Runtime, Quantinuum, CUDA-Q, ...) with two responsibilities:

* :meth:`Backend.compile` — turn a Qiskit ``QuantumCircuit`` into a
  :class:`CompiledArtifact` ready for execution on this backend.
* :meth:`Backend.run_compiled` — execute a compiled artifact and return a
  :class:`~qlsas.measurement_result.MeasurementResult`.

The convenience :meth:`Backend.run(qc, shots, token)` chains the two and
exposes a signature compatible with Qrisp's ``VirtualBackend.run`` so a
future ``as_qrisp_backend(backend)`` adapter is one line.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from qiskit import QuantumCircuit

from qlsas.measurement_result import MeasurementResult


@dataclass
class RegisterPlan:
    """Per-circuit register metadata that survives the compile→run boundary.

    Today this is only the ordered list of classical-register names a readout
    will join for post-selection.  It exists as a struct (rather than a bare
    ``list[str]``) so future fields (per-register width, qubit-to-clbit map)
    can be added without touching every call site.
    """

    register_names: list[str] = field(default_factory=list)


@dataclass
class CompiledArtifact:
    """Output of :meth:`Backend.compile`.

    ``payload`` is whatever the backend's compile step produces — a Qiskit
    ``QuantumCircuit``, a pytket ``Circuit``, or (for CUDA-Q) a kernel
    handle.  ``backend_metadata`` is opaque to all callers other than the
    backend that produced it, and is the channel by which compile-time
    information (e.g. Quantinuum's measurement plan) is forwarded to
    ``run_compiled`` without polluting the public signature.
    """

    payload: Any
    register_plan: RegisterPlan = field(default_factory=RegisterPlan)
    backend_metadata: dict = field(default_factory=dict)


class Backend(ABC):
    """Abstract execution backend.

    Concrete subclasses wrap a real backend object (e.g. ``AerSimulator``,
    ``QuantinuumBackendConfig``) and implement :meth:`compile` /
    :meth:`run_compiled`.  The :meth:`run` convenience signature matches
    Qrisp's :class:`VirtualBackend.run(qc, shots, token)` for forward
    compatibility with a Qrisp-based algorithm layer.
    """

    # ------------------------------------------------------------------
    # Identification
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short label used in logs and plot legends (e.g. ``refiner.py``)."""
        ...

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def compile(
        self,
        qc: QuantumCircuit,
        optimization_level: int = 2,
    ) -> CompiledArtifact:
        """Optimise *qc* for this backend; return the artifact for execution."""
        ...

    @abstractmethod
    def run_compiled(
        self,
        artifact: CompiledArtifact,
        shots: int = 1024,
        *,
        verbose: bool = True,
        **kwargs: Any,
    ) -> MeasurementResult:
        """Execute a compiled artifact and return a wrapped result.

        ``kwargs`` is reserved for backend-specific runtime options (e.g.
        an active IBM ``Session``) that the existing ``Executer`` threads
        through.  Concrete backends should accept and ignore unknown kwargs
        rather than failing — the kwargs surface is intentionally loose
        during the protocol-introduction phase and is expected to tighten
        once the solver passes artifacts directly.
        """
        ...

    # ------------------------------------------------------------------
    # Qrisp-compatible convenience
    # ------------------------------------------------------------------

    def run(
        self,
        qc: QuantumCircuit,
        shots: int = 1024,
        token: str = "",
    ) -> MeasurementResult:
        """Compile + execute.  Mirrors Qrisp ``VirtualBackend.run``.

        ``token`` is accepted for Qrisp signature parity and ignored: cloud
        backends authenticate through the credentials loaded at adapter
        construction (e.g. ``QiskitRuntimeService``, ``qnexus`` login).
        """
        del token  # accepted for VirtualBackend parity
        artifact = self.compile(qc)
        return self.run_compiled(artifact, shots)

    # ------------------------------------------------------------------
    # Capability flags (default implementations)
    # ------------------------------------------------------------------

    @property
    def supports_multi_circuit(self) -> bool:
        """Whether this backend has been validated for the multi-circuit
        readout dispatch path (HRF, future shadow tomography).

        Default ``True`` — override to ``False`` only for backends where the
        path has been explicitly verified to fail (today: the Quantinuum
        path, which has no multi-circuit support yet).
        """
        return True
