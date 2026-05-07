"""Backwards-compatible facade over the :mod:`qlsas.backends` adapters.

``Transpiler`` is now a thin coordinator: it picks the appropriate
:class:`~qlsas.backends.base.Backend` adapter for the requested raw
backend and delegates the actual compile work to it.  The legacy public
API (``optimize()``, ``optimize_qiskit()``, ``optimize_quantinuum()``,
plus the ``register_infos`` / ``measurement_plan`` attributes the solver
threads to the executer) is preserved unchanged so that existing call
sites and tests work without modification.
"""

from __future__ import annotations

from typing import Union

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import IBMBackend
from qiskit_aer import AerSimulator
from qiskit.providers.backend import BackendV2
from pytket.circuit import Circuit

from qlsas.backends.base import Backend
from qlsas.backends.dispatch import adapt
from qlsas.backends.qiskit_backend import QiskitBackend
from qlsas.backends.quantinuum_backend import QuantinuumBackend
from qlsas.guppy_runner import RegisterInfo, RegisterMeasurement
from qlsas.quantinuum_config import QuantinuumBackendConfig


class Transpiler:
    """Coordinator that dispatches circuit optimisation to a Backend adapter.

    Kept for back-compat with existing call sites; ``Backend.compile`` is
    the canonical entry point for new code.
    """

    def __init__(
        self,
        circuit: Union[QuantumCircuit, Circuit],
        backend: Union[BackendV2, QuantinuumBackendConfig, Backend],
        optimization_level: int,
    ):
        self.circuit = circuit
        self.backend = backend
        self.optimization_level = optimization_level
        self.register_infos: list[RegisterInfo] = []
        self.measurement_plan: list[RegisterMeasurement] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self) -> Union[QuantumCircuit, Circuit]:
        """Optimise the circuit for the configured backend.

        Returns the transpiled circuit.  As a side effect, populates
        ``self.register_infos`` and ``self.measurement_plan`` for the
        Quantinuum path (left empty for the Qiskit path).
        """
        adapter = adapt(self.backend)
        artifact = adapter.compile(self.circuit, self.optimization_level)
        self.register_infos = artifact.backend_metadata.get("register_infos", [])
        self.measurement_plan = artifact.backend_metadata.get("measurement_plan", [])
        return artifact.payload

    # ------------------------------------------------------------------
    # Legacy direct paths (kept as thin shims for back-compat)
    # ------------------------------------------------------------------

    def optimize_qiskit(self) -> QuantumCircuit:
        """Direct Qiskit-path compile.  Equivalent to :meth:`optimize` when
        the backend is Aer / IBM."""
        backend = self.backend
        adapter: Backend = (
            backend if isinstance(backend, Backend)
            else QiskitBackend(backend)  # type: ignore[arg-type]
        )
        artifact = adapter.compile(self.circuit, self.optimization_level)
        return artifact.payload

    def optimize_quantinuum(self) -> Circuit:
        """Direct Quantinuum-path compile.  Equivalent to :meth:`optimize`
        when the backend is :class:`QuantinuumBackendConfig`."""
        backend = self.backend
        adapter: Backend = (
            backend if isinstance(backend, Backend)
            else QuantinuumBackend(backend)  # type: ignore[arg-type]
        )
        artifact = adapter.compile(self.circuit, self.optimization_level)
        self.register_infos = artifact.backend_metadata.get("register_infos", [])
        self.measurement_plan = artifact.backend_metadata.get("measurement_plan", [])
        return artifact.payload
