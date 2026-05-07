"""Quantinuum backend adapter (Selene local + Nexus cloud).

Wraps a :class:`~qlsas.quantinuum_config.QuantinuumBackendConfig` behind
the unified :class:`~qlsas.backends.base.Backend` interface.

Compile path branches on ``config.use_local_emulator``:

* **Local Selene** — strips measurements, applies pytket peephole
  optimisation and an ``AutoRebase`` to ``{CX, Rz, H}``, returns a
  measurement-free pytket circuit.  ``register_infos`` and the
  ``measurement_plan`` are stashed in the artifact's ``backend_metadata``
  for use at run-time by the Guppy wrapper.

* **Nexus cloud** — keeps measurements, applies only minimal pre-
  processing (Aer decomposition + ``DecomposeBoxes``); device-native
  compilation happens inside the Nexus compile job at execution time.

Run path delegates to :func:`build_and_run_guppy` (Selene) or
:func:`run_nexus_pytket` (cloud).
"""

from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit
from pytket.circuit import Circuit

from qlsas.backends.base import Backend, CompiledArtifact
from qlsas.guppy_runner import (
    build_and_run_guppy,
    extract_measurement_plan,
    prepare_pytket_circuit,
    prepare_pytket_circuit_for_nexus,
    run_nexus_pytket,
)
from qlsas.measurement_result import MeasurementResult
from qlsas.quantinuum_config import QuantinuumBackendConfig


class QuantinuumBackend(Backend):
    """Adapter around a :class:`QuantinuumBackendConfig`."""

    def __init__(self, config: QuantinuumBackendConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Identification
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def config(self) -> QuantinuumBackendConfig:
        return self._config

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    @property
    def supports_multi_circuit(self) -> bool:
        # Multi-circuit dispatch (HRF, future shadow tomography) has not
        # been validated against the Guppy / Nexus paths yet.  Solver
        # consults this flag to raise a clear error early.
        return False

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def compile(
        self,
        qc: QuantumCircuit,
        optimization_level: int = 2,
    ) -> CompiledArtifact:
        if isinstance(qc, Circuit):
            raise TypeError(
                "QuantinuumBackend.compile expects a Qiskit QuantumCircuit, "
                "got pytket Circuit. Convert with tk_to_qiskit first."
            )

        measurement_plan = extract_measurement_plan(qc)

        if self._config.use_local_emulator:
            pytket_circuit, register_infos = prepare_pytket_circuit(
                qc, optimization_level=optimization_level
            )
        else:
            pytket_circuit, register_infos = prepare_pytket_circuit_for_nexus(qc)

        return CompiledArtifact(
            payload=pytket_circuit,
            backend_metadata={
                "register_infos": register_infos,
                "measurement_plan": measurement_plan,
                "optimization_level": optimization_level,
            },
        )

    def run_compiled(
        self,
        artifact: CompiledArtifact,
        shots: int = 1024,
        *,
        verbose: bool = True,
        **_unused: Any,
    ) -> MeasurementResult:
        meta = artifact.backend_metadata
        register_infos = meta.get("register_infos")
        measurement_plan = meta.get("measurement_plan")
        optimization_level = meta.get("optimization_level", 2)

        if register_infos is None or measurement_plan is None:
            raise ValueError(
                "QuantinuumBackend.run_compiled requires register_infos and "
                "measurement_plan in artifact.backend_metadata. These are "
                "produced by QuantinuumBackend.compile()."
            )

        if self._config.use_local_emulator:
            counts = build_and_run_guppy(
                pytket_circuit=artifact.payload,
                register_infos=register_infos,
                measurement_plan=measurement_plan,
                config=self._config,
                shots=shots,
                verbose=verbose,
            )
        else:
            counts = run_nexus_pytket(
                pytket_circuit=artifact.payload,
                config=self._config,
                shots=shots,
                optimization_level=optimization_level,
                measurement_plan=measurement_plan,
                verbose=verbose,
            )

        return MeasurementResult(counts)
