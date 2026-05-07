"""Backwards-compatible facade over the :mod:`qlsas.backends` adapters.

``Executer`` retains the public API existing call sites (solver, refiner,
notebooks, tests) depend on, but the per-backend execution logic now
lives on the :class:`~qlsas.backends.base.Backend` adapters.  Each
``Executer.run*`` method picks the appropriate adapter and delegates.

The IBM Runtime ``Session`` lifecycle continues to live on ``Executer``
in PR A1 because the solver and refiner currently hold an executer
instance and call ``open_session`` / ``close_session`` on it directly.
PR A2 will move session ownership to the backend adapter and remove
this facade.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator, Optional, Union

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import IBMBackend, SamplerV2 as Sampler, Session  # noqa: F401  re-exported for monkeypatch back-compat
from qiskit.providers.backend import BackendV2
from pytket.circuit import Circuit

from qlsas.backends.base import Backend, CompiledArtifact
from qlsas.backends.dispatch import adapt
from qlsas.backends.qiskit_backend import QiskitBackend
from qlsas.backends.quantinuum_backend import QuantinuumBackend
from qlsas.quantinuum_config import QuantinuumBackendConfig
from qlsas.measurement_result import MeasurementResult
from qlsas.guppy_runner import RegisterInfo, RegisterMeasurement
from qlsas.ibm_options import IBMExecutionOptions


class Executer:
    """Coordinator that dispatches execution to a Backend adapter.

    Holds an optional IBM Runtime ``Session`` so that batches of jobs
    submitted to the same IBM backend share priority scheduling.  The
    session is a no-op for non-IBM backends.
    """

    def __init__(self, ibm_options: Optional[IBMExecutionOptions] = None) -> None:
        self._session: Optional[Session] = None
        self.ibm_options = ibm_options

    # ------------------------------------------------------------------
    # IBM Runtime Session lifecycle (no-op for non-IBM backends)
    # ------------------------------------------------------------------

    def open_session(
        self,
        backend: Union[BackendV2, QuantinuumBackendConfig, Backend],
        max_time: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        if self._session is not None:
            return

        underlying = (
            backend.underlying  # type: ignore[attr-defined]
            if isinstance(backend, QiskitBackend)
            else backend
        )
        if not isinstance(underlying, IBMBackend):
            return

        kwargs: dict[str, Any] = {"backend": underlying}
        if max_time is not None:
            kwargs["max_time"] = max_time
        self._session = Session(**kwargs)
        if verbose:
            print(f">>> Opened IBM Runtime Session (backend={underlying.name})")

    def close_session(self, verbose: bool = True) -> None:
        if self._session is not None:
            self._session.close()
            if verbose:
                print(">>> Closed IBM Runtime Session")
            self._session = None

    @property
    def session_active(self) -> bool:
        return self._session is not None

    @contextmanager
    def session(
        self,
        backend: Union[BackendV2, QuantinuumBackendConfig, Backend],
        max_time: Optional[str] = None,
        verbose: bool = True,
    ) -> Generator[None, None, None]:
        self.open_session(backend, max_time=max_time, verbose=verbose)
        try:
            yield
        finally:
            self.close_session(verbose=verbose)

    # ------------------------------------------------------------------
    # Execution dispatch
    # ------------------------------------------------------------------

    def run(
        self,
        transpiled_circuit: Union[QuantumCircuit, Circuit],
        backend: Union[BackendV2, QuantinuumBackendConfig, Backend],
        shots: int,
        ibm_options: Optional[IBMExecutionOptions] = None,
        verbose: bool = True,
        register_infos: Optional[list[RegisterInfo]] = None,
        measurement_plan: Optional[list[RegisterMeasurement]] = None,
        optimization_level: int = 2,
    ) -> MeasurementResult:
        """Run the transpiled circuit on the backend; return :class:`MeasurementResult`."""
        try:
            adapter = adapt(backend, ibm_options=ibm_options or self.ibm_options)
        except ValueError as exc:
            raise ValueError(f"Invalid backend type: {type(backend).__name__}") from exc

        if isinstance(adapter, QuantinuumBackend):
            if register_infos is None or measurement_plan is None:
                raise ValueError(
                    "register_infos and measurement_plan are required for "
                    "Quantinuum execution. These are produced by "
                    "Transpiler.optimize_quantinuum()."
                )
            artifact = CompiledArtifact(
                payload=transpiled_circuit,
                backend_metadata={
                    "register_infos": register_infos,
                    "measurement_plan": measurement_plan,
                    "optimization_level": optimization_level,
                },
            )
            return adapter.run_compiled(artifact, shots, verbose=verbose)

        # Qiskit / Aer / IBM path
        artifact = CompiledArtifact(payload=transpiled_circuit)
        return adapter.run_compiled(
            artifact,
            shots,
            verbose=verbose,
            session=self._session,
            ibm_options=ibm_options,
        )

    # ------------------------------------------------------------------
    # Legacy direct paths (kept as thin shims for back-compat)
    # ------------------------------------------------------------------

    def run_qiskit(
        self,
        transpiled_circuit: QuantumCircuit,
        backend: Union[BackendV2, IBMBackend],
        shots: int,
        ibm_options: Optional[IBMExecutionOptions] = None,
        verbose: bool = True,
    ) -> Any:
        """Direct Qiskit-path execution; returns the unwrapped raw result.

        Wraps *backend* in a :class:`QiskitBackend` directly (bypassing
        :func:`adapt`) so callers can pass duck-typed fakes in tests.
        """
        adapter = QiskitBackend(backend, ibm_options=ibm_options or self.ibm_options)
        artifact = CompiledArtifact(payload=transpiled_circuit)
        result = adapter.run_compiled(
            artifact,
            shots,
            verbose=verbose,
            session=self._session,
            ibm_options=ibm_options,
        )
        return result.raw

    def run_quantinuum(
        self,
        transpiled_circuit: Circuit,
        backend: QuantinuumBackendConfig,
        shots: int,
        register_infos: list[RegisterInfo],
        measurement_plan: list[RegisterMeasurement],
        verbose: bool = True,
        optimization_level: int = 2,
    ) -> dict[str, int]:
        """Direct Quantinuum-path execution; returns the unwrapped counts dict."""
        adapter = QuantinuumBackend(backend)
        artifact = CompiledArtifact(
            payload=transpiled_circuit,
            backend_metadata={
                "register_infos": register_infos,
                "measurement_plan": measurement_plan,
                "optimization_level": optimization_level,
            },
        )
        result = adapter.run_compiled(artifact, shots, verbose=verbose)
        return result.raw
