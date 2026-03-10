from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator, Optional, Union
import time

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import IBMBackend, SamplerV2 as Sampler, Session
from qiskit.providers.backend import BackendV2
from qiskit.providers.jobstatus import JobStatus
from pytket.circuit import Circuit

from qlsas.quantinuum_config import QuantinuumBackendConfig
from qlsas.guppy_runner import (
    RegisterInfo,
    RegisterMeasurement,
    build_and_run_guppy,
    run_nexus_pytket,
)
from qlsas.ibm_options import (
    IBMExecutionOptions,
    apply_ibm_error_mitigation_options,
)


class Executer:
    """Executer class for running circuits on target backends.

    Supports IBM/Qiskit backends (via SamplerV2) and Quantinuum backends
    (via Guppy/Selene/Nexus).
    """

    def __init__(self, ibm_options: Optional[IBMExecutionOptions] = None) -> None:
        self._session: Optional[Session] = None
        self.ibm_options = ibm_options

    # ------------------------------------------------------------------
    # Session lifecycle (IBM only; no-op for other backends)
    # ------------------------------------------------------------------

    def open_session(
        self,
        backend: Union[BackendV2, QuantinuumBackendConfig],
        max_time: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        """Open an IBM Runtime Session for the given backend.

        For non-IBM backends (AerSimulator, QuantinuumBackendConfig) the call
        is a no-op so that callers do not need to guard on backend type.
        """
        if self._session is not None:
            return

        if not isinstance(backend, IBMBackend):
            return

        kwargs: dict[str, Any] = {"backend": backend}
        if max_time is not None:
            kwargs["max_time"] = max_time
        self._session = Session(**kwargs)
        if verbose:
            print(f">>> Opened IBM Runtime Session (backend={backend.name})")

    def close_session(self, verbose: bool = True) -> None:
        """Close the active session, if any."""
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
        backend: Union[BackendV2, QuantinuumBackendConfig],
        max_time: Optional[str] = None,
        verbose: bool = True,
    ) -> Generator[None, None, None]:
        """Context manager that opens and closes a session."""
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
        backend: Union[BackendV2, QuantinuumBackendConfig],
        shots: int,
        ibm_options: Optional[IBMExecutionOptions] = None,
        verbose: bool = True,
        register_infos: Optional[list[RegisterInfo]] = None,
        measurement_plan: Optional[list[RegisterMeasurement]] = None,
        optimization_level: int = 2,
    ) -> Any:
        """Run the circuit on the backend.

        For IBM backends, returns a ``SamplerPubResult``.
        For Quantinuum backends, returns a ``dict[str, int]`` counts dict
        ready for the post-processor.

        ``optimization_level`` is forwarded to ``qnx.start_compile_job`` for
        Nexus cloud execution and ignored for IBM and local Selene backends.
        """
        if isinstance(backend, (BackendV2, IBMBackend)):
            return self.run_qiskit(
                transpiled_circuit,
                backend,
                shots,
                ibm_options=ibm_options,
                verbose=verbose,
            )
        elif isinstance(backend, QuantinuumBackendConfig):
            if register_infos is None or measurement_plan is None:
                raise ValueError(
                    "register_infos and measurement_plan are required for "
                    "Quantinuum execution. These are produced by "
                    "Transpiler.optimize_quantinuum()."
                )
            return self.run_quantinuum(
                transpiled_circuit,
                backend,
                shots,
                register_infos=register_infos,
                measurement_plan=measurement_plan,
                verbose=verbose,
                optimization_level=optimization_level,
            )
        else:
            raise ValueError(f"Invalid backend type: {type(backend)}")

    # ------------------------------------------------------------------
    # IBM / Qiskit
    # ------------------------------------------------------------------

    def run_qiskit(
        self,
        transpiled_circuit: QuantumCircuit,
        backend: BackendV2,
        shots: int,
        ibm_options: Optional[IBMExecutionOptions] = None,
        verbose: bool = True,
    ) -> Any:
        """Run the circuit on a Qiskit backend.

        If an IBM Runtime Session is active the job is submitted through it
        (priority scheduling, no re-queuing).  Otherwise the job runs in
        standalone job mode.
        """
        if self._session is not None:
            sampler = Sampler(mode=self._session)
        else:
            sampler = Sampler(mode=backend)

        effective_ibm_options = ibm_options or self.ibm_options
        if isinstance(backend, IBMBackend):
            apply_ibm_error_mitigation_options(sampler.options, effective_ibm_options)

        job = sampler.run([transpiled_circuit], shots=shots)
        job_id = job.job_id()
        if verbose:
            print(f">>> Job ID: {job_id}")
        self._wait_for_qiskit_job(job, poll_interval_s=5, verbose=verbose)

        result = job.result()[0]
        return result

    def _wait_for_qiskit_job(
        self, job: Any, poll_interval_s: float = 5, verbose: bool = True
    ) -> str:
        """Poll a Qiskit job until it reaches a terminal status."""
        terminal_state_names = {"DONE", "CANCELLED", "ERROR"}

        def status_name(s: object) -> str:
            return s.name if isinstance(s, JobStatus) else str(s)

        status = job.status()
        last_len = 0
        while status_name(status) not in terminal_state_names:
            if verbose:
                msg = f">>> Job Status: {status_name(status)}..."
                pad = " " * max(0, last_len - len(msg))
                print(f"\r{msg}{pad}", end="", flush=True)
                last_len = len(msg)
            time.sleep(poll_interval_s)
            status = job.status()

        final_status = status_name(status)
        if verbose:
            msg = f">>> Job Status: {final_status}"
            pad = " " * max(0, last_len - len(msg))
            print(f"\r{msg}{pad}", flush=True)
        return final_status

    # ------------------------------------------------------------------
    # Quantinuum (via Guppy)
    # ------------------------------------------------------------------

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
        """Run the circuit on a Quantinuum backend.

        Dispatches to one of two paths based on ``backend.use_local_emulator``:

        - **Local Selene**: wraps the measurement-free pytket circuit in a Guppy
          program and runs it on the local Selene emulator.
        - **Nexus cloud**: uploads the pytket circuit (with measurements) to
          Nexus, runs a compile job at *optimization_level*, then submits an
          execute job and parses the ``BackendResult``.
        """
        if backend.use_local_emulator:
            return build_and_run_guppy(
                pytket_circuit=transpiled_circuit,
                register_infos=register_infos,
                measurement_plan=measurement_plan,
                config=backend,
                shots=shots,
                verbose=verbose,
            )
        else:
            return run_nexus_pytket(
                pytket_circuit=transpiled_circuit,
                config=backend,
                shots=shots,
                optimization_level=optimization_level,
                measurement_plan=measurement_plan,
                verbose=verbose,
            )
