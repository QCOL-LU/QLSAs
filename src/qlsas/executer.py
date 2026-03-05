from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator, Optional, Union
import time

from qiskit import QuantumCircuit
from qiskit_ibm_runtime import IBMBackend, SamplerV2 as Sampler, Session
from qiskit.providers.backend import BackendV2
from qiskit.providers.jobstatus import JobStatus
from qnexus import QuantinuumConfig
from pytket.circuit import Circuit


class Executer:
    """
    Executer class for running circuits on target backends.
    Must specify a hardware or simulator backend from qiskit or qnexus.
    """

    def __init__(self) -> None:
        self._session: Optional[Session] = None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def open_session(
        self,
        backend: Union[BackendV2, QuantinuumConfig],
        max_time: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        """Open an IBM Runtime Session for the given backend.

        For non-IBM backends (e.g. AerSimulator) the call is a no-op so that
        callers do not need to guard on backend type.
        """
        if self._session is not None:
            return  # session already active, reuse it

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
        backend: Union[BackendV2, QuantinuumConfig],
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
    # Execution
    # ------------------------------------------------------------------

    def run(
        self, 
        transpiled_circuit: Union[QuantumCircuit, Circuit],
        backend: Union[BackendV2, QuantinuumConfig],
        shots: int,
        verbose: bool = True
    ) -> Any:
        """
        Run the circuit on the backend.
        Args:
            transpiled_circuit: The pre-transpiled circuit to execute.
            backend: The backend to execute the circuit on.
            shots: The number of shots to run.
            verbose: Whether to print job status and ID.
        Returns:
            A result object containing the result of the execution.
        """
        if isinstance(backend, (BackendV2, IBMBackend)):
            return self.run_qiskit(transpiled_circuit, backend, shots, verbose=verbose)
        elif isinstance(backend, QuantinuumConfig):
            return self.run_qnexus(transpiled_circuit, backend, shots, verbose=verbose)
        else:
            raise ValueError(f"Invalid backend type: {type(backend)}")

    def run_qiskit(
        self, 
        transpiled_circuit: QuantumCircuit, 
        backend: BackendV2, 
        shots: int,
        verbose: bool = True
    ) -> Any:
        """
        Run the circuit on the qiskit backend.

        If an IBM Runtime Session is active on this Executer the job is
        submitted through it (priority scheduling, no re-queuing).
        Otherwise the job runs in standalone job mode.
        """
        if self._session is not None:
            sampler = Sampler(mode=self._session)
        else:
            sampler = Sampler(mode=backend)

        job = sampler.run([transpiled_circuit], shots=shots)
        job_id = job.job_id()
        if verbose:
            print(f">>> Job ID: {job_id}")
        self._wait_for_qiskit_job(job, poll_interval_s=5, verbose=verbose)

        result = job.result()[0]
        return result

    def _wait_for_qiskit_job(self, job: Any, poll_interval_s: float = 5, verbose: bool = True) -> str:
        """
        Poll a Qiskit job until it reaches a terminal status, printing a single
        status line that updates in-place.
        """
        terminal_state_names = {"DONE", "CANCELLED", "ERROR"}

        def status_name(s: object) -> str:
            # qiskit-ibm-runtime may return either a JobStatus enum or a string (RuntimeJobV2).
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

        # Final status (finish the line with a newline)
        final_status = status_name(status)
        if verbose:
            msg = f">>> Job Status: {final_status}"
            pad = " " * max(0, last_len - len(msg))
            print(f"\r{msg}{pad}", flush=True)
        return final_status

    def run_qnexus(
        self, 
        transpiled_circuit: Circuit, 
        backend: QuantinuumConfig, 
        shots: int,
        verbose: bool = True
    ) -> dict:
        """
        Run the circuit on the qnexus backend.
        """
        raise NotImplementedError("Qnexus backend execution not implemented yet.")