from qiskit import QuantumCircuit
from qiskit_ibm_runtime import IBMBackend
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, Session
from qiskit.providers.backend import BackendV2
from qiskit.providers.jobstatus import JobStatus
from qnexus import QuantinuumConfig
from pytket.circuit import Circuit
from typing import Union, Any
import time
from typing import Optional

class Executer:
    """
    Executer class for running circuits on target backends.
    Must specify a hardware or simulator backend from qiskit or qnexus.
    """

    def run(
        self, 
        transpiled_circuit: Union[QuantumCircuit, Circuit],
        backend: Union[BackendV2, QuantinuumConfig],
        shots: int,
        mode: Optional[str] = None,
        verbose: bool = True
    ) -> Any:
        """
        Run the circuit on the backend.
        Args:
            circuit: The pre-transpiled circuit to execute.
            backend: The backend to execute the circuit on.
            shots: The number of shots to run.
            mode: The mode to run the job in. Job (for single qlsa run) or Session (for iterative refinement).
            verbose: Whether to print job status and ID.
        Returns:
            A result object containing the result of the execution.
        """
        if isinstance(backend, (BackendV2, IBMBackend)):
            return self.run_qiskit(transpiled_circuit, backend, shots, mode=mode, verbose=verbose)
        elif isinstance(backend, QuantinuumConfig):
            return self.run_qnexus(transpiled_circuit, backend, shots, mode=mode, verbose=verbose)
        else:
            raise ValueError(f"Invalid backend type: {type(backend)}")

    def run_qiskit(
        self, 
        transpiled_circuit: QuantumCircuit, 
        backend: BackendV2, 
        shots: int,
        mode: Optional[str] = None,
        verbose: bool = True
    ) -> Any:
        """
        Run the circuit on the qiskit backend.
        """
        if mode == "job" or mode is None: # single qlsa run
            sampler = Sampler(mode=backend) # TODO generalize modes to batch or session when using iterative refinement
            job = sampler.run([transpiled_circuit], shots = shots)
            job_id = job.job_id()
            if verbose:
                print(f">>> Job ID: {job_id}")
            self._wait_for_qiskit_job(job, poll_interval_s=5, verbose=verbose)

            result = job.result()[0]
            return result

        else: # iterative refinement run
            raise ValueError(f"mode {mode} not implemented yet.")

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
        mode: Optional[str] = None,
        verbose: bool = True
    ) -> dict:
        """
        Run the circuit on the qnexus backend.
        """
        raise NotImplementedError("Qnexus backend execution not implemented yet.")