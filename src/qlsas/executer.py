from qiskit import QuantumCircuit
from qiskit_ibm_runtime import IBMBackend
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.providers.backend import BackendV1, BackendV2
from qnexus import QuantinuumConfig
from pytket.circuit import Circuit
from typing import Union, Any

class Executer:
    """
    Executer class for running circuits on target backends.
    Must specify a hardware or simulator backend from qiskit or qnexus.
    """

    def run(
        self, 
        transpiled_circuit: Union[QuantumCircuit, Circuit],
        backend: Union[BackendV1, BackendV2, QuantinuumConfig],
        shots: int
    ) -> Any:
        """
        Run the circuit on the backend.
        Args:
            circuit: The pre-transpiled circuit to execute.
            backend: The backend to execute the circuit on.
            shots: The number of shots to run.
        Returns:
            A result object containing the result of the execution.
        """
        if isinstance(backend, (BackendV1, BackendV2, IBMBackend)):
            return self.run_qiskit(transpiled_circuit, backend, shots)
        elif isinstance(backend, QuantinuumConfig):
            return self.run_qnexus(transpiled_circuit, backend, shots)
        else:
            raise ValueError(f"Invalid backend type: {type(backend)}")

    def run_qiskit(
        self, 
        transpiled_circuit: QuantumCircuit, 
        backend: Union[BackendV1, BackendV2], 
        shots: int
    ) -> Any:
        """
        Run the circuit on the qiskit backend.
        """
        sampler = Sampler(mode=backend) # TODO generalize modes when using iterative refinement
        job = sampler.run([transpiled_circuit], shots = shots)
        print(f">>> Job ID: {job.job_id()}")
        print(f">>> Job Status: {job.status()}...")
        result = job.result()[0]
        print(f">>> Job Status: {job.status()}")
        return result

    def run_qnexus(
        self, 
        transpiled_circuit: Circuit, 
        backend: QuantinuumConfig, 
        shots: int
    ) -> dict:
        """
        Run the circuit on the qnexus backend.
        """
        raise NotImplementedError("Qnexus backend execution not implemented yet.")