from qiskit import QuantumCircuit
from qiskit_ibm_runtime import IBMBackend
from qiskit_aer import AerSimulator
from qiskit.providers.backend import BackendV1, BackendV2
from qiskit.transpiler import generate_preset_pass_manager
from qnexus import QuantinuumConfig
from pytket.circuit import Circuit
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from typing import Union


class Transpiler:
    """
    Transpiler class for optimizing quantum circuits for target hardware.
    Encomposes qiskit and qnexus transpiler classes.
    """ 

    def __init__(
        self,
        circuit: Union[QuantumCircuit, Circuit],
        backend: Union[BackendV1, BackendV2, QuantinuumConfig],
        optimization_level: int,
    ):
        self.circuit = circuit
        self.backend = backend
        self.optimization_level = optimization_level

    def optimize(self) -> QuantumCircuit:
        """
        Optimize the circuit for target hardware.
        Args:
            circuit: The circuit to optimize.
            backend: The backend to optimize for.
        Returns:
            The optimized circuit.
        """
        if isinstance(self.backend, (BackendV1, BackendV2, IBMBackend, AerSimulator)):
            return self.optimize_qiskit()
        elif isinstance(self.backend, QuantinuumConfig):
            return self.optimize_qnexus()
        else:
            raise ValueError(f"Invalid backend type: {type(self.backend)}")

    
    def optimize_qiskit(self) -> QuantumCircuit:
        """
        Optimize the circuit for target IBM hardware.
        Args:
            circuit: The circuit to optimize.
            backend: The backend to optimize for.
            optimization_level: The level of optimization to perform. 
        Returns:
            The optimized circuit.
        """

        if self.optimization_level not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid optimization level: {self.optimization_level}. Must be 0, 1, 2, or 3.")
        
        # Convert to qiskit circuit if needed
        if isinstance(self.circuit, QuantumCircuit):
            pass
        elif isinstance(self.circuit, Circuit):
            self.circuit = tk_to_qiskit(self.circuit)
        else:
            raise ValueError(f"Invalid circuit type: {type(self.circuit)}. Must be a qiskit QuantumCircuit or a pytket Circuit.")
        
        pm = generate_preset_pass_manager(optimization_level=self.optimization_level, backend=self.backend)
        transpiled_circuit = pm.run(self.circuit)
        
        return transpiled_circuit

    def optimize_qnexus(self) -> Circuit:
        """
        Optimize the circuit for target Quantinuum hardware.
        Args:
            circuit: The circuit to optimize.
            backend: The backend to optimize for.
            optimization_level: The level of optimization to perform.
        Returns:
            The optimized circuit.
        """
        raise NotImplementedError("Quantinuum/QNexus transpilation is not implemented yet (Transpiler.optimize_qnexus).")