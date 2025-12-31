from qlsas.transpiler.base import Transpiler
from qiskit import QuantumCircuit
from qiskit.compiler.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService

class QiskitTranspiler(Transpiler):
    def __init__(self):
        pass

    def optimize(self, circuit: QuantumCircuit, backend_name: str, name: str = "QLSAs") -> QuantumCircuit:
        """
        Optimize the circuit for target hardware.
        Uses Qiskit's transpile function.
        Abstract circuits must be transpiled at this stage to an Instruction Set Architecture (ISA)
        that is compatible with the target hardware.
        Args:
            circuit: The circuit to optimize.
            backend: The name of the backend to optimize for.
            name: The name of the saved account to use.
        Returns:
            The optimized circuit.
        """
        service = QiskitRuntimeService(name=name)
        backend = service.backend(backend_name)
        pm = generate_preset_pass_manager(optimization_level=3, backend=backend)

        transpiled_circuit = pm.run(circuit)
        return transpiled_circuit