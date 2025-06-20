from HHL_Circuit import hhl_circuit
import numpy as np
import math
from numpy import linalg as LA
from numpy.linalg import solve
from qiskit import transpile
from qiskit_aer import AerSimulator
from pytket.extensions.qiskit import qiskit_to_tk
import qnexus as qnx

def quantum_linear_solver(A, b, backend, t0=2*np.pi, shots=1024):
    """
    Run the HHL circuit on a quantum backend and post-process the result.
    The backend can be a string for a Quantinuum device (e.g., 'H2-2') or an AerSimulator instance.

    Returns:
    The post-processed result of the quantum linear solver (x), and a dictionary of stats about the circuit and job.
    """
    csol = solve(A, b)
    solution = {}

    hhl_circ = hhl_circuit(A, b, t0)
    qiskit_circuit = transpile(hhl_circ, AerSimulator())
    pytket_circuit = qiskit_to_tk(qiskit_circuit)
    
    result = None

    if isinstance(backend, str):
        backend_name = backend
        print(f"Running on {backend_name} via qnexus")
        compiled_circuit = qnx.compile(pytket_circuit, device_name=backend_name, optimisation_level=2)
        
        try:
            solution['cost'] = qnx.estimate_cost(compiled_circuit, shots, backend_name=backend_name)
        except Exception as e:
            print(f"Cost estimation failed: {e}")
            solution['cost'] = 0

        solution['number_of_qubits'] = compiled_circuit.n_qubits
        solution['circuit_depth'] = compiled_circuit.depth()
        solution['total_gates'] = compiled_circuit.n_gates
        solution['two_qubit_gates'] = compiled_circuit.n_2qb_gates()

        print("Submitting job...")
        job = qnx.submit(compiled_circuit, device_name=backend_name, n_shots=shots, project_name='HHL-IR')
        print(f"Job submitted with ID: {job.job_id}")
        solution['job'] = job
        
        print("Waiting for results...")
        result = job.results(timeout_min=30)
        status = job.status()
        print(f"Job status: {status}")
        solution['runtime'] = 'N/A with qnexus'

    elif isinstance(backend, AerSimulator):
        print(f"Running on {backend.name}")
        qiskit_circuit = transpile(hhl_circ, backend, optimization_level=3)
        solution['number_of_qubits'] = qiskit_circuit.num_qubits
        solution['circuit_depth'] = qiskit_circuit.depth()
        solution['total_gates'] = qiskit_circuit.size()
        solution['two_qubit_gates'] = qiskit_circuit.num_two_qubit_gates()        
        solution['cost'] = 0
        job = backend.run(qiskit_circuit, shots=shots)
        result = job.result()
        solution['job'] = job
    else:
        raise TypeError("backend should be a string (backend_name) or an AerSimulator instance")

    def process_result(res, backend_instance):
        counts = res.get_counts()
        b_num = int(math.log2(len(b)))
        num = 0
        app_sol = np.zeros(2 ** b_num)
        is_hardware = isinstance(backend_instance, str)

        for key, value in counts.items():
            key_str = "".join(map(str, key)) if is_hardware else key
            if key_str[-1] == '1':
                num += value
                binary_str = key_str[:-1] if is_hardware else key_str.split(' ')[1]
                cord = int(binary_str, 2)
                app_sol[cord] = value
        
        if num == 0: return app_sol
        app_sol = np.sqrt(app_sol / num)
        qsol = app_sol
        for idx, (i, j) in enumerate(zip(csol, qsol)):
            if i < 0 and j > 0: qsol[idx] = -j
        return qsol

    x = process_result(result, backend)
    solution['x'] = x
    solution['two_norm_error'] = LA.norm(csol - x)
    solution['residual_error'] = LA.norm(b - A @ x)
    return solution