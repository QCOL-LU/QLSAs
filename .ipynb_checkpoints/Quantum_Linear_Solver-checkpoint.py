from HHL_Circuit import hhl_circuit
import numpy as np
import math
from numpy import linalg as LA
from numpy.linalg import solve
from qiskit import transpile
from qiskit_aer import AerSimulator
from pytket.extensions.nexus import NexusBackend
from pytket.extensions.qiskit import qiskit_to_tk


def quantum_linear_solver(A, b, backend, t0=2*np.pi, shots=1024):  # run hhl circuit on a quantum backend and post-process the result
    """
    Run the hhl circuit on a quantinuum backend and return the result and the compiled circuit.
    Returns:
    The post-processed result of the quantum linear solver (x), and a whole bunch of stats about the circuit.
    """
    csol = solve(A, b)
    solution = {}

    hhl_circ = hhl_circuit(A, b, t0)
    # transpile in qiskit then convert to a tket circuit if backend is from quantinuum
    qiskit_circuit = transpile(hhl_circ, AerSimulator())

    # if isinstance(backend, AerSimulator):
    #     print(f"Running on {backend.name}")
    #     qiskit_circuit = transpile(hhl_circ, backend, optimization_level=3)
    #     solution['number_of_qubits'] = qiskit_circuit.num_qubits
    #     solution['circuit_depth'] = qiskit_circuit.depth()
    #     solution['total_gates'] = qiskit_circuit.size()
    #     solution['cost'] = 0
    #     job = backend.run(qiskit_circuit, shots=shots)
    #     result = job.result()

    if isinstance(backend, NexusBackend):
        print(f"Running on {backend.backend_config.device_name}")
        qtuum_circuit = qiskit_to_tk(qiskit_circuit)
        # re-transpile to quantinuum backend
        new_qtuum_circuit = backend.get_compiled_circuit(qtuum_circuit, optimisation_level=2)

        try:
            syntax_checker = backend.backend_config.device_name[:4] + "SC"
            solution['cost'] = backend.cost(new_qtuum_circuit, shots, syntax_checker)
        except Exception:
            solution['cost'] = 0

        # Get circuit stats
        solution['number_of_qubits'] = new_qtuum_circuit.n_qubits
        solution['circuit_depth'] = new_qtuum_circuit.depth()
        solution['total_gates'] = new_qtuum_circuit.n_gates
        solution['two_qubit_gates'] = new_qtuum_circuit.n_2qb_gates()

        # Run circuit
        result_handle = backend.process_circuit(new_qtuum_circuit, n_shots=shots)
        solution['result_handle'] = result_handle

        result = backend.get_result(result_handle, timeout=None)  # only works if job completed
        status = backend.circuit_status(result_handle)

        try:
            solution['runtime'] = status.completed_time - status.running_time
        except Exception:
            # Skip the line if an error occurs
            solution['runtime'] = 'Not Found'

    else:
        print('backend should be a NexusBackend')

    def process_result(result):  # process the result of the quantum linear solver and return the solution vector
        """
        Process the result of the quantum linear solver and return the solution vector.
        """
        # Get the counts
        counts = result.get_counts()

        def solution_vector(counts, b):
            b_num = int(math.log2(len(b)))
            num = 0  # for normalization
            app_sol = np.zeros(2 ** b_num)

            if isinstance(backend, AerSimulator):
                for key, value in counts.items():
                    if key[-1] == '1':
                        num += value
                        cord = int(key[:b_num], base=2)  # position in b vector from binary string
                        app_sol[cord] = value
                if num == 0:
                    return app_sol
                app_sol = np.sqrt(app_sol/num)
                # app_sol = app_sol/num
                return app_sol

            else:
                for key, value in counts.items():
                    key_str = "".join(str(bit) for bit in key)
                    if key_str[-1] == '1':
                        num += value
                        cord = int(key_str[:b_num], base=2)
                        app_sol[cord] = value
                if num == 0:
                    return app_sol
                app_sol = np.sqrt(app_sol/num)
                # app_sol = app_sol/num
                return app_sol

        # Extract approximate solution
        qsol = solution_vector(counts, b)

        # Avoiding sign estimation for now
        for idx, (i, j) in enumerate(zip(csol, qsol)):
            if i < 0:
                qsol[idx] = -j
        return qsol

    x = process_result(result)
    solution['x'] = x

    two_norm_error = LA.norm(csol - x)
    solution['two_norm_error'] = two_norm_error

    residual_error = LA.norm(b - A @ x)
    solution['residual_error'] = residual_error

    return solution