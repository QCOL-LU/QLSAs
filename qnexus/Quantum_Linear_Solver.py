from HHL_Circuit import hhl_circuit
from datetime import datetime
import numpy as np
import math
from numpy import linalg as LA
from numpy.linalg import solve
from qiskit import transpile
from qiskit_aer import AerSimulator
from pytket.extensions.qiskit import qiskit_to_tk
import qnexus as qnx
import concurrent.futures
import time

def robust_wait_for(job_ref, status_func, target_status="COMPLETED", timeout=None, poll_interval=5):
    """
    Polls the job status until it is COMPLETED or ERROR, or until timeout.
    - job_ref: the job reference object
    - status_func: a function that returns the job status (e.g., qnx.jobs.status)
    - target_status: status string to wait for (default: 'COMPLETED')
    - timeout: max seconds to wait (default: None for infinite)
    - poll_interval: seconds between polls (default: 5)
    Returns the final status.
    """
    start_time = time.time()
    while True:
        status = status_func(job_ref)
        print(f"Job status: {status}")
        if str(status) == target_status or str(status) == "ERROR":
            break
        if timeout is not None and (time.time() - start_time > timeout):
            print("Timeout waiting for job to complete.")
            break
        time.sleep(poll_interval)
    return status

def quantum_linear_solver(A, b, backend, t0=2*np.pi, shots=1024, iteration=None):
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
    
    # result = None

    project_ref = qnx.projects.get_or_create(name="HHL-IR")
    qnx.context.set_active_project(project_ref)

    if isinstance(backend, str):
        backend_name = backend
        print(f"Running on {backend_name} via qnexus")

        # Step 1: Upload the circuit. This returns a CircuitRef to the UNCOMPILED circuit.
        circuit_name = f"hhl-circuit-{len(b)}x{len(b)}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        if iteration is not None:
            circuit_name += f"-iter{iteration}"
        
        print(f"Uploading circuit '{circuit_name}'...")
        circuit_ref = qnx.circuits.upload(
            name=circuit_name,
            circuit=pytket_circuit,
            project=project_ref
        )
        if not circuit_ref:
            raise RuntimeError("Circuit upload failed.")
        print(f"Circuit uploaded with ID: {circuit_ref.id}")

        # Step 2: Compile the circuit using the reference. This returns a NEW CircuitRef to the COMPILED circuit.
        config = qnx.QuantinuumConfig(device_name=backend_name, attempt_batching=True)
        
        print("Compiling circuit...")
        ref_compile_job = qnx.start_compile_job(
            circuits =[circuit_ref],  # Must be a list of references
            backend_config=config,
            optimisation_level=2,
            name=f"hhl-ir-compile-{len(b)}x{len(b)}-iter{iteration}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        )
        if not ref_compile_job:
            raise RuntimeError("Circuit compilation failed.")
        
        robust_wait_for(ref_compile_job, qnx.jobs.status, target_status="COMPLETED", timeout=None, poll_interval=5)
        ref_compiled_circuit = qnx.jobs.results(ref_compile_job)[0].get_output()
        print(f"Compilation successful. Compiled circuit ID: {ref_compiled_circuit.id}")

        # Step 3: GET the full compiled circuit object to access its properties.
        
        # full_compiled_circuit = qnx.circuits.get(id = compiled_ref.id)
        compiled_circuit = ref_compiled_circuit.download_circuit()

        # Step 4: Correctly call the cost estimation function from the `jobs` module.
        if backend_name == 'H1-1' or backend_name == 'H1-1E' or backend_name == 'H1-1LE':
            syntax_checker = 'H1-1SC'
        elif backend_name == 'H2-1' or backend_name == 'H2-1E' or backend_name == 'H2-1LE':
            syntax_checker = 'H2-1SC'
        else:
            syntax_checker = None

        try:
            # Use the compiled reference for cost estimation
            def get_cost():
                return qnx.circuits.cost(
                    circuit_ref=ref_compiled_circuit, 
                    n_shots=shots, 
                    backend_config=config,
                    syntax_checker=syntax_checker
                )
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(get_cost)
                try:
                    solution['cost'] = future.result(timeout=30)  # 30 seconds
                except concurrent.futures.TimeoutError:
                    print('Cost estimation timed out.')
                    solution['cost'] = 0
        except Exception as e:
            print(f"Cost estimation failed: {e}")
            solution['cost'] = 0

        # Step 5: Access properties from the full_compiled_circuit object.
        solution['number_of_qubits'] = compiled_circuit.n_qubits
        solution['circuit_depth'] = compiled_circuit.depth()
        solution['total_gates'] = compiled_circuit.n_gates
        solution['two_qubit_gates'] = compiled_circuit.n_2qb_gates()

        # Step 6: Execute the circuit
        print("Executing job...")
        ref_execute_job = qnx.start_execute_job(
            circuits =[ref_compiled_circuit],  # Must be a list of references
            n_shots=[shots],
            backend_config=config,
            name=f"hhl-ir-execute-{len(b)}x{len(b)}-iter{iteration}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        )
        
        print("Waiting for results...")
        robust_wait_for(ref_execute_job, qnx.jobs.status, target_status="COMPLETED", timeout=None, poll_interval=5)
        ref_result = qnx.jobs.results(ref_execute_job)[0]
        print(f"Execution successful. Job ID: {ref_result.id}")

        result = ref_result.download_result()
        solution['runtime'] = 'N/A with qnexus'

    elif isinstance(backend, AerSimulator):
        # Simulator logic remains unchanged
        print(f"Running on {backend.name}")
        qiskit_circuit = transpile(hhl_circ, backend, optimization_level=3)
        solution['number_of_qubits'] = qiskit_circuit.num_qubits
        solution['circuit_depth'] = qiskit_circuit.depth()
        solution['total_gates'] = qiskit_circuit.size()
        solution['two_qubit_gates'] = qiskit_circuit.num_two_qubit_gates()        
        solution['cost'] = 0
        job = backend.run(qiskit_circuit, shots=shots)
        result = job.result()
        solution['job id'] = job
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