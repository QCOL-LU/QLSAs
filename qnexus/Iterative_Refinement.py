from Quantum_Linear_Solver import quantum_linear_solver
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve, norm
import itertools
from qiskit_aer import AerSimulator
import os
from datetime import datetime

def norm_estimation(A, b, x):
    v = A @ x
    denominator = np.dot(v, v)
    if denominator == 0:
        # If denominator is zero, the vector x is in the null space of A
        # This indicates a degenerate case. Return a small value to continue iteration.
        # In practice, this might indicate the system is ill-conditioned.
        return 1e-10  # Use a smaller value for better numerical stability
    return np.dot(v, b) / denominator

def IR(A, b, precision, max_iter, backend, n_qpe_qubits, shots=1024, noisy=True):
    """
    Iterative Refinement for quantum linear solver.
    Args:
        A: Matrix
        b: Vector
        precision: float
        max_iter: int
        backend: Backend name or AerSimulator
        noisy: bool, optional. If True, enables noisy simulation (default True)
        n_qpe_qubits: int. Number of QPE qubits.
    Returns:
        dict with refined solution, residuals, errors, etc.
    """
    nabla, rho, d = 1, 2, len(A)
    iteration = 0
    x = np.zeros(d)
    csol = solve(A, b)
    res_list, error_list = [], []

    print("IR: Obtaining initial solution...")
    initial_solution = quantum_linear_solver(A, b, backend=backend, shots=shots, iteration=0, noisy=noisy, n_qpe_qubits=n_qpe_qubits)
    x = initial_solution['x']
    r = b - np.dot(A, x)
    error_list.append(norm(csol - x))
    res_list.append(norm(r))
    print(f"Initial residual: {res_list[0]:.4f}, Initial error: {error_list[0]:.4f}\n")
    
    iteration = 1
    while (norm(r) > precision and iteration <= max_iter):
        print(f"IR Iteration: {iteration}")
        new_r = nabla * r
        result = quantum_linear_solver(A, new_r, backend=backend, shots=shots, iteration=iteration, noisy=noisy, n_qpe_qubits=n_qpe_qubits)
        c = result['x']
        alpha = norm_estimation(A, new_r, c)
        x += (alpha / nabla) * c
        
        r = b - np.dot(A, x)
        err = norm(csol - x)
        res = norm(r)
        error_list.append(err)
        res_list.append(res)
        
        print(f"  residual: {res:.4f}, error: {err:.4f}, alpha: {alpha:.4f}")
        print()
        
        if res < 1e-9:
             nabla *= rho
        else:
             nabla = min(rho * nabla, 1 / res)
        iteration += 1

    final_result = {
        'refined_x': x,
        'residuals': res_list,
        'errors': error_list,
        'total_iterations': iteration - 1,
        'initial_solution': initial_solution
    }


    backend_label = backend if isinstance(backend, str) else backend.name
    iterations_range = np.arange(len(res_list))
    size = len(b)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    plt.figure()
    plt.plot(iterations_range, [np.log10(r) for r in res_list], 'o--', label=f'{size}x{size} on {backend_label}')
    plt.ylabel(r"$\log_{10}(\|Ax-b\|_2)$")
    plt.xlabel("IR Iteration")
    plt.legend()
    plt.title("Residual Norm vs. Iteration")
    plt.tight_layout()
    residuals_filename = f"plot_residuals_{backend_label}_{size}x{size}_qpe{n_qpe_qubits}_{timestamp}.png"
    plt.savefig(os.path.join(data_dir, residuals_filename))
    #plt.show()

    plt.figure()
    plt.plot(iterations_range, [np.log10(e) for e in error_list], 'o--', label=f'{size}x{size} on {backend_label}')
    plt.ylabel(r"$\log_{10}(\|x-x_{\mathrm{classical}}\|_2)$")
    plt.xlabel("IR Iteration")
    plt.legend()
    plt.title("Solution Error vs. Iteration")
    plt.tight_layout()
    errors_filename = f"plot_errors_{backend_label}_{size}x{size}_qpe{n_qpe_qubits}_{timestamp}.png"
    plt.savefig(os.path.join(data_dir, errors_filename))
    #plt.show()
        
    return final_result