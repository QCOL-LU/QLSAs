from Quantum_Linear_Solver import quantum_linear_solver
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve, norm
import itertools
from qiskit_aer import AerSimulator

def norm_estimation(A, b, x):
    v = A @ x
    denominator = np.dot(v, v)
    if denominator == 0:
        return 1e-5
    return np.dot(v, b) / denominator

def IR(A, b, precision, max_iter, backend, plot=False):
    nabla, rho, d = 1, 2, len(A)
    iteration = 0
    x = np.zeros(d)
    csol = solve(A, b)
    res_list, error_list = [], []

    print("IR: Obtaining initial solution...")
    initial_solution = quantum_linear_solver(A, b, backend=backend, shots=1024)
    x = initial_solution['x']
    r = b - np.dot(A, x)
    error_list.append(norm(csol - x))
    res_list.append(norm(r))
    print(f"Initial residual: {res_list[0]:.4f}, Initial error: {error_list[0]:.4f}\n")
    
    iteration = 1
    while (norm(r) > precision and iteration <= max_iter):
        print(f"IR Iteration: {iteration}")
        new_r = nabla * r
        result = quantum_linear_solver(A, new_r, backend=backend, shots=1024)
        c = result['x']
        alpha = norm_estimation(A, new_r, c)
        x += (alpha / nabla) * c
        
        r = b - np.dot(A, x)
        err = norm(csol - x)
        res = norm(r)
        error_list.append(err)
        res_list.append(res)
        
        print(f"  residual: {res:.4f}, error: {err:.4f}, alpha: {alpha:.4f}")
        
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

    if plot:
        backend_label = backend if isinstance(backend, str) else backend.name
        iterations_range = np.arange(len(res_list))
        plt.figure()
        plt.plot(iterations_range, [np.log10(r) for r in res_list], 'o--', label=f'HHL with IR on {backend_label}')
        plt.ylabel(r"$\log_{10}(\|Ax-b\|_2)$")
        plt.xlabel("IR Iteration (0 = Initial HHL)")
        plt.legend()
        plt.title("Residual Norm vs. Iteration")
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.plot(iterations_range, [np.log10(e) for e in error_list], 'o--', label=f'HHL with IR on {backend_label}')
        plt.ylabel(r"$\log_{10}(\|x-x_{\mathrm{classical}}\|_2)$")
        plt.xlabel("IR Iteration (0 = Initial HHL)")
        plt.legend()
        plt.title("Solution Error vs. Iteration")
        plt.tight_layout()
        plt.show()
        
    return final_result