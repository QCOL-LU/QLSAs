from Quantum_Linear_Solver import quantum_linear_solver
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve, norm
import itertools
from qiskit_aer import AerSimulator


# Iterative Refinement
def norm_estimation(A, b, x):
    # this function finds scaling factor alpha such that alpha = argmin ||A alpha x - b||_2^2
    v = A @ x
    denominator = np.dot(v, v)
    if denominator == 0:
        # This means v is a zero vector, so we can't compute alpha as intended.
        # You might choose to handle this special case differently:
        # For example, if v is zero, alpha doesn't matter since A*x = 0.
        # Maybe return alpha = 0 or raise an error.
        alpha = 1e-5
        return alpha
    alpha = (np.dot(v, b)) / denominator
    return alpha


def sign_estimation(A, b, x):
    # this function finds solution z such that alpha = argmin ||A z - b||_2^2 and |z|=|x|
    n = len(x)
    matrix = []
    for bits in itertools.product([-1, 1], repeat=n):
        if any(column == list(bits) for column in matrix):
            continue
        matrix.append(list(bits))
    z = np.zeros(n)
    mins = np.infty
    for i in range(len(matrix)):
        t = np.linalg.norm(A@np.multiply(matrix[i], x) - b)
        if t < mins:
            mins = t
            z = np.multiply(matrix[i], x)
    return z


def IR(A, b, precision, max_iter, backend, plot=False):
    # Scaled Iterative Refinement for solving a linear system
    nabla             = 1                             # Scaling factor
    rho               = 2                             # Incremental scaling
    d                 = len(A)                        # Dimension
    iteration         = 1                             # Iteration counter
    x                 = np.zeros(d)                   # Solution
    r                 = b                             # Residual
    con               = np.linalg.cond(A)             # Condition number
    cost              = 0                             # Cost of credits
    csol              = solve(A, b)                   # Classical solution
    res_list=[]
    error_list=[]

    while (norm(r) > precision and iteration <= max_iter):
        print("Iteration:", iteration)
        new_r = nabla*r
        result = quantum_linear_solver(A, new_r, backend=backend, t0=2*np.pi, shots=1024)
        cost += result['cost']
        c = result['x']
        alpha = norm_estimation(A, new_r, c)            # Calculating best norm estimation
        x = x + (alpha/nabla)*c                         # Updating solution
        err = norm(csol - x)
        error_list.append(err)
        r = b - np.dot(A, x)
        res = norm(r)
        if alpha == 0:
            res_list.append(res)
            print("residual:", res)
            print("error:", err)
            print("Alpha is zero, stopping iteration.")
            break
        if res == 0:
            # print("Residual norm is too small, stopping iteration.")
            # res_list.append(res)
            # break
            print("Residual norm is zero!")
            nabla = rho*nabla
        else:
            nabla = min(rho*nabla, 1/(res))  # Updating scaling factor
        # print("Nabla", nabla)
        res_list.append(res)
        print("residual:", res)
        print("error:", err)
        print("alpha:", alpha)
        iteration+=1
        # print("Precision:",LA.norm((res - A*c*alpha)))
        print()

    result = {}
    result['refined_x'] = x
    result['residuals'] = res_list
    result['errors'] = error_list
    result['total_iterations'] = iteration - 1
    result['total_cost'] = cost

    if plot:
        plt.plot(np.array(range(len(res_list))), [np.log10(res_list[i]) for i in range(len(res_list))],'o--',
                 label=f'HHL with IR on {backend.name if isinstance(backend, AerSimulator) else backend.backend_config.device_name}')
        plt.ylabel(r"$\log_{10}(\|Ax-b\|_2)$")
        plt.xlabel("IR Iteration")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.plot(np.array(range(len(error_list))), [np.log10(error_list[i]) for i in range(len(error_list))], 'o--', label=f'HHL with IR on {backend.name if isinstance(backend, AerSimulator) else backend.backend_config.device_name}')
        plt.ylabel(r"$\log_{10}(\|x-x_{\mathrm{classical}}\|_2)$")
        plt.xlabel("IR Iteration")
        plt.legend()
        plt.tight_layout()
        plt.show()
    return result