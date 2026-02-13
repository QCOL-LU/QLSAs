from qlsas.solver import QuantumLinearSolver
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA


class Refiner:
    def __init__(self, 
        A: np.ndarray,
        b: np.ndarray,
        solver: QuantumLinearSolver
    ) -> None:
        self.A = A
        self.b = b
        self.solver = solver


    def refine(self,
        precision: float,
        max_iter: int,
        plot: bool = True,
        verbose: bool = True
    ) -> dict:
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
        A                     = self.A
        b                     = self.b
        nabla                 = 1              # scaling factor
        rho                   = 2              # incremental scaling
        d                     = len(A)         # dimension
        iteration             = 0              # iteration counter
        x                     = np.zeros(d)    # solution
        r                     = b              # residual
        con                   = LA.cond(A)     # condition number
        csol                  = LA.solve(A, b) # classical solution
        csol_normalized       = csol / LA.norm(csol)
        res_list              = []             # residual list
        error_list            = []             # error list
        x_list                = []             # solution list

        # terminition conditions
        while (LA.norm(r) > precision and iteration <= max_iter):
            if verbose:
                print(f"IR Iteration: {iteration}")
            new_r             = nabla * r
            A_normalized      = A / LA.norm(new_r)
            new_r_normalized  = new_r / LA.norm(new_r)
            new_x             = self.solver.solve(A_normalized, new_r_normalized, verbose=verbose) # quantum linear solver result
            alpha             = self.norm_estimation(A, new_r, new_x)
            x                += (alpha / nabla) * new_x # scale x by norm of csol
            x_list.append(x) # append new solution to list
            x_normalized      = x / LA.norm(x)
            r                 = (b - A @ x) # next residual
            assert np.isclose(LA.norm(x_normalized), 1, atol=1e-10), f"x_normalized is not normalized: {LA.norm(x_normalized)}"
            assert np.isclose(LA.norm(csol_normalized), 1, atol=1e-10), f"csol_normalized is not normalized: {LA.norm(csol_normalized)}"
            err               = LA.norm(csol_normalized - x_normalized) # both normalized
            res               = LA.norm(r)
            error_list.append(err)
            res_list.append(res)
            
            if verbose:
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
            'initial_solution': x_list[0]
        }

        if plot:
            backend_label = self.solver.backend.name
            size = len(b)
            iterations_range = np.arange(len(res_list))
            plt.figure()
            plt.plot(iterations_range, [np.log10(r) for r in res_list], 'o--', label=f'{size}x{size} on {backend_label}')
            plt.ylabel(r"$\log_{10}(\|Ax-b\|_2)$")
            plt.xlabel("IR Iteration")
            plt.legend()
            plt.title("Residual Norm vs. Iteration")
            plt.tight_layout()
            # residuals_filename = f"plot_residuals_{backend_label}_{size}x{size}_qpe{n_qpe_qubits}_{timestamp}.png"
            # plt.savefig(os.path.join(data_dir, residuals_filename))
            plt.show()

            plt.figure()
            plt.plot(iterations_range, [np.log10(e) for e in error_list], 'o--', label=f'{size}x{size} on {backend_label}')
            plt.ylabel(r"$\log_{10}(\|x-x_{\mathrm{classical}}\|_2)$")
            plt.xlabel("IR Iteration")
            plt.legend()
            plt.title("Solution Error vs. Iteration")
            plt.tight_layout()
            # errors_filename = f"plot_errors_{backend_label}_{size}x{size}_qpe{n_qpe_qubits}_{timestamp}.png"
            # plt.savefig(os.path.join(data_dir, errors_filename))
            plt.show()
            
        return final_result
    
    def norm_estimation(self, A, b, x):
        v = A @ x
        denominator = np.dot(v, v)
        if denominator == 0:
            # If denominator is zero, the vector x is in the null space of A
            # This indicates a degenerate case. Return a small value to continue iteration.
            # In practice, this might indicate the system is ill-conditioned.
            return 1e-10  # Use a smaller value for better numerical stability
        return np.dot(v, b) / denominator