from cudaq_qlsa.solver import QuantumLinearSolver
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
        plot: bool = True
    ) -> dict:
        """
        Iterative Refinement for quantum linear solver.
        Args:
            A: Matrix
            b: Vector
            precision: float
            max_iter: int
            backend: Backend name
            noisy: bool, optional. If True, enables noisy simulation (default True)
            n_qpe_qubits: int. Number of QPE qubits.
        Returns:
            dict with refined solution, residuals, errors, etc.
        """
        A                = self.A
        b                = self.b
        nabla            = 1              # scaling factor
        rho              = 2              # incremental scaling
        d                = len(A)         # dimension
        iteration        = 0              # iteration counter
        x                = np.zeros(d)    # solution

        r                = b              # residual
        con              = LA.cond(A)     # condition number
        csol             = LA.solve(A, b) # classical solution
        csol_normalized  = csol / LA.norm(csol)
        res_list         = []             # residual list
        error_list       = []             # error list
        x_list           = []             # solution list

        # Obtaining initial solution
        assert np.isclose(LA.norm(r), 1, atol=1e-10), f"r(b) is not normalized: {LA.norm(r)}"
        A_normalized = A / LA.norm(r)
        x = self.solver.solve(A_normalized, r)

        for idx in range(len(csol)):
            if np.sign(csol[idx]) != np.sign(x[idx]):
                x[idx] = (-1)*x[idx]
        
        x_post = x * LA.norm(csol)
        
        r = (b - A@x_post)
    
        nabla = 1/LA.norm(r)
    
        assert np.isclose(LA.norm(x), 1, atol=1e-10), f"x is not normalized: {LA.norm(x)}"
        assert np.isclose(LA.norm(csol_normalized), 1, atol=1e-10), f"csol_normalized is not normalized: {LA.norm(csol_normalized)}"
    
        error_list.append(LA.norm(csol - x_post)) # both normalized
        res_list.append(LA.norm(r))
        print(f"Initial residual: {res_list[0]:.4f}, Initial error: {error_list[0]:.4f}\n")
    
        iteration = 1
    
        while (LA.norm(r) > precision and iteration <= max_iter):
            print(f"IR Iteration: {iteration}")
            new_r = nabla*r

            A_normalized = A / LA.norm(new_r)
            new_r_normalized = new_r / LA.norm(new_r)

            x_new = self.solver.solve(A_normalized, new_r_normalized)

            csol_ir = LA.solve(A, new_r)
            csol_ir_normalized = csol_ir / LA.norm(csol_ir)
            
            for idx in range(len(csol_ir)):
                if np.sign(csol_ir[idx]) != np.sign(x_new[idx]):
                    x_new[idx] = (-1)*x_new[idx]

            x_newpost = x_new * LA.norm(csol_ir)
                    
            sys_res = LA.norm(new_r - A*x_newpost)
            print(f"  IR System residual: {sys_res:.4f}")
            
            x_post = x_post + (1/nabla)*x_newpost                             # Updating solution
            x_normalized = x_post/LA.norm(x_post)
            
            r = (b- A@x_post)
            assert np.isclose(LA.norm(x_normalized), 1, atol=1e-10), f"x_normalized is not normalized: {LA.norm(x_normalized)}"
            assert np.isclose(LA.norm(csol_ir_normalized), 1, atol=1e-10), f"csol_normalized is not normalized: {LA.norm(csol_normalized)}"
            err = LA.norm(csol_ir - x_newpost) # both normalized
            res = LA.norm(r)
            error_list.append(err)
            res_list.append(res)
            
            print(f"  residual: {res:.2e}, error: {err:.2e}")
            print(".............................................................")
            
            if res < 1e-9:
                nabla *= rho
            else:
                nabla = min(rho * nabla, 1 / res)
            iteration += 1
            
            final_result = {
                'refined_x': x_post,
                'residuals': res_list,
                'errors': error_list,
                'total_iterations': iteration,
                'initial_solution': x
            }

        if plot:
            backend_label = self.solver.backend
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