import pandas as pd
import numpy as np
from qiskit_aer import AerSimulator
from Iterative_Refinement import norm_estimation, sign_estimation, IR
from Generate_Problem_2 import generate_problem
from Quantum_Linear_Solver import quantum_linear_solver
from HHL_Circuit import hhl_circuit


def generate_quantum_solver_results(
    problem_sizes=[2, 4, 8],
    condition_numbers=[1, 5, 10, 100, 500, 1000],
    sparsity=1,
    is_diagonal=False,
    negativity=0,
    integer_elements=False,
    precision=1e-5,
    max_iter=5,
    t0=2 * np.pi,
    shots=1024,
    include_full_data=False
):
    """
    Generates a DataFrame of quantum solver results over multiple problem sizes and condition numbers.

    Parameters:
    - problem_sizes: list of integers
    - condition_numbers: list of floats
    - include_full_data: if True, keeps A, b, error/residual lists in final DataFrame

    Returns:
    - df: pandas DataFrame of raw data
    - df_display: cleaned DataFrame for display
    """
    backend = AerSimulator()
    df = pd.DataFrame()

    for k in problem_sizes:
        for i in condition_numbers:
            problem = generate_problem(
                n=k,
                cond_number=i,
                sparsity=sparsity,
                is_diagonal=is_diagonal,
                negativity=negativity,
                integer_elements=integer_elements
            )
            A = problem['A']
            b = problem['b']
            solution = quantum_linear_solver(A, b, backend=backend, t0=t0, shots=shots)

            # Iterative Refinement
            refined_solution = IR(A, b, precision=precision, max_iter=max_iter, backend=backend, plot=False)

            datarow = {
                "Backend": backend.name,
                "Problem Size": f"{len(b)} x {len(b)}",
                "A": A,
                "Type": "Dense",
                "b": b,
                "Condition Number": problem["condition_number"],
                "Sparsity": problem["sparsity"],
                "Number of Qubits": solution["number_of_qubits"],
                "Circuit Depth": solution["circuit_depth"],
                "Total Gates": solution["total_gates"],
                # "Two-Qubit Gates": solution.get("two_qubit_gates"),
                # "Runtime": solution.get("runtime"),
                "||x_c - x_q|| without IR": solution["two_norm_error"],
                "||x_c - x_q|| with IR": refined_solution["errors"][-1],
                "||Ax - b|| without IR": solution["residual_error"],
                "||Ax - b|| with IR": refined_solution["residuals"][-1],
                "Total Iterations of IR": refined_solution["total_iterations"],
                "Error list": refined_solution["errors"],
                "Residual list": refined_solution["residuals"],
            }

            df = pd.concat([df, pd.DataFrame([datarow])], ignore_index=True)

    # Drop large columns for display
    if include_full_data:
        df_display = df.copy()
    else:
        df_display = df.drop(columns=["Error list", "Residual list", "A", "b"])

    return df, df_display