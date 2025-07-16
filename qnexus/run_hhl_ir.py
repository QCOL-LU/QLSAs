import argparse
import pandas as pd
import numpy as np
import qiskit
import qiskit_aer
import pytket
import qnexus as qnx
import os
from datetime import datetime
import matplotlib.pyplot as plt
import json

from Generate_Problem import generate_problem
from Iterative_Refinement import IR
from Quantum_Linear_Solver import quantum_linear_solver

def main(args):
    """Main execution function."""
    #qnx.login()

    # project_ref = qnx.projects.get_or_create(name="HHL-IR")
    # qnx.context.set_active_project(project_ref)

    # --- Problem Definition ---
    print(f"\n--- Generating {args.size}x{args.size} Problem ---")
    problem = generate_problem(args.size, cond_number=5, sparsity=0.5, seed=1)
    print("Problem Details:")
    print(f"  Condition Number: {problem['condition_number']:.4f}")
    print(f"  Sparsity: {problem['sparsity']:.4f}")
    
    A = problem['A']
    b = problem['b']

    # --- Backend Configuration ---
    # For qnexus, we just need the backend name as a string.
    backend_name = args.backend
    print(f"\nTarget backend: {backend_name}")

    # --- Run HHL with Iterative Refinement ---
    print("\n--- Running HHL with Iterative Refinement ---")
    refined_solution = IR(
        A, b, 
        precision=1e-5, 
        max_iter=args.iterations, 
        backend=backend_name, 
        plot=args.plot
    )
    print("\nRefinement Complete.")

    # --- Results Summary ---
    df = pd.DataFrame()
    initial_solution = refined_solution['initial_solution']
    
    # Prepare summary values
    before_error = initial_solution["two_norm_error"]
    after_error = refined_solution["errors"][-1] if refined_solution["errors"] else None
    before_residual = initial_solution["residual_error"]
    after_residual = refined_solution["residuals"][-1] if refined_solution["residuals"] else None
    error_pct = 100 * (before_error - after_error) / before_error if before_error else None
    residual_pct = 100 * (before_residual - after_residual) / before_residual if before_residual else None

    # Print problem/circuit data as key-value pairs
    print("\n--- Results Summary ---")
    print(f"Backend: {backend_name}")
    print(f"Problem Size: {len(b)} x {len(b)}")
    print(f"Condition Number: {problem['condition_number']:.6f}")
    print(f"Sparsity: {problem['sparsity']:.6f}")
    print(f"Number of Qubits: {initial_solution['number_of_qubits']}")
    print(f"Circuit Depth: {initial_solution['circuit_depth']}")
    print(f"Total Gates: {initial_solution['total_gates']}")
    print(f"Two-Qubit Gates: {initial_solution.get('two_qubit_gates')}")
    print(f"Total Iterations of IR: {refined_solution['total_iterations']}")

    # Print comparison table for IR vs no IR
    print("\nComparison of IR vs No IR:")
    print(f"{'Metric':<30}{'Before IR':>15}{'After IR':>15}{'% Decrease':>15}")
    print(f"{'||x_c - x_q||':<30}{before_error:>15.6f}{after_error:>15.6f}{error_pct:>15.2f}")
    print(f"{'||Ax - b||':<30}{before_residual:>15.6f}{after_residual:>15.6f}{residual_pct:>15.2f}")

    # Prepare datarow for CSV (including lists, but not printing them)
    datarow = {
        "Backend": backend_name,
        "Problem Size": f"{len(b)} x {len(b)}",
        "Condition Number": problem["condition_number"],
        "Sparsity": problem["sparsity"],
        "Number of Qubits": initial_solution["number_of_qubits"],
        "Circuit Depth": initial_solution["circuit_depth"],
        "Total Gates": initial_solution["total_gates"],
        "Two-Qubit Gates": initial_solution.get("two_qubit_gates"),
        "||x_c - x_q|| without IR": before_error,
        "||x_c - x_q|| with IR": after_error,
        "||Ax - b|| without IR": before_residual,
        "||Ax - b|| with IR": after_residual,
        "Total Iterations of IR": refined_solution["total_iterations"],
        "Residuals List": json.dumps(refined_solution["residuals"]),
        "Errors List": json.dumps(refined_solution["errors"]),
    }
    df = pd.concat([df, pd.DataFrame([datarow])], ignore_index=True)
    
    # Save to CSV in data folder with timestamp
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_filename = f"results_{backend_name}_{args.size}x{args.size}_{timestamp}.csv"
    output_filepath = os.path.join(data_dir, output_filename)
    df.to_csv(output_filepath, index=False)
    print(f"\nResults saved to {output_filepath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run HHL with Iterative Refinement on Quantinuum.")
    parser.add_argument('--size', type=int, default=2, help='Size of the linear system (e.g., 2 for 2x2).')
    parser.add_argument('--backend', type=str, default='H1-1E', help='Quantinuum backend name (e.g., H1-1E).')
    parser.add_argument('--shots', type=int, default=1024, help='Number of shots per circuit execution.')
    parser.add_argument('--iterations', type=int, default=5, help='Max iterations for iterative refinement.')
    parser.add_argument('--plot', action='store_true', help='Display plots of error and residual norms.')
    
    args = parser.parse_args()
    main(args)