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

def run_hhl_ir(size=2, backend='H1-1E', shots=1024, iterations=5, qpe_qubits=None, noisy=True):
    """Run HHL with Iterative Refinement and return results as a dictionary."""
    # --- Problem Definition ---
    problem = generate_problem(size, cond_number=5, sparsity=0.5, seed=2)
    A = problem['A']
    b = problem['b']

    print("Problem Details:")
    print(f"  Condition Number: {problem['condition_number']:.4f}")
    print(f"  Sparsity: {problem['sparsity']:.4f}")

    print(f"\nTarget backend: {backend}")
    

    if qpe_qubits is None:
        qpe_qubits = int(np.log2(size))

    # --- Run HHL with Iterative Refinement ---
    print("\n--- Running HHL with Iterative Refinement ---")
    refined_solution = IR(
        A, b, 
        precision=1e-5, 
        max_iter=iterations, 
        backend=backend, 
        shots=shots,
        noisy=noisy,
        n_qpe_qubits=qpe_qubits
    )
    print("\nRefinement Complete.")

    # --- Prepare Results ---
    df = pd.DataFrame()
    initial_solution = refined_solution['initial_solution']
    before_error = initial_solution["two_norm_error"]
    after_error = refined_solution["errors"][-1] if refined_solution["errors"] else None
    before_residual = initial_solution["residual_error"]
    after_residual = refined_solution["residuals"][-1] if refined_solution["residuals"] else None
    error_pct = 100 * (before_error - after_error) / before_error if before_error else None
    residual_pct = 100 * (before_residual - after_residual) / before_residual if before_residual else None

     # Print problem/circuit data as key-value pairs
    print("\n--- Results Summary ---")
    print(f"Backend: {backend}")
    print(f"Problem Size: {len(b)} x {len(b)}")
    print(f"Condition Number: {problem['condition_number']:.6f}")
    print(f"Sparsity: {problem['sparsity']:.6f}")
    print(f"Number of Qubits: {initial_solution['number_of_qubits']}")
    print(f"Circuit Depth: {initial_solution['circuit_depth']}")
    print(f"Total Gates: {initial_solution['total_gates']}")
    print(f"Two-Qubit Gates: {initial_solution.get('two_qubit_gates')}")
    print(f"Total Iterations of IR: {refined_solution['total_iterations']}")
    print(f"Runtime per iteration: {initial_solution.get('runtime')}")


    # Print comparison table for IR vs no IR
    print("\nComparison of IR vs No IR:")
    print(f"{'Metric':<30}{'Before IR':>15}{'After IR':>15}{'% Decrease':>15}")
    print(f"{'||x_c - x_q||':<30}{before_error:>15.6f}{after_error:>15.6f}{error_pct:>15.2f}")
    print(f"{'||Ax - b||':<30}{before_residual:>15.6f}{after_residual:>15.6f}{residual_pct:>15.2f}")

    # Prepare datarow for CSV (including lists, but not printing them)
    datarow = {
        "Backend": backend,
        "Problem Size": f"{len(b)} x {len(b)}",
        "Condition Number": problem["condition_number"],
        "Sparsity": problem["sparsity"],
        "Number of Qubits": initial_solution["number_of_qubits"],
        "Circuit Depth": initial_solution["circuit_depth"],
        "Total Gates": initial_solution["total_gates"],
        "Two-Qubit Gates": initial_solution.get("two_qubit_gates"),
        "Iteration Runtime": initial_solution.get("runtime"),
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
    output_filename = f"results_{backend}_{size}x{size}_{timestamp}.csv"
    output_filepath = os.path.join(data_dir, output_filename)
    df.to_csv(output_filepath, index=False)
    print(f"\nResults saved to {output_filepath}")

    # Return results as a dictionary
    return {
        "residuals": refined_solution["residuals"],
        "errors": refined_solution["errors"],
        "datarow": datarow
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run HHL with Iterative Refinement on Quantinuum.")
    parser.add_argument('--size', type=int, default=2, help='Size of the linear system (e.g., 2 for 2x2).')
    parser.add_argument('--backend', type=str, default='H1-1E', help='Quantinuum backend name (e.g., H1-1E).')
    parser.add_argument('--shots', type=int, default=1024, help='Number of shots per circuit execution.')
    parser.add_argument('--iterations', type=int, default=5, help='Max iterations for iterative refinement.')
    parser.add_argument('--qpe-qubits', type=int, default=None, help='Number of QPE (phase estimation) qubits. Default: log_2(problem size).')
    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument('--noisy', dest='noisy', action='store_true', help='Enable noisy simulation (default).')
    noise_group.add_argument('--noiseless', dest='noisy', action='store_false', help='Disable noisy simulation.')
    parser.set_defaults(noisy=True)
    args = parser.parse_args()
    
    run_hhl_ir(
        size=args.size,
        backend=args.backend,
        shots=args.shots,
        iterations=args.iterations,
        qpe_qubits=args.qpe_qubits,
        noisy=args.noisy
    )