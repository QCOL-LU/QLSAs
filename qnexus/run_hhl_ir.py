import argparse
import pandas as pd
import numpy as np
import qiskit
import qiskit_aer
import pytket
import qnexus as qnx

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
    print("\n--- Preparing Summary ---")
    df = pd.DataFrame()
    initial_solution = refined_solution['initial_solution']
    
    datarow = {
        "Backend": backend_name,
        "Problem Size": f"{len(b)} x {len(b)}",
        "Condition Number": problem["condition_number"],
        "Sparsity": problem["sparsity"],
        "Number of Qubits": initial_solution["number_of_qubits"],
        "Circuit Depth": initial_solution["circuit_depth"],
        "Total Gates": initial_solution["total_gates"],
        "Two-Qubit Gates": initial_solution.get("two_qubit_gates"),
        "||x_c - x_q|| without IR": initial_solution["two_norm_error"],
        "||x_c - x_q|| with IR": refined_solution["errors"][-1] if refined_solution["errors"] else None,
        "||Ax - b|| without IR": initial_solution["residual_error"],
        "||Ax - b|| with IR": refined_solution["residuals"][-1] if refined_solution["residuals"] else None,
        "Total Iterations of IR": refined_solution["total_iterations"],
    }
    df = pd.concat([df, pd.DataFrame([datarow])], ignore_index=True)

    # --- Display and Save Results ---
    df_display = df.style.hide(axis="index").format(precision=6).set_caption(f"{backend_name} Results for {args.size}x{args.size} problem")
    print("\n--- Results Summary ---")
    print(df_display.to_string())
    
    # Save to CSV
    output_filename = f"results_{backend_name}_{args.size}x{args.size}.csv"
    df.to_csv(output_filename, index=False)
    print(f"\nResults saved to {output_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run HHL with Iterative Refinement on Quantinuum.")
    parser.add_argument('--size', type=int, default=2, help='Size of the linear system (e.g., 2 for 2x2).')
    parser.add_argument('--backend', type=str, default='H1-1E', help='Quantinuum backend name (e.g., H1-1E).')
    parser.add_argument('--shots', type=int, default=1024, help='Number of shots per circuit execution.')
    parser.add_argument('--iterations', type=int, default=5, help='Max iterations for iterative refinement.')
    parser.add_argument('--plot', action='store_true', help='Display plots of error and residual norms.')
    
    args = parser.parse_args()
    main(args)