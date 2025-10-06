#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import time
import argparse
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import local modules
from Iterative_Refinement import IR

def load_problem_from_name(problem_name, problems_dir="problems"):
    """
    Load a problem instance from the dataset.
    
    Parameters
    ----------
    problem_name : str
        Name of the problem (e.g., "problem_2x2_0.5_10_1")
    problems_dir : str
        Directory containing the problem files
    
    Returns
    -------
    dict
        Problem data with keys: 'A', 'b', 'csol', 'metadata'
    """
    try:
        A = np.load(os.path.join(problems_dir, f"{problem_name}_A.npy"))
        b = np.load(os.path.join(problems_dir, f"{problem_name}_b.npy"))
        csol = np.load(os.path.join(problems_dir, f"{problem_name}_csol.npy"))
        
        with open(os.path.join(problems_dir, f"{problem_name}_metadata.json"), 'r') as f:
            metadata = json.load(f)
        
        return {
            'A': A,
            'b': b,
            'csol': csol,
            'metadata': metadata
        }
    except Exception as e:
        print(f"Error loading problem {problem_name}: {e}")
        return None

def run_hhl_on_problem(problem_name, A, b, backend_name="H1-1E", max_iterations=10):
    """
    Run HHL with iterative refinement on a single problem.
    
    Parameters
    ----------
    problem_name : str
        Name of the problem
    A : numpy.ndarray
        Matrix A
    b : numpy.ndarray
        Vector b
    backend_name : str
        Backend to use
    max_iterations : int
        Maximum number of IR iterations
    
    Returns
    -------
    dict
        Results including residuals and errors
    """
    print(f"Running HHL on {problem_name}...")
    
    try:
        # Run iterative refinement
        result = IR(
            A, 
            b, 
            precision=1e-5,  # Add missing precision parameter
            max_iter=max_iterations,
            backend=backend_name,
            n_qpe_qubits=int(np.log2(len(b))),  
            shots=128,      # Default shots
            noisy=False       # Enable noise for realistic results
        )
        
        return {
            'problem_name': problem_name,
            'residuals': result['residuals'],
            'errors': result['errors'],
            'total_iterations': result['total_iterations'],
            'success': True
        }
        
    except Exception as e:
        print(f"Error running HHL on {problem_name}: {e}")
        return {
            'problem_name': problem_name,
            'residuals': [],
            'errors': [],
            'total_iterations': 0,
            'success': False,
            'error': str(e)
        }

def analyze_dataset(problems_dir="problems", backend_name="H1-1E", max_iterations=10, max_size=None):
    """
    Analyze the entire dataset by running HHL on all problems.
    
    Parameters
    ----------
    problems_dir : str
        Directory containing the problem files
    backend_name : str
        Backend to use for all runs
    max_iterations : int
        Maximum number of IR iterations
    max_size : int, optional
        Maximum problem size to include (e.g., 8 for 2x2, 4x4, 8x8 only)
    
    Returns
    -------
    dict
        Complete analysis results
    """
    print("Starting HHL Dataset Analysis...")
    print("=" * 60)
    
    # Get all problem files
    problem_files = [f for f in os.listdir(problems_dir) if f.endswith('_A.npy')]
    problem_names = [f.replace('_A.npy', '') for f in problem_files]
    
    print(f"Found {len(problem_names)} problems to analyze")
    
    # Group problems by size
    problems_by_size = {}
    for problem_name in problem_names:
        size = int(problem_name.split('x')[1].split('_')[0])
        if size not in problems_by_size:
            problems_by_size[size] = []
        problems_by_size[size].append(problem_name)
    
    print(f"Problem sizes found: {sorted(problems_by_size.keys())}")
    
    # Filter by max_size if specified
    if max_size is not None:
        problems_by_size = {size: problems for size, problems in problems_by_size.items() if size <= max_size}
        print(f"Filtered to sizes ≤ {max_size}: {sorted(problems_by_size.keys())}")
    
    # Run analysis on all problems
    all_results = {}
    start_time = time.time()
    
    for size in sorted(problems_by_size.keys()):
        print(f"\nAnalyzing {size}x{size} problems...")
        size_results = []
        
        for problem_name in problems_by_size[size]:
            # Load problem
            problem_data = load_problem_from_name(problem_name, problems_dir)
            if problem_data is None:
                continue
            
            # Run HHL
            result = run_hhl_on_problem(
                problem_name, 
                problem_data['A'], 
                problem_data['b'],
                backend_name,
                max_iterations
            )
            
            size_results.append(result)
            
            # Small delay to avoid overwhelming the backend
            time.sleep(1)
        
        all_results[size] = size_results
        print(f"Completed {len(size_results)} {size}x{size} problems")
    
    total_time = time.time() - start_time
    print(f"\nAnalysis completed in {total_time:.1f} seconds")
    
    return all_results

def aggregate_results(results):
    """
    Aggregate results by system size for plotting.
    
    Parameters
    ----------
    results : dict
        Results from analyze_dataset
    
    Returns
    -------
    dict
        Aggregated data for plotting
    """
    aggregated = {}
    
    for size, size_results in results.items():
        # Filter successful runs
        successful_runs = [r for r in size_results if r['success']]
        
        if not successful_runs:
            print(f"Warning: No successful runs for size {size}")
            continue
        
        # Get residuals for each iteration
        max_iter = max(len(r['residuals']) for r in successful_runs)
        residuals_by_iteration = [[] for _ in range(max_iter)]
        
        for run in successful_runs:
            residuals = run['residuals']
            for i, residual in enumerate(residuals):
                if i < max_iter:
                    residuals_by_iteration[i].append(residual)
        
        # Calculate statistics
        means = []
        stds = []
        for iter_residuals in residuals_by_iteration:
            if iter_residuals:
                means.append(np.mean(iter_residuals))
                stds.append(np.std(iter_residuals))
            else:
                means.append(np.nan)
                stds.append(np.nan)
        
        aggregated[size] = {
            'iterations': list(range(len(means))),
            'mean_residuals': means,
            'std_residuals': stds,
            'successful_runs': len(successful_runs),
            'total_runs': len(size_results)
        }
    
    return aggregated

def create_high_quality_plot(aggregated_data, save_path="hhl_dataset_analysis.pdf"):
    """
    Create a high-quality, paper-ready plot.
    
    Parameters
    ----------
    aggregated_data : dict
        Aggregated data from aggregate_results
    save_path : str
        Path to save the plot
    """
    # Set up high-quality plotting style
    plt.style.use('default')
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['errorbar.capsize'] = 4
    
    # Create figure with high quality
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Color scheme for different system sizes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    markers = ['o', 's', '^', 'D']  # Circle, Square, Triangle, Diamond
    
    for i, (size, data) in enumerate(sorted(aggregated_data.items())):
        iterations = data['iterations']
        means = data['mean_residuals']
        stds = data['std_residuals']
        
        # Convert to log10 for plotting (matching run_hhl_ir.py format)
        log_means = np.log10(means)
        log_stds = stds / (np.array(means) * np.log(10))  # Error propagation for log10
        
        # Plot with error bars
        ax.errorbar(
            iterations, log_means, yerr=log_stds,
            label=f'{size}×{size} (n={data["successful_runs"]})',
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            linewidth=2.5,
            markersize=8,
            capsize=4,
            capthick=1.5
        )
    
    # Customize plot (matching run_hhl_ir.py format)
    ax.set_xlabel('IR Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$\log_{10}(\|Ax-b\|_2)$', fontsize=14, fontweight='bold')
    
    # Set axis properties (let matplotlib auto-scale like run_hhl_ir.py)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    ax.tick_params(axis='both', which='minor', width=1, length=3)
    
    # Add minor grid
    ax.grid(True, which='minor', alpha=0.2, linestyle=':', linewidth=0.5)
    
    # Customize legend
    legend = ax.legend(
        fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
        edgecolor='black',
        borderpad=0.5
    )
    legend.get_frame().set_linewidth(1.0)

    
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(save_path, format='pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none', 
                pad_inches=0.1)
    
    print(f"High-quality plot saved as: {save_path}")
    
    return fig

def save_analysis_data(results, aggregated_data, save_path="hhl_dataset_analysis_data.json"):
    """
    Save all analysis data in JSON format.
    
    Parameters
    ----------
    results : dict
        Raw results from analyze_dataset
    aggregated_data : dict
        Aggregated data for plotting
    save_path : str
        Path to save the JSON file
    """
    # Prepare data for JSON serialization
    json_data = {
        'analysis_info': {
            'timestamp': datetime.now().isoformat(),
            'backend': 'H1-1E',
            'max_iterations': 5,
            'total_problems': sum(len(size_results) for size_results in results.values()),
            'successful_problems': sum(
                sum(1 for r in size_results if r['success']) 
                for size_results in results.values()
            )
        },
        'raw_results': results,
        'aggregated_data': aggregated_data
    }
    
    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    
    print(f"Analysis data saved as: {save_path}")

def main():
    """Main function to run the complete analysis."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='HHL Dataset Analysis with Iterative Refinement')
    parser.add_argument('--max-size', type=int, default=None, 
                       help='Maximum problem size to include (e.g., 8 for 2x2, 4x4, 8x8 only)')
    parser.add_argument('--backend', type=str, default='H1-1E',
                       help='Backend to use (default: H1-1E)')
    parser.add_argument('--max-iterations', type=int, default=5,
                       help='Maximum number of IR iterations (default: 5)')
    parser.add_argument('--problems-dir', type=str, default='problems',
                       help='Directory containing problem files (default: problems)')
    
    args = parser.parse_args()
    
    print("HHL Dataset Analysis - Complete Dataset Evaluation")
    print("=" * 60)
    
    # Configuration
    problems_dir = args.problems_dir
    backend_name = args.backend
    max_iterations = args.max_iterations
    max_size = args.max_size
    
    if max_size is not None:
        print(f"Running analysis on problems up to {max_size}x{max_size}")
    else:
        print("Running analysis on all problem sizes")
    
    # Check if problems directory exists
    if not os.path.exists(problems_dir):
        print(f"Error: {problems_dir} directory not found!")
        print("Please run the dataset generator first.")
        return
    
    # Create output directory
    if max_size is not None:
        output_dir = f"hhl_analysis_results_subset_{max_size}"
    else:
        output_dir = "hhl_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run analysis
    print("Starting HHL analysis on all dataset problems...")
    results = analyze_dataset(problems_dir, backend_name, max_iterations, max_size=max_size)
    
    # Aggregate results
    print("\nAggregating results...")
    aggregated_data = aggregate_results(results)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create high-quality plot
    print("\nCreating high-quality plot...")
    if max_size is not None:
        plot_filename = f"hhl_dataset_analysis_subset_{max_size}_{timestamp}.pdf"
    else:
        plot_filename = f"hhl_dataset_analysis_{timestamp}.pdf"
    plot_path = os.path.join(output_dir, plot_filename)
    fig = create_high_quality_plot(aggregated_data, plot_path)
    
    # Save all data
    print("\nSaving analysis data...")
    if max_size is not None:
        data_filename = f"hhl_dataset_analysis_subset_{max_size}_{timestamp}.json"
    else:
        data_filename = f"hhl_dataset_analysis_data_{timestamp}.json"
    data_path = os.path.join(output_dir, data_filename)
    save_analysis_data(results, aggregated_data, data_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    
    for size in sorted(aggregated_data.keys()):
        data = aggregated_data[size]
        print(f"{size}×{size}: {data['successful_runs']}/{data['total_runs']} successful")
    
    print(f"\nResults saved in: {output_dir}/")
    print(f"  - {plot_filename} (high-quality plot)")
    print(f"  - {data_filename} (complete data)")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    main()
