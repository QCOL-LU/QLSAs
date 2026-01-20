import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot_interrupted_data():
    """
    Plot the interrupted HHL IR data for 8x8 system on H1-1 backend.
    """
    # Data from the interrupted run
    residuals = [0.2185, 0.1351, 0.0894, 0.0543, 0.0418, 0.0277, 0.0191, 0.0075, 0.0040]
    errors = [0.7483, 0.4176, 0.2516, 0.1731, 0.1116, 0.0682, 0.0472, 0.0224, 0.0108]
    
    # Parameters
    backend_label = "H1-1"  # Assuming noisy simulation
    size = 8
    n_qpe_qubits = 3
    shots = 64
    
    # Create iterations range (0 to 8, since we have 9 data points including initial)
    iterations_range = np.arange(len(residuals))
    
    # Create timestamp and data directory
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
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
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 8
    
    # Plot residuals
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations_range, [np.log10(r) for r in residuals], 'o--', label=f'{size}x{size} on {backend_label}', 
            linewidth=2, markersize=8, markeredgewidth=1.5)
    ax.set_ylabel(r"$\log_{10}(\|Ax-b\|_2)$", fontsize=14)
    ax.set_xlabel("IR Iteration", fontsize=14)
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax.set_title("Residual Norm vs. Iteration", fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    residuals_filename = f"plot_residuals_{backend_label}_{size}x{size}_qpe{n_qpe_qubits}_{timestamp}.png"
    plt.savefig(os.path.join(data_dir, residuals_filename), dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved residuals plot: {residuals_filename}")
    
    # Plot errors
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations_range, [np.log10(e) for e in errors], 'o--', label=f'{size}x{size} on {backend_label}', 
            linewidth=2, markersize=8, markeredgewidth=1.5)
    ax.set_ylabel(r"$\log_{10}(\|x-x_{\mathrm{classical}}\|_2)$", fontsize=14)
    ax.set_xlabel("IR Iteration", fontsize=14)
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax.set_title("Solution Error vs. Iteration", fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    errors_filename = f"plot_errors_{backend_label}_{size}x{size}_qpe{n_qpe_qubits}_{timestamp}.png"
    plt.savefig(os.path.join(data_dir, errors_filename), dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved errors plot: {errors_filename}")
    
    # Print summary statistics
    print(f"\nSummary of run:")
    print(f"System size: {size}x{size}")
    print(f"Backend: {backend_label}")
    print(f"QPE qubits: {n_qpe_qubits}")
    print(f"Shots: {shots}")
    print(f"Total iterations completed: {len(residuals)-1}")
    print(f"Final residual: {residuals[-1]:.6f}")
    print(f"Final error: {errors[-1]:.6f}")
    print(f"Residual improvement: {residuals[0]/residuals[-1]:.2f}x")
    print(f"Error improvement: {errors[0]/errors[-1]:.2f}x")
    
    # Save data as numpy arrays for future analysis
    data_filename = f"data_{backend_label}_{size}x{size}_qpe{n_qpe_qubits}_{timestamp}.npz"
    np.savez(os.path.join(data_dir, data_filename),
              residuals=np.array(residuals),
              errors=np.array(errors),
              iterations=iterations_range,
              backend=backend_label,
              size=size,
              n_qpe_qubits=n_qpe_qubits,
              shots=shots)
    print(f"Saved data arrays: {data_filename}")

if __name__ == "__main__":
    plot_interrupted_data() 