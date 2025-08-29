import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import pandas as pd
from run_hhl_ir import run_hhl_ir

# Settings
size = 16
backend = 'H1-1E'
noisy = False
iterations = 8
qpe_qubits_list = [1, 2, 3, 4, 5, 6, 7, 8]
data_dir = os.path.join(os.path.dirname(__file__), 'data', 'qpe_sweep')
os.makedirs(data_dir, exist_ok=True)

# Prepare matrices: rows=QPE qubits, cols=IR iterations
residuals_matrix = np.zeros((len(qpe_qubits_list), iterations))
errors_matrix = np.zeros((len(qpe_qubits_list), iterations))

emulators = {"H1-1E", "H2-1E", "H2-2E"}
def run_single_qpe(qpe_qubits):
    print(f"===================Running size {size} for QPE qubits = {qpe_qubits}===================")
    if backend in emulators:
        print(f"Noisy: {noisy}")
    result = run_hhl_ir(
        size=size,
        backend=backend,
        shots=1024,
        iterations=iterations,
        qpe_qubits=qpe_qubits,
        noisy=noisy
    )
    residuals = np.array(result['residuals'])
    errors = np.array(result['errors'])
    if len(residuals) < iterations:
        residuals = np.pad(residuals, (0, iterations - len(residuals)), constant_values=np.nan)
    if len(errors) < iterations:
        errors = np.pad(errors, (0, iterations - len(errors)), constant_values=np.nan)
    return residuals[:iterations], errors[:iterations]

if __name__ == "__main__":
    if backend in emulators:
        backend_label = backend+f' (noisy:{noisy})'
    else:
        backend_label = backend

    for i, qpe_qubits in enumerate(qpe_qubits_list):
        residuals, errors = run_single_qpe(qpe_qubits)
        residuals_matrix[i, :] = residuals
        errors_matrix[i, :] = errors

    # Save matrices
    now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    residuals_file = os.path.join(data_dir, f'residuals_matrix_{backend_label}_{size}x{size}_{now}.npy')
    errors_file = os.path.join(data_dir, f'errors_matrix_{backend_label}_{size}x{size}_{now}.npy')
    np.save(residuals_file, residuals_matrix)
    np.save(errors_file, errors_matrix)
    print(f"Saved residuals matrix to {residuals_file}")
    print(f"Saved errors matrix to {errors_file}")

    # Plot colormaps
    fig1, ax1 = plt.subplots()
    c1 = ax1.imshow(residuals_matrix, aspect='auto', origin='lower', cmap='viridis')
    ax1.set_title(f'{size}x{size} on {backend}')
    ax1.set_xlabel('IR Iteration')
    ax1.set_ylabel('QPE Qubits')
    ax1.set_yticks(np.arange(len(qpe_qubits_list)))
    ax1.set_yticklabels(qpe_qubits_list)
    fig1.colorbar(c1, ax=ax1, label='Residual')
    residuals_fig_file = os.path.join(data_dir, f'plot_residuals_matrix_{backend_label}_{size}x{size}_{now}.png')
    fig1.savefig(residuals_fig_file)
    print(f"Saved residuals colormap to {residuals_fig_file}")
    #plt.show()

    fig2, ax2 = plt.subplots()
    c2 = ax2.imshow(errors_matrix, aspect='auto', origin='lower', cmap='magma')
    ax2.set_title(f'{size}x{size} on {backend}')
    ax2.set_xlabel('IR Iteration')
    ax2.set_ylabel('QPE Qubits')
    ax2.set_yticks(np.arange(len(qpe_qubits_list)))
    ax2.set_yticklabels(qpe_qubits_list)
    fig2.colorbar(c2, ax=ax2, label='Error')
    errors_fig_file = os.path.join(data_dir, f'plot_errors_matrix_{backend_label}_{size}x{size}_{now}.png')
    fig2.savefig(errors_fig_file)
    print(f"Saved errors colormap to {errors_fig_file}")
    #plt.show() 