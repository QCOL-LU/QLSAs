import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import pandas as pd
from run_hhl_ir import run_hhl_ir

# Settings
size = 8
backend = 'H1-1E'
iterations = 10
shots_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
data_dir = os.path.join(os.path.dirname(__file__), 'data', 'shots_sweep')
os.makedirs(data_dir, exist_ok=True)

# Prepare matrices: rows=Shots, cols=IR iterations
residuals_matrix = np.zeros((len(shots_list), iterations))
errors_matrix = np.zeros((len(shots_list), iterations))

def run_single_shots(shots):
    print(f"===================Running size {size} for shots = {shots}===================")
    result = run_hhl_ir(
        size=size,
        backend=backend,
        shots=shots,
        iterations=iterations,
        noisy=False
    )
    residuals = np.array(result['residuals'])
    errors = np.array(result['errors'])
    if len(residuals) < iterations:
        residuals = np.pad(residuals, (0, iterations - len(residuals)), constant_values=np.nan)
    if len(errors) < iterations:
        errors = np.pad(errors, (0, iterations - len(errors)), constant_values=np.nan)
    return residuals[:iterations], errors[:iterations]

if __name__ == "__main__":
    for i, shots in enumerate(shots_list):
        residuals, errors = run_single_shots(shots)
        residuals_matrix[i, :] = residuals
        errors_matrix[i, :] = errors

    # Save matrices
    now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    residuals_file = os.path.join(data_dir, f'residuals_matrix_{backend}_{size}x{size}_{now}.npy')
    errors_file = os.path.join(data_dir, f'errors_matrix_{backend}_{size}x{size}_{now}.npy')
    np.save(residuals_file, residuals_matrix)
    np.save(errors_file, errors_matrix)
    print(f"Saved residuals matrix to {residuals_file}")
    print(f"Saved errors matrix to {errors_file}")

    # Plot colormaps
    fig1, ax1 = plt.subplots()
    c1 = ax1.imshow(residuals_matrix, aspect='auto', origin='lower', cmap='viridis')
    ax1.set_title(f'{size}x{size} on {backend}')
    ax1.set_xlabel('IR Iteration')
    ax1.set_ylabel('Shots')
    ax1.set_yticks(np.arange(len(shots_list)))
    ax1.set_yticklabels(shots_list)
    fig1.colorbar(c1, ax=ax1, label='Residual')
    residuals_fig_file = os.path.join(data_dir, f'plot_residuals_matrix_{backend}_{size}x{size}_{now}.png')
    fig1.savefig(residuals_fig_file)
    print(f"Saved residuals colormap to {residuals_fig_file}")
    #plt.show()

    fig2, ax2 = plt.subplots()
    c2 = ax2.imshow(errors_matrix, aspect='auto', origin='lower', cmap='magma')
    ax2.set_title(f'{size}x{size} on {backend}')
    ax2.set_xlabel('IR Iteration')
    ax2.set_ylabel('Shots')
    ax2.set_yticks(np.arange(len(shots_list)))
    ax2.set_yticklabels(shots_list)
    fig2.colorbar(c2, ax=ax2, label='Error')
    errors_fig_file = os.path.join(data_dir, f'plot_errors_matrix_{backend}_{size}x{size}_{now}.png')
    fig2.savefig(errors_fig_file)
    print(f"Saved errors colormap to {errors_fig_file}")
    #plt.show() 