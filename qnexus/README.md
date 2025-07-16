# HHL with Iterative Refinement on Quantinuum using QNexus

This project implements the HHL algorithm combined with Scaled Iterative Refinement to solve linear systems of equations on Quantinuum's H-Series quantum computers via the `qnexus` library.

## Setup

1.  **Install Dependencies:** Make sure all required packages are installed.
    ```bash
    pip install -r requirements.txt
    ```

2.  **Login to Quantinuum:** Authenticate your session to be able to submit jobs to the hardware.
    ```bash
    qnx login
    ```

## Running Experiments

All experiments are run from the `run_hhl_ir.py` script. You can specify the problem size, backend, and other parameters from the command line.

### Basic Usage

To run the 2x2 problem on the H2-2 machine:
```bash
python run_hhl_ir.py --size 2
```

To run the 4x4 problem on the H2-2 machine:
```bash
python run_hhl_ir.py --size 4
```

### Command-Line Arguments

You can see all available options by running:
```bash
python run_hhl_ir.py --help
```

**Key Arguments:**
*   `--size`: The size of the linear system (e.g., `2` for 2x2, `4` for 4x4). Default: `2`.
*   `--backend`: The Quantinuum backend name (e.g., `H2-2`, `H1-1E`). Default: `H1-1E`.
*   `--shots`: The number of shots for each quantum circuit execution. Default: `1024`.
*   `--iterations`: The maximum number of iterations for the iterative refinement. Default: `5`.
*   `--noisy` / `--noiseless`: Control whether to use noisy simulation (default is noisy). Use `--noiseless` to disable noise.

### Example with More Options

Run a 4x4 problem on the `H1-1E` backend for 3 iterations with noiseless simulation:
```bash
python run_hhl_ir.py --size 4 --backend H1-1E --iterations 3 --noiseless
```

### Output

The script will:
1.  Print live progress to the console, including job submission status and iteration results.
2.  Generate a `.csv` file with the summary of the results (e.g., `results_H1-1E_4x4_<timestamp>.csv`) in the `data/` folder.

---

## Example Output

Below is a sample of the console output you might see when running a larger problem (e.g., 8x8 system, 10 IR iterations):

```
--- Generating 8x8 Problem ---
Problem Details:
  Condition Number: 2.2326
  Sparsity: 0.7500

Target backend: H1-1E

--- Running HHL with Iterative Refinement ---
IR: Obtaining initial solution...
Running on H1-1E via qnexus
Uploading circuit 'hhl-circuit-8x8-20250716-023755-iter0'...
Circuit uploaded with ID: ea9fb8f9-63a3-4b14-b0aa-64518bdef1f4
Compiling circuit...
Compilation successful. Compiled circuit ID: 1b14d2a1-9e31-4b7c-9775-52effe35142d
Executing job...
Waiting for results...
Execution successful. Job ID: 60831a04-ccdb-486a-8ff6-cd388af80438
Initial residual: 0.1698, Initial error: 0.5548

IR Iteration: 1
Running on H1-1E via qnexus
Uploading circuit 'hhl-circuit-8x8-20250716-024811-iter1'...
... (output truncated for brevity) ...
IR Iteration: 10
Running on H1-1E via qnexus
Uploading circuit 'hhl-circuit-8x8-20250716-040139-iter10'...
Compiling circuit...
Compilation successful. Compiled circuit ID: cfe8d25b-5461-47a1-b285-a8e54174c464
Executing job...
Waiting for results...
Execution successful. Job ID: 019acbbb-0d22-41d2-adc3-a3cfef9e3425
Execution job runtime: 0:00:22.694631
  residual: 0.0088, error: 0.0313, alpha: 0.7623


Refinement Complete.

--- Results Summary ---
Backend: H1-1E
Problem Size: 8 x 8
Condition Number: 2.232571
Sparsity: 0.750000
Number of Qubits: 7
Circuit Depth: 1287
Total Gates: 1740
Two-Qubit Gates: 714
Total Iterations of IR: 10
Runtime per iteration: None

Comparison of IR vs No IR:
Metric                              Before IR       After IR     % Decrease
||x_c - x_q||                        0.554755       0.031299          94.36
||Ax - b||                           0.169850       0.008765          94.84

Results saved to /path/to/your/qlsas/qnexus/data/results_H1-1E_8x8_<timestamp>.csv
```

### CSV Output

A CSV file is generated in the `data/` folder for each run, containing summary statistics and lists of errors and residuals for each IR iteration. The filename includes the backend, problem size, and a timestamp for easy identification.

---

For further details, see the code in the `qnexus` folder or contact the maintainers.