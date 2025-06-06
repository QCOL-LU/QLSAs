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
*   `--backend`: The Quantinuum backend name (e.g., `H2-2`, `H1-1E`). Default: `H2-2`.
*   `--shots`: The number of shots for each quantum circuit execution. Default: `1024`.
*   `--iterations`: The maximum number of iterations for the iterative refinement. Default: `5`.
*   `--plot`: If included, the script will display plots of the residual and error norms.

### Example with More Options

Run a 4x4 problem on the `H1-1E` backend for 3 iterations and show plots:
```bash
python run_hhl_ir.py --size 4 --backend H1-1E --iterations 3 --plot
```

### Output

The script will:
1.  Print live progress to the console, including job submission status and iteration results.
2.  Generate a `.csv` file with the summary of the results (e.g., `results_H2-2_4x4.csv`).
3.  Display plots of the error and residual norms if the `--plot` flag is used.