# HHL with Iterative Refinement on Quantinuum using QNexus

This project implements the **HHL (Harrow-Hassidim-Lloyd) algorithm** combined with **Scaled Iterative Refinement** to solve linear systems of equations on Quantinuum's H-Series quantum computers via the `qnexus` library.

## 🎯 Overview

The HHL algorithm is a quantum algorithm for solving linear systems of equations. This implementation enhances the basic HHL algorithm with iterative refinement to improve solution accuracy and handle quantum noise effects.

### Key Features
- **Quantum Linear Solver**: Implements the HHL algorithm for solving Ax = b
- **Iterative Refinement**: Improves solution accuracy through classical post-processing
- **Quantinuum Integration**: Runs on H-Series quantum computers via qnexus
- **Parameter Sweeps**: Automated exploration of QPE qubits and shot counts
- **Comprehensive Analysis**: Detailed error and residual tracking
- **Visualization**: Automatic generation of convergence plots

## 📋 Prerequisites

- Python 3.8+
- Quantinuum account with access to H-Series devices
- qnexus library access

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd qnexus
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Authenticate with Quantinuum
```bash
qnx login
```

## 📦 Dependencies

The project requires the following packages:
- `numpy==1.26.4` - Numerical computations
- `pandas==2.2.3` - Data analysis and CSV handling
- `matplotlib==3.9.2` - Plotting and visualization
- `qiskit==1.2.2` - Quantum circuit construction
- `qiskit-aer==0.15.1` - Quantum simulation
- `pytket==1.39.0` - Circuit compilation and optimization
- `pytket-qiskit==0.56.0` - Qiskit to pytket conversion
- `qnexus` - Quantinuum cloud access

## 🏗️ Project Structure

```
qnexus/
├── README.md                           # This file
├── requirements.txt                     # Python dependencies
├── run_hhl_ir.py                      # Main execution script
├── run_hhl_ir_qpe_sweep.py            # QPE qubits parameter sweep
├── run_hhl_ir_shots_sweep.py          # Shots parameter sweep
├── plot_compare.py                     # Compare multiple results
├── Quantum_Linear_Solver.py            # Core quantum solver
├── Iterative_Refinement.py             # IR implementation
├── HHL_Circuit.py                      # HHL circuit construction
├── Generate_Problem.py                 # Problem generation
├── HHL_Circuit_Old.py                 # Legacy circuit implementation
├── data/                               # Results and plots
│   ├── qpe_sweep/                     # QPE sweep results
│   ├── shots_sweep/                   # Shots sweep results
│   └── *.csv, *.png                   # Individual run results
└── venv/                              # Virtual environment
```

## 🎮 Usage

### Basic Usage

Run a simple 2x2 problem:
```bash
python run_hhl_ir.py --size 2
```

Run a 4x4 problem with custom parameters:
```bash
python run_hhl_ir.py --size 4 --backend H1-1E --iterations 3 --noiseless
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--size` | int | 2 | Problem size (must be power of 2) |
| `--backend` | str | H1-1E | Quantinuum backend name |
| `--shots` | int | 1024 | Number of shots per circuit |
| `--iterations` | int | 5 | Max IR iterations |
| `--qpe-qubits` | int | None | QPE qubits (default: log₂(size)) |
| `--noisy` | flag | True | Enable noisy simulation |
| `--noiseless` | flag | False | Disable noisy simulation |

### Available Backends

**Emulators (for testing):**
- `H1-1E` - H1-1 Emulator
- `H2-1E` - H2-1 Emulator  
- `H2-2E` - H2-2 Emulator

**Hardware (requires access):**
- `H1-1` - H1-1 Quantum Computer
- `H2-1` - H2-1 Quantum Computer
- `H2-2` - H2-2 Quantum Computer

### Parameter Sweeps

**QPE Qubits Sweep:**
```bash
python run_hhl_ir_qpe_sweep.py
```
Sweeps QPE qubits from 1 to 7 for 2x2 problems.

**Shots Sweep:**
```bash
python run_hhl_ir_shots_sweep.py
```
Sweeps shots from 2 to 1024 for 8x8 problems.

### Comparing Results

Compare multiple CSV files:
```bash
python plot_compare.py results_H1-1E_2x2_*.csv
```

## 🔬 Algorithm Details

### HHL Algorithm
The HHL algorithm solves linear systems Ax = b using quantum computing:

1. **State Preparation**: Encode vector b into quantum state
2. **Quantum Phase Estimation**: Estimate eigenvalues of A
3. **Eigenvalue Inversion**: Apply controlled rotations based on eigenvalues
4. **Measurement**: Extract solution vector x

### Iterative Refinement
Classical post-processing to improve quantum solution accuracy:

1. **Initial Solution**: Get approximate solution x₀ from HHL
2. **Residual Calculation**: Compute r = b - Ax₀
3. **Correction**: Solve Aδx = r using HHL
4. **Update**: x₁ = x₀ + δx
5. **Iterate**: Repeat until convergence

### Problem Generation
Generates Hermitian matrices with specified:
- **Condition Number**: Controls eigenvalue spread
- **Sparsity**: Fraction of non-zero off-diagonal elements
- **Size**: Must be power of 2 for HHL algorithm

## 📊 Output and Results

### Console Output
Real-time progress including:
- Problem details (condition number, sparsity)
- Job submission and compilation status
- Iteration progress with residuals and errors
- Final results summary

### Generated Files

**CSV Results:**
- `results_{backend}_{size}x{size}_{timestamp}.csv`
- Contains summary statistics and iteration data

**Plots:**
- `plot_residuals_{backend}_{size}x{size}_{timestamp}.png`
- `plot_errors_{backend}_{size}x{size}_{timestamp}.png`
- Show convergence of residuals and errors

**Sweep Results:**
- `qpe_sweep/` - QPE qubits parameter exploration
- `shots_sweep/` - Shot count parameter exploration
- Contains matrices and heatmaps

### Metrics Tracked

| Metric | Description |
|--------|-------------|
| `||x_c - x_q||` | Error between classical and quantum solutions |
| `||Ax - b||` | Residual norm |
| `Circuit Depth` | Number of circuit layers |
| `Total Gates` | Total number of quantum gates |
| `Two-Qubit Gates` | Number of two-qubit operations |
| `Runtime` | Execution time per iteration |

## 🔧 Configuration

### Environment Variables
- `QNEXUS_API_KEY` - Quantinuum API key (if not using qnx login)

### Backend Configuration
The code automatically configures backends:
- **Emulators**: Enable noisy simulation by default
- **Hardware**: Use standard compilation settings
- **Optimization**: Level 2 optimization for all backends

## 🐛 Troubleshooting

### Common Issues

**1. Authentication Errors**
```bash
qnx login
# Follow prompts to authenticate
```

**2. Backend Not Available**
- Check backend name spelling
- Verify account has access to requested backend
- Try emulator backends for testing

**3. Memory Issues**
- Reduce problem size for large systems
- Use fewer shots for initial testing
- Monitor system memory usage

**4. Job Timeouts**
- Increase timeout in `robust_wait_for` function
- Check backend queue status
- Try different backend if available

**5. Import Errors**
```bash
pip install -r requirements.txt
source venv/bin/activate
```

### Debug Mode
Add debug prints to `Quantum_Linear_Solver.py`:
```python
# In robust_wait_for function
print(f"DEBUG: Status = {status.status}")
```

## 📈 Performance Tips

1. **Start Small**: Begin with 2x2 problems to verify setup
2. **Use Emulators**: Test with H1-1E before hardware
3. **Monitor Convergence**: Check residual plots for proper convergence
4. **Parameter Tuning**: Experiment with QPE qubits and shots
5. **Batch Processing**: Use sweep scripts for systematic exploration

## 🔬 Research Applications

This implementation is suitable for:
- **Quantum Algorithm Research**: HHL algorithm studies
- **Noise Mitigation**: Iterative refinement techniques
- **Parameter Optimization**: QPE qubits and shot count studies
- **Benchmarking**: Quantum vs classical performance comparison
- **Educational**: Learning quantum linear algebra

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Add license information here]

## 🙏 Acknowledgments

- Quantinuum for providing quantum computing access
- Qiskit team for quantum circuit framework
- pytket team for circuit compilation tools

## 📞 Support

For issues and questions:
1. Check this README and troubleshooting section
2. Review console output for error messages
3. Check Quantinuum backend status
4. Contact maintainers with detailed error information

---

**Note**: This implementation is for research and educational purposes. Production use may require additional error handling and optimization.