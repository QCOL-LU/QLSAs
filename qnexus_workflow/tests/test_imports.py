#!/usr/bin/env python3
"""
Test script to verify all imports and dependencies work correctly.
This will help identify any missing dependencies or import issues.
"""

import sys
import traceback

def test_imports():
    """Test all required imports and report any issues."""
    print("=" * 60)
    print("IMPORT TESTING")
    print("=" * 60)
    
    # Core Python libraries
    core_libs = [
        'numpy',
        'pandas', 
        'matplotlib',
        'datetime',
        'os',
        'json',
        'time',
        'concurrent.futures',
        'itertools'
    ]
    
    # Quantum computing libraries
    quantum_libs = [
        'qiskit',
        'qiskit_aer',
        'pytket',
        'pytket.extensions.qiskit'
    ]
    
    # Project-specific imports
    project_modules = [
        'Generate_Problem_V2',
        'HHL_Circuit', 
        'Quantum_Linear_Solver',
        'Iterative_Refinement'
    ]
    
    # External service
    external_libs = [
        'qnexus'
    ]
    
    all_tests = {
        'Core Libraries': core_libs,
        'Quantum Libraries': quantum_libs,
        'Project Modules': project_modules,
        'External Services': external_libs
    }
    
    failed_imports = []
    successful_imports = []
    
    for category, modules in all_tests.items():
        print(f"\n{category}:")
        print("-" * 40)
        
        for module in modules:
            try:
                if module in project_modules:
                    # Import local modules
                    __import__(module)
                else:
                    # Import external modules
                    __import__(module)
                print(f"✓ {module}")
                successful_imports.append(module)
            except ImportError as e:
                print(f"✗ {module}: {e}")
                failed_imports.append((module, str(e)))
            except Exception as e:
                print(f"✗ {module}: {e}")
                failed_imports.append((module, str(e)))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Successful imports: {len(successful_imports)}")
    print(f"Failed imports: {len(failed_imports)}")
    
    if failed_imports:
        print("\nFailed imports:")
        for module, error in failed_imports:
            print(f"  - {module}: {error}")
        return False
    else:
        print("\nAll imports successful! ✓")
        return True

def test_basic_functionality():
    """Test basic functionality of key modules."""
    print("\n" + "=" * 60)
    print("BASIC FUNCTIONALITY TESTING")
    print("=" * 60)
    
    try:
        # Test problem generation
        from Generate_Problem_V2 import generate_problem
        problem = generate_problem(2, cond_number=5, sparsity=0.5, seed=42)
        print("✓ Problem generation works")
        print(f"  - Generated {len(problem['A'])}x{len(problem['A'])} matrix")
        print(f"  - Condition number: {problem['condition_number']:.4f}")
        print(f"  - Sparsity: {problem['sparsity']:.4f}")
        
        # Test HHL circuit creation
        from HHL_Circuit import hhl_circuit
        A = problem['A']
        b = problem['b']
        circuit = hhl_circuit(A, b, n_qpe_qubits=2)
        print("✓ HHL circuit creation works")
        print(f"  - Circuit has {circuit.num_qubits} qubits")
        print(f"  - Circuit depth: {circuit.depth()}")
        
        # Test quantum linear solver (simulator only)
        from qiskit_aer import AerSimulator
        from Quantum_Linear_Solver import quantum_linear_solver
        
        backend = AerSimulator()
        result = quantum_linear_solver(A, b, backend, n_qpe_qubits=2, shots=100)
        print("✓ Quantum linear solver works (simulator)")
        print(f"  - Solution error: {result['two_norm_error']:.6f}")
        print(f"  - Residual error: {result['residual_error']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting comprehensive import and functionality testing...")
    
    import_success = test_imports()
    if import_success:
        func_success = test_basic_functionality()
        if func_success:
            print("\n🎉 All tests passed! The qnexus folder is ready for use.")
        else:
            print("\n❌ Basic functionality tests failed. Check the errors above.")
    else:
        print("\n❌ Import tests failed. Please install missing dependencies.")
    
    print("\n" + "=" * 60) 