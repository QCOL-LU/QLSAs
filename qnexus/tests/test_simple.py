#!/usr/bin/env python3
"""
Simple test script for the qnexus HHL quantum linear solver.
This script can be run directly to verify basic functionality.
"""

import sys
import os
import traceback

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test basic imports."""
    print("Testing basic imports...")
    
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import qiskit
        import qiskit_aer
        import pytket
        import qnexus as qnx
        print("✓ All external dependencies imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_local_imports():
    """Test local module imports."""
    print("\nTesting local module imports...")
    
    try:
        from Generate_Problem_V2 import generate_problem
        from HHL_Circuit import hhl_circuit
        from Quantum_Linear_Solver import quantum_linear_solver
        from Iterative_Refinement import IR
        print("✓ All local modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Local import error: {e}")
        return False

def test_problem_generation():
    """Test problem generation."""
    print("\nTesting problem generation...")
    
    try:
        from Generate_Problem_V2 import generate_problem
        
        # Test 2x2 problem
        problem = generate_problem(2, cond_number=5, sparsity=0.5, seed=42)
        
        # Check problem properties
        assert problem['A'].shape == (2, 2), "Matrix A has wrong shape"
        assert len(problem['b']) == 2, "Vector b has wrong length"
        assert problem['condition_number'] > 0, "Condition number should be positive"
        
        print("✓ Problem generation works correctly")
        print(f"  - Matrix shape: {problem['A'].shape}")
        print(f"  - Condition number: {problem['condition_number']:.4f}")
        print(f"  - Sparsity: {problem['sparsity']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Problem generation failed: {e}")
        traceback.print_exc()
        return False

def test_hhl_circuit():
    """Test HHL circuit construction."""
    print("\nTesting HHL circuit construction...")
    
    try:
        from Generate_Problem_V2 import generate_problem
        from HHL_Circuit import hhl_circuit
        
        # Generate test problem
        problem = generate_problem(2, cond_number=5, sparsity=0.5, seed=42)
        A = problem['A']
        b = problem['b']
        
        # Create circuit
        circuit = hhl_circuit(A, b, n_qpe_qubits=2)
        
        # Check circuit properties
        assert circuit.num_qubits > 0, "Circuit should have positive number of qubits"
        assert circuit.depth() > 0, "Circuit should have positive depth"
        
        print("✓ HHL circuit construction works correctly")
        print(f"  - Total qubits: {circuit.num_qubits}")
        print(f"  - Circuit depth: {circuit.depth()}")
        print(f"  - Total gates: {circuit.size()}")
        
        return True
    except Exception as e:
        print(f"✗ HHL circuit construction failed: {e}")
        traceback.print_exc()
        return False

def test_quantum_solver_simulator():
    """Test quantum linear solver with simulator."""
    print("\nTesting quantum linear solver (simulator)...")
    
    try:
        from Generate_Problem_V2 import generate_problem
        from Quantum_Linear_Solver import quantum_linear_solver
        from qiskit_aer import AerSimulator
        
        # Generate test problem
        problem = generate_problem(2, cond_number=5, sparsity=0.5, seed=42)
        A = problem['A']
        b = problem['b']
        
        # Test with simulator
        backend = AerSimulator()
        result = quantum_linear_solver(A, b, backend, n_qpe_qubits=2, shots=100)
        
        # Check result structure
        required_keys = ['x', 'two_norm_error', 'residual_error', 'number_of_qubits', 
                        'circuit_depth', 'total_gates']
        
        for key in required_keys:
            assert key in result, f"Missing key in result: {key}"
        
        print("✓ Quantum linear solver works correctly (simulator)")
        print(f"  - Solution error: {result['two_norm_error']:.6f}")
        print(f"  - Residual error: {result['residual_error']:.6f}")
        print(f"  - Circuit qubits: {result['number_of_qubits']}")
        
        return True
    except Exception as e:
        print(f"✗ Quantum linear solver failed: {e}")
        traceback.print_exc()
        return False

def test_config():
    """Test configuration system."""
    print("\nTesting configuration system...")
    
    try:
        from config import (
            EMULATOR_BACKENDS, HARDWARE_BACKENDS, DEFAULT_BACKEND,
            DEFAULT_SHOTS, DEFAULT_TIMEOUT, validate_backend,
            validate_problem_size, validate_shots
        )
        
        # Test validation functions
        assert validate_backend("H1-1E"), "H1-1E should be valid"
        assert validate_backend("H2-1E"), "H2-1E should be valid"
        assert not validate_backend("INVALID"), "INVALID should not be valid"
        
        assert validate_problem_size(2), "2 should be valid"
        assert validate_problem_size(4), "4 should be valid"
        assert not validate_problem_size(3), "3 should not be valid"
        
        assert validate_shots(1024), "1024 should be valid"
        assert not validate_shots(0), "0 should not be valid"
        
        print("✓ Configuration system works correctly")
        print(f"  - Emulator backends: {EMULATOR_BACKENDS}")
        print(f"  - Hardware backends: {HARDWARE_BACKENDS}")
        print(f"  - Default backend: {DEFAULT_BACKEND}")
        print(f"  - Default shots: {DEFAULT_SHOTS}")
        print(f"  - Default timeout: {DEFAULT_TIMEOUT}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration system failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("SIMPLE FUNCTIONALITY TEST")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Local Imports", test_local_imports),
        ("Problem Generation", test_problem_generation),
        ("HHL Circuit", test_hhl_circuit),
        ("Quantum Solver", test_quantum_solver_simulator),
        ("Configuration", test_config)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The system is working correctly.")
        print("\nNext steps:")
        print("1. Test with different problem sizes")
        print("2. Test with different backends")
        print("3. Run iterative refinement tests")
        print("4. Test with real quantum hardware (if available)")
    else:
        print(f"❌ {total - passed} tests failed. Please review the errors above.")
        print("\nTroubleshooting:")
        print("1. Check that virtual environment is activated")
        print("2. Verify all dependencies are installed")
        print("3. Check Python path and import issues")
        print("4. Review error messages for specific issues")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main() 