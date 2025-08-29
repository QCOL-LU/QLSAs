#!/usr/bin/env python3
"""
Comprehensive functionality testing for the HHL quantum linear solver.
Tests all major components and their interactions.
"""

import numpy as np
import traceback
from datetime import datetime

def test_problem_generation():
    """Test the problem generation functionality."""
    print("=" * 60)
    print("TESTING PROBLEM GENERATION")
    print("=" * 60)
    
    try:
        from Generate_Problem_V2 import generate_problem
        
        # Test different problem sizes
        sizes = [2, 4, 8]
        for size in sizes:
            problem = generate_problem(size, cond_number=5, sparsity=0.5, seed=42)
            
            # Verify matrix properties
            A = problem['A']
            b = problem['b']
            
            # Check dimensions
            assert A.shape == (size, size), f"Matrix A has wrong shape: {A.shape}"
            assert len(b) == size, f"Vector b has wrong length: {len(b)}"
            
            # Check Hermitian property
            assert np.allclose(A, A.T.conjugate()), "Matrix A is not Hermitian"
            
            # Check condition number
            assert problem['condition_number'] > 0, "Condition number should be positive"
            
            # Check sparsity
            assert 0 <= problem['sparsity'] <= 1, "Sparsity should be between 0 and 1"
            
            print(f"✓ {size}x{size} problem generated successfully")
            print(f"  - Condition number: {problem['condition_number']:.4f}")
            print(f"  - Sparsity: {problem['sparsity']:.4f}")
            print(f"  - Eigenvalues: {problem['eigs']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Problem generation test failed: {e}")
        traceback.print_exc()
        return False

def test_hhl_circuit():
    """Test the HHL circuit construction."""
    print("\n" + "=" * 60)
    print("TESTING HHL CIRCUIT CONSTRUCTION")
    print("=" * 60)
    
    try:
        from Generate_Problem_V2 import generate_problem
        from HHL_Circuit import hhl_circuit
        
        # Generate a test problem
        problem = generate_problem(2, cond_number=5, sparsity=0.5, seed=42)
        A = problem['A']
        b = problem['b']
        
        # Test different QPE qubit counts
        qpe_qubits_list = [1, 2, 3]
        
        for n_qpe_qubits in qpe_qubits_list:
            circuit = hhl_circuit(A, b, n_qpe_qubits=n_qpe_qubits)
            
            # Check circuit properties
            assert circuit.num_qubits > 0, "Circuit should have positive number of qubits"
            assert circuit.depth() > 0, "Circuit should have positive depth"
            
            # Check that circuit has measurements
            has_measurements = any(isinstance(op, circuit.__class__.measure) 
                                 for op in circuit.data)
            assert has_measurements, "Circuit should have measurement operations"
            
            print(f"✓ HHL circuit with {n_qpe_qubits} QPE qubits constructed successfully")
            print(f"  - Total qubits: {circuit.num_qubits}")
            print(f"  - Circuit depth: {circuit.depth()}")
            print(f"  - Total gates: {circuit.size()}")
        
        return True
        
    except Exception as e:
        print(f"✗ HHL circuit test failed: {e}")
        traceback.print_exc()
        return False

def test_quantum_linear_solver_simulator():
    """Test the quantum linear solver with simulator backend."""
    print("\n" + "=" * 60)
    print("TESTING QUANTUM LINEAR SOLVER (SIMULATOR)")
    print("=" * 60)
    
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
        
        # Check solution properties
        assert len(result['x']) == len(b), "Solution vector has wrong length"
        assert result['two_norm_error'] >= 0, "Error should be non-negative"
        assert result['residual_error'] >= 0, "Residual should be non-negative"
        
        print("✓ Quantum linear solver (simulator) test passed")
        print(f"  - Solution error: {result['two_norm_error']:.6f}")
        print(f"  - Residual error: {result['residual_error']:.6f}")
        print(f"  - Circuit qubits: {result['number_of_qubits']}")
        print(f"  - Circuit depth: {result['circuit_depth']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Quantum linear solver test failed: {e}")
        traceback.print_exc()
        return False

def test_iterative_refinement():
    """Test the iterative refinement functionality."""
    print("\n" + "=" * 60)
    print("TESTING ITERATIVE REFINEMENT")
    print("=" * 60)
    
    try:
        from Generate_Problem_V2 import generate_problem
        from Iterative_Refinement import IR
        from qiskit_aer import AerSimulator
        
        # Generate test problem
        problem = generate_problem(2, cond_number=5, sparsity=0.5, seed=42)
        A = problem['A']
        b = problem['b']
        
        # Test with simulator backend
        backend = AerSimulator()
        
        # Run iterative refinement with small parameters for testing
        result = IR(A, b, precision=1e-3, max_iter=3, backend=backend, 
                   n_qpe_qubits=2, shots=50, noisy=False)
        
        # Check result structure
        required_keys = ['refined_x', 'residuals', 'errors', 'total_iterations', 
                        'initial_solution']
        
        for key in required_keys:
            assert key in result, f"Missing key in IR result: {key}"
        
        # Check that residuals and errors are lists
        assert isinstance(result['residuals'], list), "Residuals should be a list"
        assert isinstance(result['errors'], list), "Errors should be a list"
        
        # Check that we have at least one iteration
        assert len(result['residuals']) > 0, "Should have at least one residual"
        assert len(result['errors']) > 0, "Should have at least one error"
        
        # Check that errors and residuals are non-negative
        for residual in result['residuals']:
            assert residual >= 0, "Residuals should be non-negative"
        
        for error in result['errors']:
            assert error >= 0, "Errors should be non-negative"
        
        print("✓ Iterative refinement test passed")
        print(f"  - Total iterations: {result['total_iterations']}")
        print(f"  - Initial residual: {result['residuals'][0]:.6f}")
        print(f"  - Final residual: {result['residuals'][-1]:.6f}")
        print(f"  - Initial error: {result['errors'][0]:.6f}")
        print(f"  - Final error: {result['errors'][-1]:.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Iterative refinement test failed: {e}")
        traceback.print_exc()
        return False

def test_integration():
    """Test the full integration of all components."""
    print("\n" + "=" * 60)
    print("TESTING FULL INTEGRATION")
    print("=" * 60)
    
    try:
        from run_hhl_ir import run_hhl_ir
        from qiskit_aer import AerSimulator
        
        # Test with simulator backend
        backend = AerSimulator()
        
        # Run a small test
        result = run_hhl_ir(size=2, backend=backend, shots=50, iterations=2, 
                           qpe_qubits=2, noisy=False)
        
        # Check that we got results
        assert 'residuals' in result, "Missing residuals in result"
        assert 'errors' in result, "Missing errors in result"
        assert 'datarow' in result, "Missing datarow in result"
        
        print("✓ Full integration test passed")
        print(f"  - Residuals: {len(result['residuals'])} iterations")
        print(f"  - Errors: {len(result['errors'])} iterations")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling and edge cases."""
    print("\n" + "=" * 60)
    print("TESTING ERROR HANDLING")
    print("=" * 60)
    
    try:
        from Generate_Problem_V2 import generate_problem
        from HHL_Circuit import hhl_circuit
        
        # Test invalid problem size
        try:
            problem = generate_problem(3, cond_number=5, sparsity=0.5, seed=42)
            print("✗ Should have raised error for non-power-of-2 size")
            return False
        except ValueError:
            print("✓ Correctly rejected non-power-of-2 problem size")
        
        # Test invalid sparsity
        try:
            problem = generate_problem(2, cond_number=5, sparsity=1.5, seed=42)
            print("✗ Should have raised error for invalid sparsity")
            return False
        except Exception:
            print("✓ Correctly handled invalid sparsity")
        
        # Test invalid condition number
        try:
            problem = generate_problem(2, cond_number=0, sparsity=0.5, seed=42)
            print("✗ Should have raised error for invalid condition number")
            return False
        except Exception:
            print("✓ Correctly handled invalid condition number")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all functionality tests."""
    print("Starting comprehensive functionality testing...")
    
    tests = [
        ("Problem Generation", test_problem_generation),
        ("HHL Circuit", test_hhl_circuit),
        ("Quantum Linear Solver", test_quantum_linear_solver_simulator),
        ("Iterative Refinement", test_iterative_refinement),
        ("Integration", test_integration),
        ("Error Handling", test_error_handling)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} test PASSED")
            else:
                print(f"✗ {test_name} test FAILED")
        except Exception as e:
            print(f"✗ {test_name} test FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All functionality tests passed! The system is working correctly.")
    else:
        print(f"❌ {total - passed} tests failed. Please review the errors above.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main() 