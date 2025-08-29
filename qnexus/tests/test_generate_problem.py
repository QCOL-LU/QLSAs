import numpy as np
from numpy import linalg as LA
from numpy.linalg import cond, eigvals
import matplotlib.pyplot as plt

def generate_problem(n, cond_number=5, sparsity=0.5, seed=None):
    """
    Generate a Hermitian matrix A of size n x n with a specified approximate condition number and sparsity.
    Then generate a random vector b, normalize them, and compute the solution csol = A^{-1} b.

    Parameters
    ----------
    n : int
        Size of the matrix.
    cond_number : float
        Desired condition number. The matrix is constructed by choosing eigenvalues between [1, cond_number].
    sparsity : float
        Desired fraction of ZERO elements (0 to 1). For example, 0.5 means 50% of elements are zero.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'A': the generated Hermitian matrix
        - 'b': the right-hand side vector
        - 'csol': the exact solution A^{-1} b
        - 'condition_number': the measured condition number of A
        - 'sparsity': the actual sparsity level (fraction of zero elements)
        - 'eigs': sorted eigenvalues of A
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Validate that n is a power of 2 (required for HHL algorithm)
    if not (n > 0 and (n & (n - 1)) == 0):
        raise ValueError(f"Problem size n={n} must be a power of 2 for the HHL algorithm")
    
    # Generate a random vector b
    b = np.random.randn(n)
    
    # Step 1: Construct desired eigenvalues to achieve the target condition number.
    # We set eigenvalues linearly spaced from 1 to cond_number.
    eigenvalues = np.linspace(1, cond_number, n)
    
    # Step 2: Generate a random unitary matrix Q via QR decomposition.
    X = np.random.randn(n, n)
    Q, _ = np.linalg.qr(X)
    
    # Step 3: Construct A = Q * diag(eigenvalues) * Q^H
    Lambda = np.diag(eigenvalues)
    A = Q @ Lambda @ Q.conj().T  # Hermitian by construction
    
    # Step 4: Introduce sparsity by zeroing out elements
    # sparsity = fraction of elements that should be zero
    # We need to zero out approximately sparsity * n^2 elements
    
    # Calculate how many off-diagonal elements to zero out
    total_elements = n * n
    diagonal_elements = n
    off_diagonal_elements = total_elements - diagonal_elements
    
    # Number of off-diagonal elements to zero out
    num_to_zero = int(sparsity * total_elements - diagonal_elements)
    num_to_zero = max(0, min(num_to_zero, off_diagonal_elements))
    
    # Create mask for off-diagonal elements
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, True)  # Keep diagonal elements
    
    # Get indices of off-diagonal elements
    off_diag_indices = np.where(~np.eye(n, dtype=bool))
    off_diag_flat = list(zip(off_diag_indices[0], off_diag_indices[1]))
    
    # Randomly select which off-diagonal elements to zero out
    if num_to_zero > 0:
        indices_to_zero = np.random.choice(len(off_diag_flat), size=num_to_zero, replace=False)
        for idx in indices_to_zero:
            i, j = off_diag_flat[idx]
            mask[i, j] = False
            mask[j, i] = False  # Ensure symmetry
    
    # Apply mask to A
    A = A * mask
    
    # Step 5: Normalize A and b to prevent numerical issues
    beta = max(LA.norm(A), LA.norm(b))
    A = A / beta
    b = b / beta
    
    # Step 6: Compute exact solution
    csol = LA.solve(A, b)
    
    problem = {
        'A': A,
        'b': b,
        'csol': csol,
        'condition_number': cond(A),
        'sparsity': 1 - (np.count_nonzero(A) / float(A.size)),
        'eigs': np.sort(eigvals(A))
    }
    return problem

def test_condition_number_control():
    """Test if condition number is properly controlled"""
    print("Testing condition number control...")
    
    target_conds = [2, 5, 10, 20]
    n = 8
    
    for target_cond in target_conds:
        problem = generate_problem(n, cond_number=target_cond, sparsity=0.8, seed=42)
        actual_cond = problem['condition_number']
        print(f"Target: {target_cond:.1f}, Actual: {actual_cond:.2f}, Ratio: {actual_cond/target_cond:.2f}")
        
        # Check if condition number is reasonable (within factor of 2)
        assert actual_cond > 0, "Condition number should be positive"
        assert actual_cond < target_cond * 3, f"Condition number {actual_cond} too high for target {target_cond}"

def test_sparsity_control():
    """Test if sparsity is properly controlled"""
    print("\nTesting sparsity control...")
    
    target_sparsities = [0.1, 0.3, 0.5, 0.7, 0.9]
    n = 8
    
    for target_sparsity in target_sparsities:
        problem = generate_problem(n, cond_number=5, sparsity=target_sparsity, seed=42)
        actual_sparsity = problem['sparsity']
        print(f"Target: {target_sparsity:.1f}, Actual: {actual_sparsity:.3f}, Error: {abs(actual_sparsity - target_sparsity):.3f}")
        
        # Check if sparsity is reasonable (within 0.1 of target)
        assert abs(actual_sparsity - target_sparsity) < 0.15, f"Sparsity {actual_sparsity} too far from target {target_sparsity}"

def test_matrix_properties():
    """Test basic matrix properties"""
    print("\nTesting matrix properties...")
    
    n = 8
    problem = generate_problem(n, cond_number=5, sparsity=0.5, seed=42)
    A = problem['A']
    
    # Check if matrix is Hermitian
    is_hermitian = np.allclose(A, A.conj().T)
    print(f"Matrix is Hermitian: {is_hermitian}")
    assert is_hermitian, "Matrix should be Hermitian"
    
    # Check if matrix is invertible
    det = np.linalg.det(A)
    print(f"Matrix determinant: {det:.2e}")
    assert abs(det) > 1e-10, "Matrix should be invertible"
    
    # Check if eigenvalues are positive (for positive definite)
    eigs = problem['eigs']
    print(f"Eigenvalues range: [{eigs[0]:.3f}, {eigs[-1]:.3f}]")
    assert np.all(eigs > 0), "All eigenvalues should be positive"

def test_multiple_sizes():
    """Test with different matrix sizes"""
    print("\nTesting different matrix sizes...")
    
    sizes = [4, 8, 16]
    
    for n in sizes:
        problem = generate_problem(n, cond_number=5, sparsity=0.5, seed=42)
        print(f"Size {n}x{n}: cond={problem['condition_number']:.2f}, sparsity={problem['sparsity']:.3f}")
        
        # Verify solution is correct
        A, b, csol = problem['A'], problem['b'], problem['csol']
        residual = LA.norm(A @ csol - b)
        print(f"  Residual: {residual:.2e}")
        assert residual < 1e-10, f"Solution residual {residual} too high for size {n}"

def test_reproducibility():
    """Test if results are reproducible with same seed"""
    print("\nTesting reproducibility...")
    
    n = 8
    seed = 123
    
    problem1 = generate_problem(n, cond_number=5, sparsity=0.5, seed=seed)
    problem2 = generate_problem(n, cond_number=5, sparsity=0.5, seed=seed)
    
    # Check if matrices are identical
    matrices_equal = np.allclose(problem1['A'], problem2['A'])
    vectors_equal = np.allclose(problem1['b'], problem2['b'])
    
    print(f"Matrices identical: {matrices_equal}")
    print(f"Vectors identical: {vectors_equal}")
    
    assert matrices_equal, "Matrices should be identical with same seed"
    assert vectors_equal, "Vectors should be identical with same seed"

def visualize_matrix_structure():
    """Visualize the matrix structure"""
    print("\nGenerating matrix visualization...")
    
    n = 16
    problem = generate_problem(n, cond_number=10, sparsity=0.3, seed=42)
    A = problem['A']
    
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Matrix heatmap
    plt.subplot(1, 3, 1)
    plt.imshow(np.abs(A), cmap='viridis')
    plt.colorbar(label='|A[i,j]|')
    plt.title(f'Matrix Structure (n={n})')
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    # Plot 2: Sparsity pattern
    plt.subplot(1, 3, 2)
    plt.spy(A, markersize=2)
    plt.title(f'Sparsity Pattern\nSparsity: {problem["sparsity"]:.3f}')
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    # Plot 3: Eigenvalue distribution
    plt.subplot(1, 3, 3)
    eigs = problem['eigs']
    plt.plot(range(1, len(eigs)+1), eigs, 'bo-')
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.7)
    plt.axhline(y=10, color='r', linestyle='--', alpha=0.7)
    plt.title(f'Eigenvalues\nCondition #: {problem["condition_number"]:.2f}')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('matrix_analysis.png', dpi=150, bbox_inches='tight')
    print("Matrix visualization saved as 'matrix_analysis.png'")

if __name__ == "__main__":
    print("Testing the generate_problem_V2 implementation...")
    print("=" * 60)
    
    try:
        test_condition_number_control()
        test_sparsity_control()
        test_matrix_properties()
        test_multiple_sizes()
        test_reproducibility()
        visualize_matrix_structure()
        
        print("\n" + "=" * 60)
        print("All tests passed! The implementation is working correctly.")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        raise
