import numpy as np
from numpy import linalg as LA
from numpy.linalg import cond, eigvals
from scipy.sparse import random as sparse_random
from scipy.sparse import tril, triu
from scipy.linalg import eigh  # eigh is for symmetric/Hermitian matrices
from scipy.sparse import spdiags  # For diagonal sparse matrices

def create_sparse_symmetric(n, density):
    """
    Create a sparse symmetric matrix with random real entries.
    
    Parameters
    ----------
    n : int
        Size of the matrix.
    density : float
        Fraction of non-zero elements (0 to 1).
    
    Returns
    -------
    numpy.ndarray
        Dense symmetric matrix with the specified density.
    """
    # Create a sparse matrix with random real entries
    # Use 'lil' format for easier element-wise manipulation
    # We only create the upper triangle initially to ensure symmetry
    S = sparse_random(n, n, density=density/2, format='lil', dtype=np.float64)

    # Make it symmetric
    for i, j in zip(*S.nonzero()):
        if i < j:
            S[j, i] = S[i, j]  # Just copy for real symmetric
        elif i == j:
            # Ensure diagonal is non-zero if density is very low to avoid issues
            if S[i,j] == 0:
                S[i,j] = np.random.rand() * 10

    # Fill any remaining zero diagonal elements with small random numbers
    # to help prevent zero eigenvalues too easily (unless desired for high cond number)
    for i in range(n):
        if S[i, i] == 0:
            S[i, i] = np.random.rand() * 1e-3  # Small positive value

    return S.toarray()

def generate_problem(n, cond_number=5, sparsity=0.5, seed=1):
    """
    Generate a Hermitian matrix A of size n x n with a specified approximate condition number and sparsity.
    This improved version uses scipy sparse matrices for better sparsity control and condition number preservation.
    
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
    
    # Calculate density (fraction of non-zero elements)
    # sparsity = fraction of zero elements, so density = 1 - sparsity
    density = 1 - sparsity
    
    # Create a sparse symmetric matrix with the desired density
    A_sparse = create_sparse_symmetric(n, density)
    
    # Ensure it's explicitly symmetric after conversion to dense
    A_sparse = (A_sparse + A_sparse.T) / 2
    
    # Get current eigenvalues and condition number
    current_evals = eigh(A_sparse)[0]
    current_cond = np.max(np.abs(current_evals)) / np.min(np.abs(current_evals)) if np.min(np.abs(current_evals)) != 0 else np.inf
    
    # Adjust condition number by shifting eigenvalues
    # Goal: shift eigenvalues so the smallest becomes 'min_target_eigen'
    max_eigen = np.max(current_evals)
    # Ensure max_eigen is not zero if we want a finite target_cond_number
    if max_eigen == 0:
        max_eigen = 1e-6  # Avoid division by zero if all initial evals are zero
    
    min_target_eigen = max_eigen / cond_number
    
    # The shift needed is (min_target_eigen - current_min_eigen)
    # This shift will be added to the diagonal
    shift_value = min_target_eigen - np.min(current_evals)
    
    # Apply the shift to the diagonal
    A_adjusted = A_sparse + shift_value * np.eye(n)
    
    # Normalize the matrix to prevent numerical issues
    beta = max(LA.norm(A_adjusted), LA.norm(b))
    A = A_adjusted / beta
    b = b / beta
    
    # Compute exact solution
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

def generate_problem_banded(n, cond_number=5, sparsity=0.5, seed=None):
    """
    Alternative approach: Generate a banded matrix with controlled condition number.
    This often preserves condition number better than random sparsification.
    """
    if seed is not None:
        np.random.seed(seed)
    
    if not (n > 0 and (n & (n - 1)) == 0):
        raise ValueError(f"Problem size n={n} must be a power of 2 for the HHL algorithm")
    
    b = np.random.randn(n)
    
    # Create a banded matrix structure
    A = np.zeros((n, n))
    
    # Set diagonal elements (maintains condition number)
    diagonal_values = np.linspace(1, cond_number, n)
    np.fill_diagonal(A, diagonal_values)
    
    # Calculate bandwidth based on sparsity
    # Higher sparsity = smaller bandwidth
    max_bandwidth = int(n * (1 - sparsity) / 2)
    bandwidth = max(1, max_bandwidth)
    
    # Fill banded structure
    for i in range(n):
        for j in range(max(0, i - bandwidth), min(n, i + bandwidth + 1)):
            if i != j:  # Skip diagonal
                # Use smaller values for off-diagonal to maintain condition number
                value = np.random.randn() * 0.05 * min(diagonal_values[i], diagonal_values[j])
                A[i, j] = value
    
    # Ensure symmetry
    A = (A + A.T) / 2
    
    # Normalize
    beta = max(LA.norm(A), LA.norm(b))
    A = A / beta
    b = b / beta
    
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

def compare_methods(n=8, cond_number=5, sparsity=0.5, seed=42):
    """Compare different methods for generating sparse matrices"""
    print(f"Comparing methods for n={n}, cond_number={cond_number}, sparsity={sparsity}")
    print("=" * 60)
    
    # Original method
    from Generate_Problem import generate_problem
    problem_orig = generate_problem(n, cond_number, sparsity, seed)
    
    # Improved method (main function in V2)
    problem_improved = generate_problem(n, cond_number, sparsity, seed)
    
    # Banded method
    problem_banded = generate_problem_banded(n, cond_number, sparsity, seed)
    
    methods = [
        ("Original", problem_orig),
        ("Improved", problem_improved),
        ("Banded", problem_banded)
    ]
    
    for name, problem in methods:
        print(f"\n{name} Method:")
        print(f"  Condition Number: {problem['condition_number']:.2f} (target: {cond_number})")
        print(f"  Sparsity: {problem['sparsity']:.3f} (target: {sparsity})")
        print(f"  Non-zero elements: {np.count_nonzero(problem['A'])}/{problem['A'].size}")
        
        # Check solution quality
        A, b, csol = problem['A'], problem['b'], problem['csol']
        residual = LA.norm(A @ csol - b)
        print(f"  Residual: {residual:.2e}")
        
        # Check matrix properties
        is_hermitian = np.allclose(A, A.conj().T)
        print(f"  Hermitian: {is_hermitian}")
    
    return methods

if __name__ == "__main__":
    # Test the improved methods
    compare_methods() 