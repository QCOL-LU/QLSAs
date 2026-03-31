"""
Generates matrices A with various structure for benchmarking QLSAs and saves them to problem_instances folder.
run  in command line: python data/problem_instances/generators.py 
"""
import sys
import shutil
from pathlib import Path
import numpy as np

def _find_repo_root(start: Path | None = None) -> Path:
    p = (start or Path(__file__).resolve().parent).resolve()
    for d in (p, *p.parents):
        if (d / ".git").exists() or (d / "pyproject.toml").exists() or (d / "src").exists():
            return d
    return p


_repo_root = _find_repo_root()
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from linear_systems_problems.random_matrix_generator_v3 import generate_problem

problem_sizes = [2, 4, 8, 16, 32, 64]
cond_numbers = [1, 5, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7]
sparsity_values = [0.1, 0.3, 0.5, 0.7, 0.9]
instances_per_combo = 10


OUTPUT_DIR = Path(__file__).resolve().parent / "unstructured_matrices"


def generate_random_matrices_and_save():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR) # delete / empty the directory if it exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for problem_size in problem_sizes:
        for cond_number in cond_numbers:
            for sparsity in sparsity_values:
                # generate 10 random matrices for each combination of problem_size, cond_number, and sparsity
                for i, seed in enumerate(range(instances_per_combo)):
                    prob = generate_problem(problem_size, cond_number, sparsity, seed=seed)
                    A = prob['A']
                    actual_sparsity = float(prob["sparsity"])
                    actual_condition = float(prob["condition_number"])
                    out_path = OUTPUT_DIR / (
                        f"size_{problem_size}_"
                        f"sparsity_{actual_sparsity:.6g}_"
                        f"cond_{actual_condition:.6g}_"
                        f"instance_{i + 1}.npy"
                    )
                    np.save(out_path, A)


if __name__ == "__main__":
    generate_random_matrices_and_save()