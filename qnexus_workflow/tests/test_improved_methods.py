import numpy as np
import matplotlib.pyplot as plt
from Generate_Problem_V2 import generate_problem, generate_problem_banded

def comprehensive_test():
    """Comprehensive test of all methods across different parameters"""
    print("Comprehensive Test of Matrix Generation Methods")
    print("=" * 60)
    
    # Test parameters
    sizes = [4, 8, 16]
    condition_numbers = [2, 5, 10, 20]
    sparsities = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Store results for analysis
    results = {
        'original': {'cond_ratios': [], 'sparsity_errors': []},
        'improved': {'cond_ratios': [], 'sparsity_errors': []},
        'banded': {'cond_ratios': [], 'sparsity_errors': []}
    }
    
    for n in sizes:
        print(f"\nMatrix Size: {n}x{n}")
        print("-" * 40)
        
        for cond_num in condition_numbers:
            for sparsity in sparsities:
                print(f"  Testing: cond={cond_num}, sparsity={sparsity}")
                
                try:
                    # Test original method
                    prob_orig = generate_problem(n, cond_num, sparsity, seed=42)
                    
                    # Test improved method (main function in V2)
                    prob_improved = generate_problem(n, cond_num, sparsity, seed=42)
                    
                    # Test banded method
                    prob_banded = generate_problem_banded(n, cond_num, sparsity, seed=42)
                    
                    # Store results
                    results['original']['cond_ratios'].append(prob_orig['condition_number'] / cond_num)
                    results['original']['sparsity_errors'].append(abs(prob_orig['sparsity'] - sparsity))
                    
                    results['improved']['cond_ratios'].append(prob_improved['condition_number'] / cond_num)
                    results['improved']['sparsity_errors'].append(abs(prob_improved['sparsity'] - sparsity))
                    
                    results['banded']['cond_ratios'].append(prob_banded['condition_number'] / cond_num)
                    results['banded']['sparsity_errors'].append(abs(prob_banded['sparsity'] - sparsity))
                    
                    print(f"    Original: cond_ratio={prob_orig['condition_number']/cond_num:.2f}, sparsity_error={abs(prob_orig['sparsity']-sparsity):.3f}")
                    print(f"    Improved: cond_ratio={prob_improved['condition_number']/cond_num:.2f}, sparsity_error={abs(prob_improved['sparsity']-sparsity):.3f}")
                    print(f"    Banded:  cond_ratio={prob_banded['condition_number']/cond_num:.2f}, sparsity_error={abs(prob_banded['sparsity']-sparsity):.3f}")
                    
                except Exception as e:
                    print(f"    Error: {e}")
                    continue
    
    # Analyze results
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    for method, data in results.items():
        cond_ratios = np.array(data['cond_ratios'])
        sparsity_errors = np.array(data['sparsity_errors'])
        
        print(f"\n{method.upper()} METHOD:")
        print(f"  Condition Number Control:")
        print(f"    Mean ratio: {np.mean(cond_ratios):.3f}")
        print(f"    Std ratio:  {np.std(cond_ratios):.3f}")
        print(f"    Min ratio:  {np.min(cond_ratios):.3f}")
        print(f"    Max ratio:  {np.max(cond_ratios):.3f}")
        
        print(f"  Sparsity Control:")
        print(f"    Mean error: {np.mean(sparsity_errors):.3f}")
        print(f"    Std error:  {np.std(sparsity_errors):.3f}")
        print(f"    Max error:  {np.max(sparsity_errors):.3f}")
    
    return results

def visualize_results(results):
    """Create visualizations of the results"""
    print("\nGenerating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Condition number ratios
    ax1 = axes[0, 0]
    methods = list(results.keys())
    cond_data = [results[method]['cond_ratios'] for method in methods]
    
    bp1 = ax1.boxplot(cond_data, labels=methods, patch_artist=True)
    ax1.set_title('Condition Number Control\n(Ratio of Actual to Target)')
    ax1.set_ylabel('Condition Number Ratio')
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Control')
    ax1.legend()
    
    # Plot 2: Sparsity errors
    ax2 = axes[0, 1]
    sparsity_data = [results[method]['sparsity_errors'] for method in methods]
    
    bp2 = ax2.boxplot(sparsity_data, labels=methods, patch_artist=True)
    ax2.set_title('Sparsity Control\n(Absolute Error from Target)')
    ax2.set_ylabel('Sparsity Error')
    
    # Plot 3: Scatter plot of condition number vs sparsity control
    ax3 = axes[1, 0]
    colors = ['blue', 'green', 'red']
    for i, method in enumerate(methods):
        cond_ratios = np.array(results[method]['cond_ratios'])
        sparsity_errors = np.array(results[method]['sparsity_errors'])
        ax3.scatter(cond_ratios, sparsity_errors, c=colors[i], label=method, alpha=0.6)
    
    ax3.set_xlabel('Condition Number Ratio')
    ax3.set_ylabel('Sparsity Error')
    ax3.set_title('Condition Number vs Sparsity Control')
    ax3.legend()
    ax3.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
    
    # Plot 4: Performance comparison
    ax4 = axes[1, 1]
    # Calculate overall performance score (lower is better)
    performance_scores = []
    for method in methods:
        cond_ratios = np.array(results[method]['cond_ratios'])
        sparsity_errors = np.array(results[method]['sparsity_errors'])
        # Score based on how close to ideal (1.0 for cond, 0.0 for sparsity)
        cond_score = np.mean((cond_ratios - 1.0)**2)
        sparsity_score = np.mean(sparsity_errors**2)
        total_score = cond_score + sparsity_score
        performance_scores.append(total_score)
    
    bars = ax4.bar(methods, performance_scores, color=colors)
    ax4.set_title('Overall Performance Score\n(Lower is Better)')
    ax4.set_ylabel('Performance Score')
    
    # Add value labels on bars
    for bar, score in zip(bars, performance_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('method_comparison.png', dpi=150, bbox_inches='tight')
    print("Method comparison visualization saved as 'method_comparison.png'")
    
    return fig

if __name__ == "__main__":
    # Run comprehensive test
    results = comprehensive_test()
    
    # Generate visualizations
    fig = visualize_results(results)
    
    print("\nTest completed successfully!")
