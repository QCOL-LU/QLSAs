#!/usr/bin/env python3
"""
Code quality and potential issue detection script.
This will help identify common problems in the codebase.
"""

import ast
import os
import sys
from pathlib import Path

def analyze_file(file_path):
    """Analyze a Python file for potential issues."""
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            tree = ast.parse(content)
    except Exception as e:
        return [f"Failed to parse file: {e}"]
    
    # Check for common issues
    for node in ast.walk(tree):
        # Check for bare except clauses
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                if handler.type is None:
                    issues.append("Bare except clause found - consider specifying exception type")
        
        # Check for unused imports (basic check)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    issues.append(f"Import alias '{alias.name}' as '{alias.asname}' - verify usage")
        
        # Check for magic numbers
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if isinstance(node.value, int) and abs(node.value) > 1000:
                issues.append(f"Large magic number: {node.value}")
            elif isinstance(node.value, float) and abs(node.value) > 100:
                issues.append(f"Large magic number: {node.value}")
    
    return issues

def check_file_structure():
    """Check the overall file structure and organization."""
    print("=" * 60)
    print("FILE STRUCTURE ANALYSIS")
    print("=" * 60)
    
    issues = []
    
    # Check for required files
    required_files = [
        'Generate_Problem_V2.py',
        'HHL_Circuit.py', 
        'Quantum_Linear_Solver.py',
        'Iterative_Refinement.py',
        'run_hhl_ir.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        issues.append(f"Missing required files: {missing_files}")
    else:
        print("✓ All required files present")
    
    # Check for __init__.py
    if not os.path.exists('__init__.py'):
        issues.append("Missing __init__.py file")
    else:
        print("✓ __init__.py present")
    
    # Check for data directory
    if not os.path.exists('data'):
        issues.append("Missing data directory")
    else:
        print("✓ data directory present")
    
    return issues

def check_imports_and_dependencies():
    """Check for import issues and dependency problems."""
    print("\n" + "=" * 60)
    print("IMPORT AND DEPENDENCY ANALYSIS")
    print("=" * 60)
    
    issues = []
    
    # Check each Python file for import issues
    python_files = [f for f in os.listdir('.') if f.endswith('.py') and f != '__init__.py']
    
    for file in python_files:
        print(f"\nAnalyzing {file}:")
        file_issues = analyze_file(file)
        if file_issues:
            for issue in file_issues:
                print(f"  ⚠️  {issue}")
                issues.append(f"{file}: {issue}")
        else:
            print(f"  ✓ No obvious issues found")
    
    return issues

def check_specific_issues():
    """Check for specific known issues in the codebase."""
    print("\n" + "=" * 60)
    print("SPECIFIC ISSUE DETECTION")
    print("=" * 60)
    
    issues = []
    
    # Check Quantum_Linear_Solver.py for specific issues
    if os.path.exists('Quantum_Linear_Solver.py'):
        with open('Quantum_Linear_Solver.py', 'r') as f:
            content = f.read()
            
            # Check for hardcoded backend names
            if 'H1-1E' in content or 'H2-1E' in content or 'H2-2E' in content:
                print("⚠️  Hardcoded emulator backend names found - consider making configurable")
                issues.append("Hardcoded backend names in Quantum_Linear_Solver.py")
            
            # Check for timeout issues
            if 'timeout=None' in content:
                print("⚠️  Infinite timeouts found - consider adding reasonable timeouts")
                issues.append("Infinite timeouts in Quantum_Linear_Solver.py")
            
            # Check for error handling
            if 'except Exception as e:' in content:
                print("✓ Generic exception handling found")
            else:
                print("⚠️  No generic exception handling found")
                issues.append("Missing generic exception handling")
    
    # Check for potential memory issues
    if os.path.exists('run_hhl_ir_qpe_sweep.py'):
        with open('run_hhl_ir_qpe_sweep.py', 'r') as f:
            content = f.read()
            if 'size = 16' in content:
                print("⚠️  Large problem size (16) in sweep - may cause memory issues")
                issues.append("Large problem size in sweep script")
    
    return issues

def check_documentation():
    """Check for documentation issues."""
    print("\n" + "=" * 60)
    print("DOCUMENTATION ANALYSIS")
    print("=" * 60)
    
    issues = []
    
    # Check for README
    if not os.path.exists('README.md'):
        issues.append("Missing README.md file")
        print("❌ No README.md found")
    else:
        print("✓ README.md present")
    
    # Check for docstrings in key files
    key_files = ['Quantum_Linear_Solver.py', 'HHL_Circuit.py', 'Generate_Problem_V2.py']
    
    for file in key_files:
        if os.path.exists(file):
            with open(file, 'r') as f:
                content = f.read()
                if '"""' not in content and "'''" not in content:
                    print(f"⚠️  {file}: No docstrings found")
                    issues.append(f"{file}: Missing docstrings")
                else:
                    print(f"✓ {file}: Has docstrings")
    
    return issues

def main():
    """Run all code quality checks."""
    print("Starting comprehensive code quality analysis...")
    
    all_issues = []
    
    # Run all checks
    structure_issues = check_file_structure()
    all_issues.extend(structure_issues)
    
    import_issues = check_imports_and_dependencies()
    all_issues.extend(import_issues)
    
    specific_issues = check_specific_issues()
    all_issues.extend(specific_issues)
    
    doc_issues = check_documentation()
    all_issues.extend(doc_issues)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if all_issues:
        print(f"Found {len(all_issues)} potential issues:")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
        print("\nRecommendations:")
        print("  1. Review hardcoded values and make them configurable")
        print("  2. Add proper error handling and timeouts")
        print("  3. Consider adding more comprehensive documentation")
        print("  4. Test with smaller problem sizes first")
    else:
        print("🎉 No obvious issues found! Code quality looks good.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main() 