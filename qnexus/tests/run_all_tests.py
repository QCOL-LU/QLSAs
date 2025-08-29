#!/usr/bin/env python3
"""
Simple test runner for the qnexus HHL quantum linear solver.
Run this script to execute all tests and get a comprehensive analysis.
"""

import sys
import os

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Run all tests and provide analysis."""
    print("Starting comprehensive debug of qnexus HHL quantum linear solver...")
    print("=" * 80)
    
    # Import and run the master debug script
    try:
        from tests.run_debug import main as run_debug_main
        run_debug_main()
    except ImportError as e:
        print(f"Failed to import debug script: {e}")
        print("Make sure you're running this from the qnexus directory with venv activated.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 