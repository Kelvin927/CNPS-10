#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNPS-10 Test Runner
Main test execution script for the Comprehensive National Power Assessment System

This script provides a convenient way to run all tests in the CNPS-10 system,
with options for different test configurations and reporting.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --verbose    # Run with verbose output
    python run_tests.py --coverage   # Run with coverage reporting

Author: CNPS-10 Research Team
Version: 3.0.0
License: MIT
"""

import unittest
import sys
import os
import argparse
from pathlib import Path

# Add parent directory to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

def discover_and_run_tests(verbose=False, pattern='test_*.py'):
    """
    Discover and run all test cases in the tests directory.
    
    Args:
        verbose (bool): Whether to run tests in verbose mode
        pattern (str): Pattern to match test files
        
    Returns:
        bool: True if all tests passed, False otherwise
    """
    # Discover tests
    test_loader = unittest.TestLoader()
    test_dir = Path(__file__).parent
    test_suite = test_loader.discover(
        start_dir=str(test_dir),
        pattern=pattern,
        top_level_dir=str(parent_dir)
    )
    
    # Configure test runner
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(
        verbosity=verbosity,
        stream=sys.stdout,
        failfast=False,
        buffer=True
    )
    
    # Run tests
    print(f"🧪 Running CNPS-10 Test Suite")
    print(f"📁 Test directory: {test_dir}")
    print(f"🔍 Pattern: {pattern}")
    print("=" * 60)
    
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"📊 Test Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        
        if result.failures:
            print(f"\n💥 Failures ({len(result.failures)}):")
            for test, traceback in result.failures:
                print(f"   - {test}")
                
        if result.errors:
            print(f"\n🚨 Errors ({len(result.errors)}):")
            for test, traceback in result.errors:
                print(f"   - {test}")
        
        return False

def main():
    """Main function to handle command line arguments and run tests."""
    parser = argparse.ArgumentParser(
        description="CNPS-10 Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --verbose          # Run with verbose output
  python run_tests.py --pattern "test_data*"  # Run specific tests
        """
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Run tests in verbose mode'
    )
    
    parser.add_argument(
        '-p', '--pattern',
        default='test_*.py',
        help='Pattern to match test files (default: test_*.py)'
    )
    
    args = parser.parse_args()
    
    # Run tests
    success = discover_and_run_tests(verbose=args.verbose, pattern=args.pattern)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
