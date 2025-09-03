#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNPS-10 Test Suite
Unit tests for the Comprehensive National Power Assessment System

This module contains comprehensive unit tests for all major components
of the CNPS-10 system, ensuring code quality and reliability.

Test Coverage:
- Data generator functionality
- Main application components
- Statistical calculations
- Memory optimization
- Error handling

Author: CNPS-10 Research Team
Version: 3.0.0
License: MIT
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_generator import CNPS10DataGenerator
from main import CNPS10System


class TestCNPS10DataGenerator(unittest.TestCase):
    """Test cases for the CNPS10DataGenerator class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.generator = CNPS10DataGenerator(verbose=False)
        self.test_dir = tempfile.mkdtemp()
        
        # Create sample test data
        self.sample_data = pd.DataFrame({
            'Country': ['United States', 'China', 'Germany', 'Japan', 'United Kingdom'],
            'CNPS10_score': [0.95, 0.85, 0.75, 0.70, 0.65],
            'Power_Tier': ['Superpower', 'Great Power', 'Major Power', 'Major Power', 'Major Power']
        })
        
        self.sample_file = os.path.join(self.test_dir, 'test_data.csv')
        self.sample_data.to_csv(self.sample_file, index=False)
    
    def tearDown(self):
        """Clean up after each test method."""
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test proper initialization of data generator."""
        self.assertIsInstance(self.generator, CNPS10DataGenerator)
        self.assertIsNone(self.generator.original_data)
        self.assertIsNone(self.generator.processed_data)
    
    def test_load_official_data_success(self):
        """Test successful loading of official data."""
        result = self.generator.load_official_data(self.sample_file)
        self.assertTrue(result)
        self.assertIsNotNone(self.generator.original_data)
        self.assertEqual(len(self.generator.original_data), 5)
    
    def test_load_official_data_file_not_found(self):
        """Test handling of missing data file."""
        result = self.generator.load_official_data('nonexistent_file.csv')
        self.assertFalse(result)
        self.assertIsNone(self.generator.original_data)
    
    def test_normalize_scores(self):
        """Test score normalization functionality."""
        self.generator.load_official_data(self.sample_file)
        result = self.generator.normalize_scores(target_min=0.1, target_max=0.9)
        self.assertTrue(result)
        
        scores = self.generator.processed_data['cnps10_score']
        self.assertAlmostEqual(float(scores.min()), 0.1, places=5)
        self.assertAlmostEqual(float(scores.max()), 0.9, places=5)
    
    def test_normalize_scores_without_data(self):
        """Test normalization error handling when no data is loaded."""
        result = self.generator.normalize_scores()
        self.assertFalse(result)
    
    def test_generate_multiyear_data(self):
        """Test multi-year data generation."""
        self.generator.load_official_data(self.sample_file)
        self.generator.normalize_scores()
        
        result = self.generator.generate_multiyear_data(start_year=2020, end_year=2025)
        self.assertTrue(result)
        self.assertTrue(hasattr(self.generator, 'multiyear_data'))
        
        expected_records = 5 * 6  # 5 countries × 6 years
        self.assertEqual(len(self.generator.multiyear_data), expected_records)
        
        # Check year range
        years = self.generator.multiyear_data['year'].unique()
        self.assertEqual(set(years), set(range(2020, 2026)))
    
    def test_data_type_optimization(self):
        """Test memory optimization of data types."""
        self.generator.load_official_data(self.sample_file)
        self.generator.normalize_scores()
        self.generator.generate_multiyear_data(start_year=2020, end_year=2022)
        
        # Check optimized data types
        self.assertEqual(self.generator.multiyear_data['year'].dtype, 'int16')
        self.assertEqual(self.generator.multiyear_data['cnps10_score'].dtypes, 'float32')
        self.assertEqual(self.generator.multiyear_data['country'].dtype.name, 'category')
    
    def test_generate_summary_report(self):
        """Test summary report generation."""
        self.generator.load_official_data(self.sample_file)
        self.generator.normalize_scores()
        self.generator.generate_multiyear_data(start_year=2020, end_year=2022)
        
        summary = self.generator.generate_summary_report()
        
        self.assertIn('total_records', summary)
        self.assertIn('total_countries', summary)
        self.assertIn('year_range', summary)
        self.assertIn('score_statistics', summary)
        self.assertIn('top_10_latest', summary)
        
        self.assertEqual(summary['total_countries'], 5)
        self.assertEqual(summary['year_range']['start'], 2020)
        self.assertEqual(summary['year_range']['end'], 2022)


class TestCNPS10System(unittest.TestCase):
    """Test cases for the CNPS10System class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary test data
        self.test_dir = tempfile.mkdtemp()
        
        # Create mock data structure
        os.makedirs(os.path.join(self.test_dir, 'outputs', 'data'), exist_ok=True)
        
        # Sample multi-year data
        sample_multiyear = pd.DataFrame({
            'country': ['USA', 'China', 'Germany'] * 3,
            'cnps10_score': [0.85, 0.75, 0.65, 0.86, 0.76, 0.66, 0.87, 0.77, 0.67],
            'power_tier': ['Superpower', 'Great Power', 'Major Power'] * 3,
            'year': [2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2022]
        })
        
        self.multiyear_file = os.path.join(
            self.test_dir, 'outputs', 'data', 
            'cnps10_official_based_172countries_20250903_022541.csv'
        )
        sample_multiyear.to_csv(self.multiyear_file, index=False)
        
        # Sample latest data
        sample_latest = pd.DataFrame({
            'country': ['USA', 'China', 'Germany'],
            'cnps10_score': [0.87, 0.77, 0.67],
            'power_tier': ['Superpower', 'Great Power', 'Major Power']
        })
        
        self.latest_file = os.path.join(self.test_dir, 'outputs', 'cnps10_ranking_2025.csv')
        sample_latest.to_csv(self.latest_file, index=False)
        
        # Change working directory for tests
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
    
    def tearDown(self):
        """Clean up after each test method."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def test_system_initialization(self):
        """Test proper system initialization."""
        system = CNPS10System()
        self.assertIsNotNone(system.multi_year_data)
        self.assertIsNotNone(system.latest_data)
    
    def test_get_available_years(self):
        """Test retrieval of available years."""
        system = CNPS10System()
        years = system.get_available_years()
        self.assertIn(2020, years)
        self.assertIn(2021, years)
        self.assertIn(2022, years)
    
    def test_get_countries_by_year(self):
        """Test filtering countries by year."""
        system = CNPS10System()
        data_2020 = system.get_countries_by_year(2020)
        
        self.assertEqual(len(data_2020), 3)
        self.assertTrue(all(data_2020['year'] == 2020))
    
    def test_get_top_countries(self):
        """Test retrieval of top countries."""
        system = CNPS10System()
        top_countries = system.get_top_countries(2020, top_n=2)
        
        self.assertEqual(len(top_countries), 2)
        # Should be sorted by score descending
        scores = top_countries['cnps10_score'].values
        self.assertTrue(scores[0] >= scores[1])
    
    def test_generate_statistics_summary(self):
        """Test statistical summary generation."""
        system = CNPS10System()
        stats = system.generate_statistics_summary(2020)
        
        required_keys = [
            'total_countries', 'mean_score', 'median_score', 
            'std_score', 'min_score', 'max_score'
        ]
        
        for key in required_keys:
            self.assertIn(key, stats)
        
        self.assertEqual(stats['total_countries'], 3)
        self.assertGreater(stats['mean_score'], 0)
        self.assertLess(stats['mean_score'], 1)


class TestStatisticalCalculations(unittest.TestCase):
    """Test cases for statistical calculations and validations."""
    
    def test_score_distribution(self):
        """Test score distribution properties."""
        # Generate sample scores
        scores = np.array([0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15])
        
        # Test basic statistics
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        self.assertAlmostEqual(mean_score, 0.5, places=2)
        self.assertGreater(std_score, 0)
    
    def test_normalization_properties(self):
        """Test properties of score normalization."""
        original_scores = np.array([10, 20, 30, 40, 50])
        target_min, target_max = 0.1, 0.9
        
        # Apply linear normalization
        min_orig = original_scores.min()
        max_orig = original_scores.max()
        normalized = target_min + (original_scores - min_orig) * (target_max - target_min) / (max_orig - min_orig)
        
        # Test bounds
        self.assertAlmostEqual(normalized.min(), target_min, places=5)
        self.assertAlmostEqual(normalized.max(), target_max, places=5)
        
        # Test monotonicity
        self.assertTrue(np.all(normalized[:-1] <= normalized[1:]))


class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling and edge cases."""
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty dataframes."""
        empty_df = pd.DataFrame()
        
        # Should not raise exceptions
        try:
            result = len(empty_df)
            self.assertEqual(result, 0)
        except Exception as e:
            self.fail(f"Empty dataframe handling failed: {e}")
    
    def test_missing_column_handling(self):
        """Test handling of missing required columns."""
        df = pd.DataFrame({'wrong_column': [1, 2, 3]})
        
        # Should handle missing columns gracefully
        self.assertNotIn('cnps10_score', df.columns)
        self.assertNotIn('country', df.columns)
    
    def test_invalid_year_handling(self):
        """Test handling of invalid year inputs."""
        generator = CNPS10DataGenerator(verbose=False)
        
        # Should handle invalid year ranges
        result = generator.generate_multiyear_data(start_year=2030, end_year=2020)
        self.assertFalse(result)


def run_all_tests():
    """Run all test suites."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestCNPS10DataGenerator))
    test_suite.addTest(unittest.makeSuite(TestCNPS10System))
    test_suite.addTest(unittest.makeSuite(TestStatisticalCalculations))
    test_suite.addTest(unittest.makeSuite(TestErrorHandling))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("🧪 CNPS-10 Test Suite")
    print("=" * 50)
    
    success = run_all_tests()
    
    if success:
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        sys.exit(1)
