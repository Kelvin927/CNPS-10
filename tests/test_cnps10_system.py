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
Version: 1.0.0
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

from data_generator import RealisticCNPS10Generator
from main import CNPS10System


class TestRealisticCNPS10Generator(unittest.TestCase):
    """Test cases for the RealisticCNPS10Generator class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.generator = RealisticCNPS10Generator(verbose=False)
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after each test method."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test proper initialization of data generator."""
        self.assertIsInstance(self.generator, RealisticCNPS10Generator)
        self.assertIsNotNone(self.generator.country_tiers)
        self.assertIn('superpowers', self.generator.country_tiers)
        self.assertIn('great_powers', self.generator.country_tiers)
    
    def test_generate_realistic_data(self):
        """Test realistic data generation."""
        result = self.generator.generate_realistic_data(base_year=2025)
        self.assertTrue(result)
        self.assertIsNotNone(self.generator.countries_data)
        
        # Check data structure
        required_columns = ['country', 'cnps10_score', 'rank', 'tier']
        for col in required_columns:
            self.assertIn(col, self.generator.countries_data.columns)
        
        # Check score range
        scores = self.generator.countries_data['cnps10_score']
        self.assertTrue(all(scores >= 0))
        self.assertTrue(all(scores <= 1))
        
        # Check that USA and China are at the top (realistic)
        top_countries = self.generator.countries_data.nsmallest(2, 'rank')['country'].tolist()
        self.assertIn('United States', top_countries)
        self.assertIn('China', top_countries)
    
    def test_generate_multiyear_data(self):
        """Test multi-year data generation."""
        # First generate base data
        self.generator.generate_realistic_data(base_year=2025)
        
        # Then generate multi-year data
        result = self.generator.generate_multiyear_data(start_year=2020, end_year=2025)
        self.assertTrue(result)
        self.assertIsNotNone(self.generator.multiyear_data)
        
        # Check year range
        years = self.generator.multiyear_data['year'].unique()
        self.assertEqual(len(years), 6)  # 2020-2025 inclusive
        self.assertEqual(min(years), 2020)
        self.assertEqual(max(years), 2025)
        
        # Check data consistency
        countries_per_year = self.generator.multiyear_data.groupby('year')['country'].nunique()
        self.assertTrue(all(countries_per_year == countries_per_year.iloc[0]))
    
    def test_save_data(self):
        """Test data saving functionality."""
        # Generate data first
        self.generator.generate_realistic_data(base_year=2025)
        self.generator.generate_multiyear_data(start_year=2020, end_year=2025)
        
        # Save to test directory
        result = self.generator.save_data(output_dir=self.test_dir)
        self.assertTrue(result)
        
        # Check if files were created
        files_created = os.listdir(self.test_dir)
        self.assertTrue(len(files_created) > 0)
        
        # Check if at least one CSV file was created
        csv_files = [f for f in files_created if f.endswith('.csv')]
        self.assertTrue(len(csv_files) > 0)


class TestCNPS10System(unittest.TestCase):
    """Test cases for the CNPS10System class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary data file for testing
        self.test_dir = tempfile.mkdtemp()
        self.test_data_dir = os.path.join(self.test_dir, 'outputs', 'data')
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Create sample test data
        sample_data = []
        countries = ['United States', 'China', 'Germany', 'Japan', 'United Kingdom']
        for year in range(2020, 2026):
            for i, country in enumerate(countries):
                sample_data.append({
                    'country': country,
                    'year': year,
                    'cnps10_score': 0.95 - (i * 0.05) + np.random.normal(0, 0.01),
                    'rank': i + 1,
                    'tier': 'superpower' if i < 2 else 'great_power',
                    'economy': 0.9 - (i * 0.1),
                    'military': 0.85 - (i * 0.08),
                    'tech': 0.8 - (i * 0.06)
                })
        
        self.test_data = pd.DataFrame(sample_data)
        self.test_file = os.path.join(self.test_data_dir, 'cnps10_official_based_172countries_20250903_022541.csv')
        self.test_data.to_csv(self.test_file, index=False)
        
        # Change to test directory
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)
    
    def tearDown(self):
        """Clean up after each test method."""
        os.chdir(self.original_dir)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_system_initialization(self):
        """Test CNPS10System initialization."""
        try:
            system = CNPS10System()
            self.assertIsNotNone(system)
            # System should load data during initialization
            self.assertIsNotNone(system.multi_year_data)
        except Exception as e:
            self.fail(f"System initialization failed: {str(e)}")
    
    def test_get_available_years(self):
        """Test getting available years from dataset."""
        system = CNPS10System()
        years = system.get_available_years()
        self.assertIsInstance(years, list)
        self.assertTrue(len(years) > 0)
        self.assertIn(2020, years)
        self.assertIn(2025, years)
    
    def test_get_countries_by_year(self):
        """Test filtering countries by year."""
        system = CNPS10System()
        countries_2025 = system.get_countries_by_year(2025)
        
        self.assertIsInstance(countries_2025, pd.DataFrame)
        self.assertTrue(len(countries_2025) > 0)
        self.assertTrue(all(countries_2025['year'] == 2025))
    
    def test_get_top_countries(self):
        """Test getting top countries."""
        system = CNPS10System()
        top_countries = system.get_top_countries(2025, top_n=3)
        
        self.assertIsInstance(top_countries, pd.DataFrame)
        self.assertEqual(len(top_countries), 3)
        
        # Check that scores are in descending order
        scores = top_countries['cnps10_score'].tolist()
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_generate_statistics_summary(self):
        """Test statistics summary generation."""
        system = CNPS10System()
        stats = system.generate_statistics_summary(2025)
        
        self.assertIsInstance(stats, dict)
        expected_keys = ['total_countries', 'mean_score', 'median_score', 'std_score']
        for key in expected_keys:
            self.assertIn(key, stats)
            self.assertIsNotNone(stats[key])
    
    def test_calculate_power_transitions(self):
        """Test power transition calculations."""
        system = CNPS10System()
        transitions = system.calculate_power_transitions(2020, 2025)
        
        self.assertIsInstance(transitions, dict)
        self.assertIn('transitions', transitions)
        self.assertIn('period', transitions)
        self.assertEqual(transitions['period'], '2020-2025')
    
    def test_perform_correlation_analysis(self):
        """Test correlation analysis."""
        system = CNPS10System()
        correlations = system.perform_correlation_analysis(2025)
        
        if correlations:  # Only test if correlations were calculated
            self.assertIsInstance(correlations, dict)
            self.assertIn('score_correlations', correlations)
    
    def test_data_types_optimization(self):
        """Test that data types are properly optimized."""
        system = CNPS10System()
        
        if system.multi_year_data is not None:
            # Check that year is int16
            if 'year' in system.multi_year_data.columns:
                self.assertEqual(system.multi_year_data['year'].dtype, 'int16')
            
            # Check that score columns are float64 (high precision)
            score_cols = [col for col in system.multi_year_data.columns if 'score' in col.lower()]
            for col in score_cols:
                self.assertEqual(system.multi_year_data[col].dtype, 'float64')


class TestDataIntegrity(unittest.TestCase):
    """Test cases for data integrity and consistency."""
    
    def test_main_data_file_exists(self):
        """Test that the main data file exists and is accessible."""
        main_file = 'outputs/data/cnps10_official_based_172countries_20250903_022541.csv'
        self.assertTrue(os.path.exists(main_file), f"Main data file not found: {main_file}")
    
    def test_main_data_file_structure(self):
        """Test the structure of the main data file."""
        main_file = 'outputs/data/cnps10_official_based_172countries_20250903_022541.csv'
        
        if os.path.exists(main_file):
            data = pd.read_csv(main_file)
            
            # Check required columns
            required_columns = ['country', 'year', 'cnps10_score', 'rank']
            for col in required_columns:
                self.assertIn(col, data.columns, f"Required column '{col}' not found")
            
            # Check data ranges
            self.assertTrue(all(data['cnps10_score'] >= 0), "CNPS scores should be non-negative")
            self.assertTrue(all(data['cnps10_score'] <= 1), "CNPS scores should not exceed 1")
            self.assertTrue(all(data['rank'] >= 1), "Ranks should start from 1")
            
            # Check for reasonable number of countries and years
            countries = data['country'].nunique()
            years = data['year'].nunique()
            self.assertGreater(countries, 50, "Should have reasonable number of countries")
            self.assertGreater(years, 5, "Should have multiple years of data")
    
    def test_no_taiwan_data(self):
        """Test that Taiwan data has been completely removed."""
        main_file = 'outputs/data/cnps10_official_based_172countries_20250903_022541.csv'
        
        if os.path.exists(main_file):
            data = pd.read_csv(main_file)
            
            taiwan_entities = ['Taiwan', 'Taiwan, China', 'Chinese Taipei', 'Hong Kong SAR, China', 
                             'Macao SAR, China', 'Hong Kong', 'Macau']
            
            for entity in taiwan_entities:
                self.assertNotIn(entity, data['country'].values, 
                               f"Taiwan-related entity '{entity}' found in data")


def run_all_tests():
    """Run all test suites."""
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestRealisticCNPS10Generator)
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestCNPS10System))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestDataIntegrity))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("🧪 CNPS-10 System Test Suite")
    print("=" * 50)
    
    success = run_all_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
