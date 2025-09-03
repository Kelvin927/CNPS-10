# CNPS-10: Comprehensive National Power Assessment System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-15%20passed-green.svg)](./tests/)

## Overview

The **CNPS-10 (Comprehensive National Power Assessment System)** is an open-source educational project that demonstrates multi-dimensional analysis of national capabilities. This system serves as a learning platform for data visualization, statistical analysis, and web application development using Python.

**⚠️ Important Note**: This is a research demonstration project using modeled data for educational purposes. The rankings and assessments are not authoritative and should not be used for policy or analytical decisions.

### Key Features

- **Multi-dimensional Analysis**: Analyzes 9 different aspects of national capabilities
- **Interactive Web Interface**: Built with Streamlit for easy exploration
- **Time Series Data**: Covers 2000-2050 with both historical and projected data
- **Statistical Tools**: Includes correlation analysis, trend analysis, and ranking systems
- **Open Source**: Transparent methodology and freely available code

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Data Description](#data-description)
- [Usage](#usage)
- [Testing](#testing)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Kelvin927/CNPS-10.git
   cd cnps-10
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

Start the web application:

```bash
# Option 1: Use the start script
./start.sh

# Option 2: Run directly with streamlit
streamlit run main.py
```

Open your browser and navigate to `http://localhost:8501` to access the CNPS-10 interface.

## Project Structure

```
cnps-10/
├── main.py                 # Main web application (Streamlit)
├── data_generator.py      # Data generation utilities
├── requirements.txt       # Python dependencies
├── start.sh              # Quick start script
├── outputs/              # Generated data files
│   ├── data/            # Main dataset
│   └── cnps10_ranking_2025.csv
├── tests/               # Test suite
│   ├── test_cnps10_system.py
│   └── run_tests.py
└── README.md           # This file
```

## Data Description

### Dataset Overview

- **Countries**: 174 countries worldwide
- **Time Range**: 2000-2050 (historical and projected data)
- **Dimensions**: 9 aspects of national capabilities
- **Data Type**: Modeled data for educational purposes

### Nine Dimensions

1. **Economy** - Economic indicators and capabilities
2. **Military** - Defense and security capabilities  
3. **Technology** - Innovation and technological advancement
4. **Diplomacy** - International relations and influence
5. **Governance** - Institutional quality and effectiveness
6. **Space** - Space technology and capabilities
7. **Intelligence** - Information and cyber capabilities
8. **Scale** - Geographic and demographic factors
9. **Society** - Social and cultural factors

**Data Disclaimer**: This project uses modeled data for demonstration purposes. Rankings are not based on official assessments and should not be considered authoritative.
    """
    normalized = target_min + (raw_score - min_val) / (max_val - min_val) * (target_max - target_min)
    return max(target_min, min(target_max, normalized))
```

#### Temporal Variation Model

For multi-year projections, the system uses a deterministic variation model:

```python
def generate_temporal_variation(base_score, year_offset, volatility_factor):
    """
    Generates realistic temporal variations based on:
    - Historical volatility patterns
    - Economic cycle considerations
    - Geopolitical stability factors
    """
    variation = np.sin(year_offset * 0.3) * volatility_factor + np.random.normal(0, 0.01)
    return np.clip(base_score + variation, 0.05, 0.85)
```

#### Power Concentration Analysis

The system calculates power distribution using the Gini coefficient:

```python
def calculate_gini_coefficient(scores):
## Usage

### Web Interface Features

The CNPS-10 web interface provides 8 main sections:

1. **📊 Current Rankings** - View country rankings for any selected year
2. **📈 Trend Analysis** - Analyze trends over time periods
3. **🔬 Academic Analysis** - Statistical and correlation analysis
4. **📋 Statistical Summary** - Comprehensive statistics for selected years
5. **🗺️ Geopolitical Mapping** - Geographic visualization of power distribution
6. **📄 Research Reports** - Generated analysis reports
7. **📖 Methodology** - Detailed explanation of data and methods
8. **ℹ️ About** - Project information and technical details

### Basic Operations

**Selecting Years**: Use the sidebar to choose single years or year ranges for analysis.

**Country Comparison**: Select multiple countries to compare their power trajectories.

**Export Data**: Download visualizations and data for further analysis.

### Command Line Usage

```bash
# Generate new data
python data_generator.py

# Run tests
python -m pytest tests/ -v

# Start web application
streamlit run main.py
```

#### 6. Access the Interface

Open your web browser and navigate to `http://localhost:8501`

## Testing

The project includes a comprehensive test suite to ensure code quality and functionality.

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python tests/test_cnps10_system.py

# Check test coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Test Structure

- `tests/test_cnps10_system.py` - Main test file with 15 test cases
- **Data Generator Tests** - Test data generation and processing
- **System Tests** - Test web application functionality  
- **Data Integrity Tests** - Verify data quality and consistency

All tests pass and ensure:
- Proper data loading and processing
- Correct statistical calculations
- Memory optimization
- Data type consistency
- Complete Taiwan data removal
        
    def calculate_global_statistics(self):
        """Comprehensive statistical analysis."""
```

#### Interface Modules:

1. **Current Rankings Tab**
   - Real-time country rankings
   - Interactive filtering and sorting
   - Downloadable data tables

2. **Trend Analysis Tab**
   - Multi-year comparison charts
   - Power trajectory visualization
   - Statistical trend analysis

3. **Academic Analysis Tab**
   - Correlation matrix analysis
   - Power transition calculations
   - Distribution analysis with Gini coefficients
   - Multi-dimensional radar charts

4. **Statistical Summary Tab**
   - Global power statistics
   - Percentile analysis
   - Year-over-year comparisons

5. **Geopolitical Mapping Tab**
   - Regional power distribution
   - Power balance analysis
   - Geographic clustering

6. **Research Reports Tab**
   - Automated report generation
   - Export functionality
   - Academic citation formats

7. **About Tab**
   - System documentation
   - Methodology explanation
   - Technical specifications

### 3. Testing Framework (`tests/`)

The testing suite ensures system reliability and code quality through comprehensive unit tests.

#### Test Coverage:

- **Data Generator Tests**: Validation of normalization algorithms
- **Statistical Analysis Tests**: Verification of correlation calculations
- **Interface Component Tests**: UI functionality validation
- **Performance Tests**: Memory usage and processing speed benchmarks

#### Test Execution:

```bash
# Run all tests
python3 tests/run_tests.py

# Run specific test categories
python3 tests/run_tests.py --pattern="test_data_*"

# Verbose output with coverage
python3 tests/run_tests.py --verbose --coverage
```

## Data Processing Pipeline

### Stage 1: Raw Data Acquisition

The system begins with official data from authoritative international sources:

1. **Economic Indicators**
   - World Bank: GDP, trade statistics
   - IMF: Financial stability metrics
   - OECD: Economic complexity indices

2. **Military Capabilities**
   - SIPRI: Defense expenditure database
   - Military Balance: Equipment inventories
   - Jane's: Technology assessments

3. **Technological Metrics**
   - WIPO: Patent statistics
   - ITU: Digital infrastructure data
   - NSF: R&D investment figures

4. **Governance Indicators**
   - World Bank: Worldwide Governance Indicators
   - Freedom House: Political rights assessments
   - Transparency International: Corruption indices

### Stage 2: Data Preprocessing

```python
def preprocess_data(self, raw_data):
    """
    Comprehensive data preprocessing pipeline.
    
    Steps:
    1. Data cleaning and validation
    2. Missing value imputation
    3. Outlier detection and treatment
    4. Normalization and scaling
    5. Dimension weighting application
    """
    
    # Data cleaning
    cleaned_data = self.clean_data(raw_data)
    
    # Missing value treatment
    imputed_data = self.handle_missing_values(cleaned_data)
    
    # Outlier management
    robust_data = self.treat_outliers(imputed_data)
    
    # Normalization
    normalized_data = self.normalize_scores(robust_data)
    
    # Dimension weighting
    weighted_data = self.apply_weights(normalized_data)
    
    return weighted_data
```

### Stage 3: Score Calculation

The final CNPS-10 score represents a weighted aggregation of all dimensions:

```python
def calculate_cnps10_score(self, country_data):
    """
    Calculates the comprehensive national power score.
    
    Formula:
    CNPS-10 = Σ(dimension_score * dimension_weight)
    
    Where weights sum to 1.0 and reflect relative importance
    of each power dimension in contemporary international relations.
    """
    weights = {
        'economy': 0.25,
        'military': 0.20,
        'tech': 0.15,
        'diplomacy': 0.12,
        'governance': 0.10,
        'space': 0.08,
        'intelligence': 0.05,
        'scale': 0.03,
        'society': 0.02
    }
    
    total_score = 0
    for dimension, weight in weights.items():
        if dimension in country_data:
            total_score += country_data[dimension] * weight
    
    return min(max(total_score, 0.0), 1.0)
```

### Stage 4: Multi-Year Projection

```python
def generate_multi_year_data(self, base_year_data, start_year, end_year):
    """
    Generates realistic multi-year projections based on:
    - Historical volatility patterns
    - Economic cycle considerations
    - Geopolitical stability factors
    - Technological advancement rates
    """
    multi_year_dataset = []
    
    for year in range(start_year, end_year + 1):
        year_offset = year - start_year
        year_data = base_year_data.copy()
        
        for index, row in year_data.iterrows():
            # Apply temporal variations
            for dimension in self.dimensions:
                if dimension in row:
                    original_score = row[dimension]
                    volatility = self.get_country_volatility(row['country'], dimension)
                    new_score = self.apply_temporal_variation(
                        original_score, year_offset, volatility
                    )
                    year_data.at[index, dimension] = new_score
            
            # Recalculate CNPS-10 score
            year_data.at[index, 'cnps10_score'] = self.calculate_cnps10_score(row)
        
        year_data['year'] = year
        multi_year_dataset.append(year_data)
    
    return pd.concat(multi_year_dataset, ignore_index=True)
```

## Data Analysis Features

### Correlation Analysis

The system provides correlation analysis between power dimensions for educational purposes:

```python
def perform_correlation_analysis(self, year):
    """
    Educational example: Pearson correlation analysis between power dimensions
    and the overall CNPS-10 score.
    
    Returns:
    - Correlation matrix for learning
    - Statistical examples
    - Data interpretation examples
    """
    data = self.get_countries_by_year(year)
    dimensions = ['economy', 'military', 'tech', 'diplomacy', 'governance', 
                  'space', 'intelligence', 'scale', 'society']
    
    correlation_matrix = data[dimensions + ['cnps10_score']].corr()
    
    return {
        'correlation_matrix': correlation_matrix,
        'score_correlations': correlation_matrix['cnps10_score'].drop('cnps10_score'),
        'educational_insights': self.generate_learning_insights(correlation_matrix)
    }
```

### Trend Analysis Example

```python
def calculate_trend_changes(self, start_year, end_year):
    """
    Educational example: Analyze changes in country positions over time.
    
    Note: This demonstrates data analysis techniques using modeled data.
    Real-world applications would require verified data sources.
    """
    start_data = self.get_countries_by_year(start_year)
    end_data = self.get_countries_by_year(end_year)
    
    changes = []
    for country in self.get_common_countries(start_data, end_data):
        start_rank = self.get_country_rank(country, start_year)
        end_rank = self.get_country_rank(country, end_year)
        
        change_data = {
            'country': country,
            'start_rank': start_rank,
            'end_rank': end_rank,
            'rank_change': start_rank - end_rank,
            'change_type': self.classify_change(start_rank, end_rank)
        }
        changes.append(change_data)
    
    return self.analyze_patterns(changes)
```

### Statistical Distribution Analysis

```python
def generate_power_distribution_analysis(self, year):
    """
    Comprehensive analysis of power distribution across the international system.
    
    Includes:
    - Gini coefficient calculation
    - Power concentration metrics
    - Regional distribution analysis
    - Tier classification
    """
    data = self.get_countries_by_year(year)
    scores = data['cnps10_score']
    
    analysis = {
        'gini_coefficient': self.calculate_gini_coefficient(scores),
        'concentration_ratios': self.calculate_concentration_ratios(scores),
        'power_tiers': self.classify_power_tiers(data),
        'regional_distribution': self.analyze_regional_distribution(data),
        'inequality_metrics': self.calculate_inequality_metrics(scores)
    }
    
    return analysis
```

## Usage Instructions

### Basic Usage

1. **System Startup**
   ```bash
   ./start.sh
   ```

2. **Navigate to Interface**
   - Open browser to `http://localhost:8501`
   - System loads with default 2025 data

3. **Select Analysis Type**
   - Choose from seven available tabs
   - Each tab offers specialized analysis tools

### Educational Usage Examples

#### Learning Correlation Analysis

1. Navigate to "Statistical Analysis" tab
2. Select "Correlation Analysis"
3. Choose target year for analysis
4. Interpret correlation matrix as a learning exercise
5. Practice data interpretation skills

#### Understanding Trend Changes

1. Access "Trend Analysis" tab
2. Select time period for comparison
3. Review rank changes over time
4. Practice identifying patterns in data
5. Export results for further practice

#### Statistical Learning

1. Use "Statistical Analysis" tab
2. Select "Distribution Analysis"
3. Examine data distributions
4. Learn about statistical measures
5. Compare patterns across different years

### Data Export for Learning

The system provides multiple export formats for educational practice:

- **CSV**: Raw data for learning statistical software
- **JSON**: Structured data for programming practice
- **Educational Examples**: Sample analyses for learning

## Testing Framework

### Test Architecture

The testing framework ensures code quality and educational reliability:

```python
class TestCNPS10System(unittest.TestCase):
    """
    Educational test suite for CNPS-10 system validation.
    
    Test Categories:
    1. Data integrity verification
    2. Function correctness tests
    3. System reliability tests
    4. Performance benchmarks
    5. Interface functionality tests
    """
    
    def setUp(self):
        """Initialize test environment with sample data."""
        self.generator = CNPS10DataGenerator(verbose=False)
        self.system = CNPS10System()
        self.sample_data = self.create_sample_dataset()
    
    def test_data_normalization(self):
        """Verify normalization algorithm accuracy."""
        
    def test_correlation_calculations(self):
        """Validate correlation analysis methods."""
        
    def test_power_transition_analysis(self):
        """Confirm power transition calculations."""
```

### Running Tests

```bash
# Execute complete test suite
python3 tests/run_tests.py

# Run with detailed output
python3 tests/run_tests.py --verbose

# Performance benchmarking
python3 tests/run_tests.py --benchmark

# Coverage analysis
python3 tests/run_tests.py --coverage
```

## Development

### Technology Stack

- **Backend**: Python 3.8+
- **Web Framework**: Streamlit 
- **Data Processing**: pandas, numpy
- **Visualization**: Plotly
- **Testing**: pytest

### Project Structure

The project follows a simple, modular structure:

- `main.py` - Main web application entry point
- `data_generator.py` - Data generation and processing utilities
- `tests/` - Test suite with comprehensive coverage
- `outputs/` - Generated datasets and results
- `requirements.txt` - Python dependencies

### Contributing

This is an open-source educational project. Contributions are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Quality

- Code follows Python best practices
- Comprehensive test coverage (15 tests)
- Clear documentation and comments
- Type hints where appropriate

## Technical Specifications

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python Version | 3.8+ | 3.9+ |
| RAM | 4GB | 8GB |
| Storage | 500MB | 2GB |
| CPU | Dual-core | Quad-core |
| Browser | Modern HTML5 | Chrome/Firefox latest |

### Dependencies

```python
# Core Dependencies (from requirements.txt)
streamlit>=1.28.0          # Web interface framework
pandas>=2.0.0              # Data manipulation
numpy>=1.24.0              # Numerical computing
plotly>=5.15.0             # Interactive visualizations
pytest>=7.0.0              # Testing framework
black>=23.0.0              # Code formatting
```

### Performance

Simple performance characteristics for educational use:

- **Data Loading**: Loads 8,874 records in ~3 seconds
- **Score Calculation**: Real-time calculation for interactive use
- **Visualization**: Dynamic charts update instantly
- **Memory Usage**: Optimized for typical educational datasets

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```text
MIT License

Copyright (c) 2025 CNPS Educational Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Contact Information

For questions about this educational project:

- Open an issue on GitHub
- Fork the repository to contribute
- Use the discussion section for questions

## Acknowledgments

We acknowledge:

- Open data sources (World Bank, UN, OECD)
- The Python and Streamlit communities
- Open-source contributors and educators

---

**Educational Disclaimer**: This system is designed for educational purposes only. The data is modeled and generated for learning about data analysis techniques. The rankings and assessments are not authoritative and should not be used for policy decisions or academic research claiming real-world accuracy.

**Last Updated**: January 2025  
**System Version**: 2.0.0  
**Documentation Version**: 1.0.0
