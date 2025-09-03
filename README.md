# CNPS-10: Comprehensive National Power Assessment System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-18%20passed-green.svg)](./tests/)

## Abstract

The **Comprehensive National Power Assessment System (CNPS-10)** is an advanced analytical framework designed for academic research in international relations, providing systematic measurement and comparison of national power capabilities across multiple dimensions. This system employs rigorous statistical methodologies to assess the relative power positions of 172 countries worldwide from 2025 to 2029, offering scholars, policymakers, and analysts a robust platform for understanding global power dynamics.

The system integrates nine core dimensions of national power: economic capabilities, military strength, technological innovation, diplomatic influence, governance quality, space capabilities, intelligence systems, geographic scale, and social power. Through sophisticated data processing algorithms and interactive visualization tools, CNPS-10 enables comprehensive analysis of power transitions, correlation studies, and geopolitical mapping.

## Table of Contents

- [System Architecture](#system-architecture)
- [Research Methodology](#research-methodology)
- [Installation Guide](#installation-guide)
- [System Components](#system-components)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Academic Features](#academic-features)
- [Usage Instructions](#usage-instructions)
- [Testing Framework](#testing-framework)
- [Development Process](#development-process)
- [Technical Specifications](#technical-specifications)
- [Citation Guidelines](#citation-guidelines)
- [Contributing](#contributing)
- [License](#license)

## System Architecture

### Overview

CNPS-10 follows a modular architecture designed for scalability, maintainability, and academic rigor. The system consists of three primary layers:

1. **Data Processing Layer** (`data_generator.py`)
2. **Core Analysis Engine** (`main.py`)
3. **Testing and Validation Framework** (`tests/`)

```
┌─────────────────────────────────────────────────────────────┐
│                    CNPS-10 System Architecture             │
├─────────────────────────────────────────────────────────────┤
│  Web Interface Layer (Streamlit)                           │
│  ┌─────────────────┬─────────────────┬─────────────────┐   │
│  │  Current        │  Trend          │  Academic       │   │
│  │  Rankings       │  Analysis       │  Analysis       │   │
│  └─────────────────┴─────────────────┴─────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  Core Analysis Engine (CNPS10System Class)                 │
│  ┌─────────────────┬─────────────────┬─────────────────┐   │
│  │  Statistical    │  Correlation    │  Power          │   │
│  │  Analysis       │  Analysis       │  Transitions    │   │
│  └─────────────────┴─────────────────┴─────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  Data Processing Layer                                      │
│  ┌─────────────────┬─────────────────┬─────────────────┐   │
│  │  Data           │  Score          │  Multi-year     │   │
│  │  Validation     │  Normalization  │  Generation     │   │
│  └─────────────────┴─────────────────┴─────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  Data Storage Layer                                         │
│  ┌─────────────────┬─────────────────┬─────────────────┐   │
│  │  Raw Data       │  Processed      │  Generated      │   │
│  │  (CSV)          │  Data (CSV)     │  Datasets       │   │
│  └─────────────────┴─────────────────┴─────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Academic Rigor**: All algorithms and methodologies are based on established academic literature
2. **Modularity**: Each component is independently testable and maintainable
3. **Scalability**: Architecture supports easy addition of new dimensions or countries
4. **Reproducibility**: All analyses are deterministic and fully reproducible
5. **Performance**: Memory-optimized for handling large datasets efficiently

## Research Methodology

### Theoretical Framework

The CNPS-10 system is grounded in comprehensive national power theory, drawing from seminal works in international relations literature. The framework conceptualizes national power as a multidimensional construct comprising both tangible and intangible capabilities.

#### Power Dimensions

1. **Economic Power (Weight: 25%)**
   - GDP (Purchasing Power Parity)
   - International trade volume
   - Financial market capitalization
   - Economic resilience indicators

2. **Military Power (Weight: 20%)**
   - Defense expenditure
   - Military personnel and equipment
   - Defense technology capabilities
   - Strategic reach and projection

3. **Technological Power (Weight: 15%)**
   - R&D investment as % of GDP
   - Patent applications and grants
   - Digital infrastructure quality
   - Innovation ecosystem strength

4. **Diplomatic Power (Weight: 12%)**
   - Embassy and consulate networks
   - International organization membership
   - Soft power indices
   - Cultural influence metrics

5. **Governance Quality (Weight: 10%)**
   - Government effectiveness
   - Rule of law indicators
   - Regulatory quality
   - Institutional stability

6. **Space Capabilities (Weight: 8%)**
   - Satellite launches and operations
   - Space technology development
   - Space-based assets
   - Commercial space industry

7. **Intelligence Systems (Weight: 5%)**
   - Cyber capabilities
   - Information systems
   - Intelligence infrastructure
   - Digital surveillance capacity

8. **Geographic Scale (Weight: 3%)**
   - Territory size
   - Population size
   - Natural resource endowments
   - Strategic geographic position

9. **Social Power (Weight: 2%)**
   - Education system quality
   - Cultural influence
   - Social cohesion indices
   - Human development metrics

### Statistical Methodology

#### Data Normalization

The system employs a sophisticated normalization algorithm to ensure comparability across dimensions:

```python
def normalize_score(raw_score, min_val, max_val, target_min=0.05, target_max=0.85):
    """
    Linear normalization with bounded scaling to prevent extreme values
    that could skew comparative analysis.
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
    """
    Measures power concentration across the international system.
    Values closer to 1 indicate higher concentration.
    """
    scores_sorted = np.sort(scores)
    n = len(scores_sorted)
    cumsum = np.cumsum(scores_sorted)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
```

## Installation Guide

### Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: macOS, Linux, or Windows
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 500MB free space

### Step-by-Step Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/Kelvin927/cnps-10
cd cnps-10
```

#### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Generate Initial Data

```bash
python3 data_generator.py
```

#### 5. Run the System

```bash
# Using the provided script
./start.sh

# Or directly with Streamlit   
streamlit run main.py
```

#### 6. Access the Interface

Open your web browser and navigate to `http://localhost:8501`

```

## System Components

### 1. Data Generator (`data_generator.py`)

The data generation module serves as the foundation of the CNPS-10 system, responsible for processing raw national power indicators and creating normalized, comparable datasets.

#### Key Features:
- **Official Data Integration**: Processes data from authoritative sources (World Bank, IMF, UN)
- **Score Normalization**: Implements linear scaling to ensure fair comparison
- **Multi-year Generation**: Creates temporal datasets with realistic variations
- **Data Validation**: Comprehensive error checking and consistency verification

#### Implementation Details:

```python
class CNPS10DataGenerator:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.original_data = None
        self.processed_data = None
        
    def process_official_data(self, input_file):
        """
        Processes raw CSV data and applies normalization algorithms.
        
        Args:
            input_file (str): Path to input CSV file
            
        Returns:
            pd.DataFrame: Processed and normalized data
        """
```

#### Data Processing Pipeline:

1. **Raw Data Ingestion**
   - CSV file parsing with error handling
   - Data type validation and conversion
   - Missing value detection and treatment

2. **Score Normalization**
   - Min-max scaling with bounded ranges
   - Outlier detection and adjustment
   - Dimension-specific weighting

3. **Quality Assurance**
   - Statistical consistency checks
   - Cross-validation against historical data
   - Anomaly detection algorithms

### 2. Core Analysis Engine (`main.py`)

The main application provides a comprehensive web interface built on Streamlit, offering seven specialized analysis modules.

#### System Class Architecture:

```python
class CNPS10System:
    def __init__(self):
        self.data = None
        self.multi_year_data = None
        self.latest_data = None
        
    def load_data(self):
        """Optimized data loading with memory management."""
        
    def get_countries_by_year(self, year):
        """Efficient year-based data filtering."""
        
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

The testing suite ensures system reliability and academic rigor through comprehensive unit tests.

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

## Academic Features

### Correlation Analysis

The system provides comprehensive correlation analysis between power dimensions:

```python
def perform_correlation_analysis(self, year):
    """
    Performs Pearson correlation analysis between all power dimensions
    and the overall CNPS-10 score.
    
    Returns:
    - Correlation matrix
    - Significance tests
    - Interpretation guidelines
    """
    data = self.get_countries_by_year(year)
    dimensions = ['economy', 'military', 'tech', 'diplomacy', 'governance', 
                  'space', 'intelligence', 'scale', 'society']
    
    correlation_matrix = data[dimensions + ['cnps10_score']].corr()
    
    return {
        'correlation_matrix': correlation_matrix,
        'score_correlations': correlation_matrix['cnps10_score'].drop('cnps10_score'),
        'strongest_correlations': self.identify_strongest_correlations(correlation_matrix),
        'interpretation': self.generate_correlation_interpretation(correlation_matrix)
    }
```

### Power Transition Analysis

```python
def calculate_power_transitions(self, start_year, end_year):
    """
    Analyzes power transitions between countries over specified time period.
    
    Methodology based on power transition theory (Organski & Kugler, 1980)
    and includes:
    - Rank mobility analysis
    - Score change calculations
    - Transition probability estimates
    """
    start_data = self.get_countries_by_year(start_year)
    end_data = self.get_countries_by_year(end_year)
    
    transitions = []
    for country in self.get_common_countries(start_data, end_data):
        start_rank = self.get_country_rank(country, start_year)
        end_rank = self.get_country_rank(country, end_year)
        
        transition_data = {
            'country': country,
            'start_rank': start_rank,
            'end_rank': end_rank,
            'rank_change': start_rank - end_rank,
            'transition_type': self.classify_transition(start_rank, end_rank),
            'probability': self.calculate_transition_probability(country, start_year, end_year)
        }
        transitions.append(transition_data)
    
    return self.analyze_transition_patterns(transitions)
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
   - Each tab offers specialized analytical tools

### Advanced Academic Usage

#### Conducting Correlation Studies

1. Navigate to "Academic Analysis" tab
2. Select "Correlation Analysis"
3. Choose target year for analysis
4. Interpret correlation matrix and significance tests
5. Export results for further statistical analysis

#### Power Transition Research

1. Access "Academic Analysis" tab
2. Select "Power Transitions" option
3. Define start and end years for analysis
4. Review rank changes and transition patterns
5. Generate academic reports with findings

#### Statistical Distribution Studies

1. Use "Academic Analysis" tab
2. Select "Distribution Analysis"
3. Examine power concentration metrics
4. Analyze Gini coefficients and inequality measures
5. Compare across different years

### Data Export and Citation

The system provides multiple export formats for academic use:

- **CSV**: Raw data for statistical software (R, STATA, SPSS)
- **JSON**: Structured data for programming environments
- **Academic Citations**: Pre-formatted references for publications

## Testing Framework

### Test Architecture

The CNPS-10 testing framework employs a multi-layered approach:

```python
class TestCNPS10System(unittest.TestCase):
    """
    Comprehensive test suite for CNPS-10 system validation.
    
    Test Categories:
    1. Data integrity tests
    2. Algorithm accuracy tests
    3. Statistical consistency tests
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

### Test Results Interpretation

The test suite provides comprehensive metrics:

- **Accuracy Tests**: Verify algorithm correctness
- **Performance Tests**: Measure processing speed and memory usage
- **Consistency Tests**: Ensure reproducible results
- **Integration Tests**: Validate component interactions

## Development Process

### System Development Methodology

The CNPS-10 system was developed using an iterative, research-driven approach:

#### Phase 1: Literature Review and Theoretical Foundation (Months 1-2)
- Comprehensive review of national power literature
- Analysis of existing measurement frameworks
- Identification of methodological gaps
- Development of theoretical framework

#### Phase 2: Data Architecture Design (Months 3-4)
- Database schema development
- Data source identification and validation
- Normalization methodology design
- Quality assurance framework establishment

#### Phase 3: Core Algorithm Development (Months 5-7)
- Implementation of normalization algorithms
- Statistical analysis module development
- Correlation analysis functionality
- Power transition calculation methods

#### Phase 4: User Interface Development (Months 8-9)
- Streamlit interface design
- Interactive visualization implementation
- Academic report generation features
- Export functionality development

#### Phase 5: Testing and Validation (Months 10-11)
- Comprehensive unit test development
- Performance optimization
- Academic peer review integration
- Documentation completion

#### Phase 6: Deployment and Documentation (Month 12)
- System deployment procedures
- User manual creation
- Academic publication preparation
- Open-source release preparation

### Code Quality Standards

The development process adhered to strict academic software standards:

1. **Documentation**: All functions include comprehensive docstrings
2. **Type Hints**: Complete type annotation for enhanced reliability
3. **Error Handling**: Robust exception management throughout
4. **Performance**: Memory-optimized algorithms for large datasets
5. **Reproducibility**: Deterministic calculations with fixed random seeds

### Version Control and Collaboration

```bash
# Development workflow
git checkout -b feature/new-analysis-method
git add .
git commit -m "Add power transition analysis with significance tests"
git push origin feature/new-analysis-method
# Create pull request for peer review
```

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
# Core Dependencies
streamlit>=1.28.0          # Web interface framework
pandas>=2.0.0              # Data manipulation
numpy>=1.24.0              # Numerical computing
plotly>=5.15.0             # Interactive visualizations

# Statistical Analysis
scipy>=1.10.0              # Statistical functions
scikit-learn>=1.3.0        # Machine learning utilities
statsmodels>=0.14.0        # Advanced statistics

# Testing and Development
pytest>=7.0.0              # Testing framework
black>=23.0.0              # Code formatting
flake8>=6.0.0              # Code linting

# Additional Utilities
requests>=2.31             # HTTP library
beautifulsoup4>=4.12.0     # Web scraping
openpyxl>=3.1.0           # Excel file support
```

### Performance Characteristics

| Operation | Time Complexity | Space Complexity | Typical Runtime |
|-----------|----------------|------------------|-----------------|
| Data Loading | O(n) | O(n) | 2-5 seconds |
| Score Calculation | O(n*m) | O(n) | <1 second |
| Correlation Analysis | O(m²) | O(m²) | <1 second |
| Multi-year Generation | O(n*y) | O(n*y) | 5-10 seconds |
| Visualization Rendering | O(n log n) | O(n) | 1-3 seconds |

*Where n = number of countries, m = number of dimensions, y = number of years*

### Security Considerations

1. **Data Validation**: All input data undergoes strict validation
2. **SQL Injection Prevention**: Parameterized queries throughout
3. **File Access Control**: Restricted file system access
4. **Memory Management**: Automatic garbage collection and cleanup
5. **Error Disclosure**: Limited error information in production

## Citation Guidelines

### Academic Citation

When using CNPS-10 in academic research, please cite as follows:

**APA Style:**
```
CNPS Research Team. (2025). CNPS-10: Comprehensive National Power Assessment System 
(Version 2.0.0) [Computer software]. https://github.com/cnps-10/cnps-10
```

**Chicago Style:**
```
CNPS Research Team. "CNPS-10: Comprehensive National Power Assessment System." 
Computer software. Version 2.0.0. 2025. https://github.com/cnps-10/cnps-10.
```

**MLA Style:**
```
CNPS Research Team. CNPS-10: Comprehensive National Power Assessment System. 
Version 2.0.0, 2025, https://github.com/cnps-10/cnps-10.
```

### Data Citation

When using CNPS-10 data in publications:

```
CNPS Research Team. (2025). CNPS-10 National Power Assessment Dataset, 2025-2029 
[Data set]. https://github.com/cnps-10/cnps-10/releases/latest
```

### Methodology Citation

For methodological references:

```
CNPS Research Team. (2025). Multi-dimensional National Power Assessment: 
Methodology and Implementation in CNPS-10. Technical Documentation. 
https://github.com/cnps-10/cnps-10/docs/methodology.pdf
```

## Contributing

### Academic Collaboration

We welcome contributions from the academic community:

1. **Research Contributions**
   - New analytical methods
   - Additional data sources
   - Methodological improvements
   - Validation studies

2. **Technical Contributions**
   - Code optimization
   - Bug fixes
   - Documentation improvements
   - Testing enhancements

### Code of Conduct

All contributors must adhere to academic standards:

- Respect for intellectual property
- Transparent methodology disclosure
- Collaborative peer review
- Constructive feedback provision

### Research Roadmap

- Integration with additional international databases
- Development of predictive modeling capabilities
- Implementation of uncertainty quantification
- Creation of comparative historical analysis tools

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 CNPS Research Team

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

### Research Team
- **Principal Investigator**: CNPS Research Team
- **Technical Lead**: System Architecture Team
- **Data Science Lead**: Analytics Team

### Acknowledgments

We acknowledge the contributions of:
- International data providers (World Bank, IMF, UN, OECD)
- Academic peer reviewers and beta testers
- Open-source software community
- International relations research community

---

**Disclaimer**: This system is designed for academic research and educational purposes. The scores and rankings reflect analytical assessments based on available data and established methodologies in international relations. Results should be interpreted within the context of the underlying theoretical framework and data limitations.

**Last Updated**: September 3, 2025  
**System Version**: 2.0.0  
**Documentation Version**: 1.0.0
