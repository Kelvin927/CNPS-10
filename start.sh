#!/bin/bash
# CNPS-10 Academic Research Platform Startup Script
# Version 2.0 - Enhanced Academic Features

set -e  # Exit on any error

echo "🔬 CNPS-10: Academic Research Platform v2.0"
echo "=============================================="
echo "📊 Enhanced with 7-tab academic interface:"
echo "   • Current Rankings"
echo "   • Trend Analysis" 
echo "   • 🆕 Academic Analysis (Correlation, Power Transitions)"
echo "   • 📈 Statistical Summary (Enhanced)" 
echo "   • 🆕 Geopolitical Mapping"
echo "   • 🆕 Research Reports & Citations"
echo "   • About & Documentation"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "🐍 Python version: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if [ ! -f "venv/.requirements_installed" ]; then
    echo "📥 Installing requirements..."
    pip install --upgrade pip
    pip install -r requirements.txt
    touch venv/.requirements_installed
    echo "✅ Requirements installed successfully"
else
    echo "✅ Requirements already installed"
fi

# Check if data exists, generate if needed
if [ ! -f "outputs/data/cnps10_official_based_172countries_20250903_022541.csv" ]; then
    echo "📊 Generating multi-year research data..."
    python data_generator.py
    echo "✅ Research datasets ready (5 years, 172 countries)"
else
    echo "✅ Research data already exists"
fi

# Run tests (optional)
echo ""
read -p "🧪 Run academic validation tests? (y/N): " run_tests
if [[ $run_tests =~ ^[Yy]$ ]]; then
    echo "🧪 Running academic validation tests..."
    python tests/run_tests.py
    if [ $? -eq 0 ]; then
        echo "✅ All academic tests passed"
    else
        echo "❌ Some tests failed, but continuing..."
    fi
fi

# Start the web application
echo ""
echo "🚀 Starting CNPS-10 Academic Research Platform..."
echo "📍 Access at: http://localhost:8501"
echo ""
echo "📚 New Academic Features Available:"
echo "   • Correlation Matrix Analysis"
echo "   • Power Transition Modeling"
echo "   • Distribution & Concentration Analysis"
echo "   • Multi-dimensional Radar Comparisons"
echo "   • Regional Power Mapping"
echo "   • Exportable Research Reports"
echo "   • Academic Citation Guidelines"
echo ""
echo "🛑 Press Ctrl+C to stop the application"
echo ""

# Start Streamlit
streamlit run main.py
