#!/bin/bash
# CNPS-10 Educational Project Startup Script

set -e  # Exit on any error

echo "🎓 CNPS-10: Educational Data Analysis Tool"
echo "=========================================="
echo "📊 Interactive web interface with:"
echo "   • Data Overview & Rankings"
echo "   • Trend Analysis"
echo "   • Statistical Analysis"
echo "   • Data Export"
echo "   • Methodology Transparency"
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
    echo "📊 Generating educational dataset..."
    python data_generator.py
    echo "✅ Educational dataset ready (174 countries, 51 years)"
else
    echo "✅ Educational data already exists"
fi

# Run tests (optional)
echo ""
read -p "🧪 Run tests to verify functionality? (y/N): " run_tests
if [[ $run_tests =~ ^[Yy]$ ]]; then
    echo "🧪 Running tests..."
    python tests/run_tests.py
    if [ $? -eq 0 ]; then
        echo "✅ All tests passed"
    else
        echo "❌ Some tests failed, but continuing..."
    fi
fi

# Start the web application
echo ""
echo "🚀 Starting CNPS-10 Educational Tool..."
echo "📍 Access at: http://localhost:8501"
echo ""
echo "📚 Features Available:"
echo "   • Interactive data visualization"
echo "   • Educational demonstrations"
echo "   • Data analysis techniques"
echo "   • Export capabilities"
echo ""
echo "⚠️  EDUCATIONAL DISCLAIMER: This tool uses modeled data for learning purposes only."
echo "    Rankings and assessments are not authoritative."
echo ""
echo "🛑 Press Ctrl+C to stop the application"
echo ""

# Start Streamlit
streamlit run main.py
