#!/bin/bash

# setup.sh - Setup Python 3.11 virtual environment and install dependencies

set -e  # Exit on error

echo "🔧 Setting up Python environment..."

# Check if Python 3.11 is available
if ! command -v python3.11 &> /dev/null; then
    echo "❌ Error: Python 3.11 is not installed"
    echo "Please install Python 3.11 first:"
    echo "  - macOS: brew install python@3.11"
    echo "  - Ubuntu/Debian: sudo apt install python3.11 python3.11-venv"
    exit 1
fi

echo "✓ Found Python 3.11: $(python3.11 --version)"

# Create virtual environment
if [ -d ".venv" ]; then
    echo "⚠️  Virtual environment already exists at .venv"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing virtual environment..."
        rm -rf .venv
    else
        echo "ℹ️  Using existing virtual environment"
    fi
fi

if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment with Python 3.11..."
    python3.11 -m venv .venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "📥 Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    echo "✓ Dependencies installed successfully"
else
    echo "⚠️  No requirements.txt found, skipping dependency installation"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
