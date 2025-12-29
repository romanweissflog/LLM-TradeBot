#!/bin/bash
# ============================================
# LLM-TradeBot 一键安装脚本
# ============================================

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

# Banner
echo "============================================"
echo "🤖 LLM-TradeBot Installation"
echo "============================================"
echo ""

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    *)          MACHINE="UNKNOWN:${OS}"
esac

print_info "Detected OS: ${MACHINE}"

# Check Python version
print_info "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
        print_success "Python $PYTHON_VERSION found"
        PYTHON_CMD=python3
    else
        print_error "Python 3.11+ required, found $PYTHON_VERSION"
        print_info "Please install Python 3.11 or higher"
        exit 1
    fi
else
    print_error "Python 3 not found"
    print_info "Please install Python 3.11+"
    exit 1
fi

# Check if virtual environment exists
VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
    print_warning "Virtual environment already exists"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        print_info "Using existing virtual environment"
    fi
fi

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    print_info "Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip -q

# Install dependencies
print_info "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt -q
    print_success "Dependencies installed"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Setup .env file
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        print_info "Creating .env file from .env.example..."
        cp .env.example .env
        print_warning "Please edit .env file and add your API keys"
        print_info "Required keys: BINANCE_API_KEY, BINANCE_SECRET_KEY, DEEPSEEK_API_KEY"
    else
        print_error ".env.example not found"
        exit 1
    fi
else
    print_success ".env file already exists"
fi

# Create necessary directories
print_info "Creating data directories..."
mkdir -p data logs models

# Installation complete
echo ""
echo "============================================"
print_success "Installation Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env file with your API keys"
echo "  2. Run: ./start.sh --test --mode continuous"
echo ""
echo "For Docker deployment:"
echo "  docker-compose up -d"
echo ""
