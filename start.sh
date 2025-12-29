#!/bin/bash
# ============================================
# LLM-TradeBot 一键启动脚本
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
echo "🚀 LLM-TradeBot Startup"
echo "============================================"
echo ""

# Check if virtual environment exists
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    print_error "Virtual environment not found"
    print_info "Please run ./install.sh first"
    exit 1
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Check .env file
if [ ! -f ".env" ]; then
    print_error ".env file not found"
    print_info "Please create .env file with your API keys"
    exit 1
fi

# Check required environment variables
print_info "Checking environment variables..."
source .env

MISSING_VARS=()
[ -z "$BINANCE_API_KEY" ] && MISSING_VARS+=("BINANCE_API_KEY")
[ -z "$BINANCE_SECRET_KEY" ] && MISSING_VARS+=("BINANCE_SECRET_KEY")
[ -z "$DEEPSEEK_API_KEY" ] && MISSING_VARS+=("DEEPSEEK_API_KEY")

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    print_error "Missing required environment variables:"
    for var in "${MISSING_VARS[@]}"; do
        echo "  - $var"
    done
    print_info "Please edit .env file and add missing keys"
    exit 1
fi

print_success "Environment variables OK"

# Parse arguments
TEST_MODE=""
RUN_MODE=""
EXTRA_ARGS=""

for arg in "$@"; do
    case $arg in
        --test)
            TEST_MODE="--test"
            ;;
        --mode)
            shift
            RUN_MODE="--mode $1"
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $arg"
            ;;
    esac
done

# Default to test mode with continuous
if [ -z "$TEST_MODE" ]; then
    TEST_MODE="--test"
fi

if [ -z "$RUN_MODE" ]; then
    RUN_MODE="--mode continuous"
fi

# Start the application
echo ""
print_info "Starting LLM-TradeBot..."
print_info "Mode: $TEST_MODE $RUN_MODE"
echo ""
print_success "Dashboard will be available at: http://localhost:8000"
echo ""

# Run main.py
python main.py $TEST_MODE $RUN_MODE $EXTRA_ARGS
