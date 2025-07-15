#!/bin/bash

# Domain Name Suggestion LLM - Environment Setup Script
# This script sets up the complete development environment

set -e # Exit on any error

echo "🚀 Setting up Domain Name Suggestion LLM Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.8+ is installed
check_python() {
    print_status "Checking Python version..."
    if command -v python3 &>/dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        REQUIRED_VERSION="3.8"

        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_success "Python $PYTHON_VERSION found"
        else
            print_error "Python 3.8+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi
}

# Check if pip is installed
check_pip() {
    print_status "Checking pip installation..."
    if command -v pip3 &>/dev/null; then
        print_success "pip3 found"
    else
        print_error "pip3 not found. Please install pip"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Upgrade pip
upgrade_pip() {
    print_status "Upgrading pip..."
    pip install --upgrade pip
    print_success "pip upgraded"
}

# Install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    print_success "Dependencies installed"
}

# Create necessary directories
create_directories() {
    print_status "Creating project directories..."

    # Create data directories
    mkdir -p data/synthetic
    mkdir -p data/evaluation
    mkdir -p data/edge_cases

    # Create model directories
    mkdir -p models/baseline
    mkdir -p models/v1_lora
    mkdir -p models/v2_qlora

    # Create logs directory
    mkdir -p logs

    # Create config directory if it doesn't exist
    mkdir -p config

    print_success "Project directories created"
}

# Setup Git hooks (optional)
setup_git_hooks() {
    if [ -d ".git" ]; then
        print_status "Setting up Git hooks..."

        # Create pre-commit hook
        cat >.git/hooks/pre-commit <<'EOF'
#!/bin/bash
# Pre-commit hook for code quality checks

echo "Running pre-commit checks..."

# Run black formatting check
if command -v black &> /dev/null; then
    black --check --diff .
    if [ $? -ne 0 ]; then
        echo "Code formatting issues found. Run 'black .' to fix."
        exit 1
    fi
fi

# Run flake8 linting
if command -v flake8 &> /dev/null; then
    flake8 src/ tests/ --max-line-length=88 --ignore=E203,W503
    if [ $? -ne 0 ]; then
        echo "Linting issues found."
        exit 1
    fi
fi

echo "Pre-commit checks passed!"
EOF

        chmod +x .git/hooks/pre-commit
        print_success "Git hooks configured"
    else
        print_warning "Git repository not found, skipping Git hooks setup"
    fi
}

# Check for GPU support
check_gpu() {
    print_status "Checking GPU support..."

    if command -v nvidia-smi &>/dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
        print_success "GPU detected: $GPU_INFO"

        # Check CUDA version
        if command -v nvcc &>/dev/null; then
            CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
            print_success "CUDA version: $CUDA_VERSION"
        else
            print_warning "CUDA not found, but GPU detected. PyTorch will use CPU."
        fi
    else
        print_warning "No GPU detected. Training will use CPU (slower)."
    fi
}

# Setup environment variables
setup_env_vars() {
    print_status "Setting up environment variables..."

    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        cat >.env <<'EOF'
# Domain Name Suggestion LLM Environment Variables

# API Keys (set these for evaluation)
# OPENAI_API_KEY=your_openai_api_key_here
# ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Model Configuration
DEFAULT_MODEL_VERSION=v2_qlora
BASE_MODEL_NAME=meta-llama/Llama-2-7b-hf

# Training Configuration
DEFAULT_BATCH_SIZE=16
DEFAULT_LEARNING_RATE=2e-4
DEFAULT_NUM_EPOCHS=7

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Logging Configuration
LOG_LEVEL=INFO
WANDB_PROJECT=domain-name-suggestion-llm
WANDB_ENTITY=

# Safety Configuration
SAFETY_ENABLED=true
MIN_CONFIDENCE_THRESHOLD=0.7
MAX_SUGGESTIONS_PER_REQUEST=5
EOF

        print_success "Environment variables file created (.env)"
        print_warning "Please edit .env file to add your API keys if needed"
    else
        print_warning ".env file already exists"
    fi
}

# Test installation
test_installation() {
    print_status "Testing installation..."

    # Test Python imports
    python3 -c "
import sys
sys.path.append('src')

try:
    from data.dataset_creation import SyntheticDatasetCreator
    from evaluation.safety_checker import SafetyChecker
    from models.domain_generator import DomainGenerator
    print('✅ All core modules imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

    if [ $? -eq 0 ]; then
        print_success "Installation test passed"
    else
        print_error "Installation test failed"
        exit 1
    fi
}

# Display next steps
show_next_steps() {
    echo ""
    echo "🎉 Environment setup completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "1. Activate virtual environment: source venv/bin/activate"
    echo "2. Create dataset: python scripts/create_dataset.py"
    echo "3. Train model: python scripts/train_model.py --model v2_qlora"
    echo "4. Run evaluation: python scripts/evaluate_model.py --model v2_qlora"
    echo "5. Start API: uvicorn src.api.main:app --reload"
    echo ""
    echo "📚 Documentation:"
    echo "- README.md: Project overview and quick start"
    echo "- docs/technical_report.md: Detailed technical report"
    echo "- notebooks/: Jupyter notebooks for experiments"
    echo ""
    echo "🔧 Configuration:"
    echo "- Edit config/model_config.yaml for model settings"
    echo "- Edit config/evaluation_config.yaml for evaluation settings"
    echo "- Edit .env for environment variables"
    echo ""
}

# Main setup function
main() {
    echo "=========================================="
    echo "Domain Name Suggestion LLM Setup"
    echo "=========================================="
    echo ""

    # Run setup steps
    check_python
    check_pip
    create_venv
    activate_venv
    upgrade_pip
    install_dependencies
    create_directories
    setup_git_hooks
    check_gpu
    setup_env_vars
    test_installation

    show_next_steps
}

# Run main function
main "$@"
