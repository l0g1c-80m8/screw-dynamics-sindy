#!/bin/bash

# Activation script for Screw Dynamics SINDy environment
# Usage: source activate_env.sh

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🔧 Activating Screw Dynamics SINDy Environment...${NC}"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}❌ Virtual environment not found. Please run setup first.${NC}"
    exit 1
fi

# Activate the virtual environment
source .venv/bin/activate

# Verify activation
if [ "$VIRTUAL_ENV" != "" ]; then
    echo -e "${GREEN}✅ Virtual environment activated: $VIRTUAL_ENV${NC}"
else
    echo -e "${RED}❌ Failed to activate virtual environment${NC}"
    exit 1
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
echo -e "${GREEN}✅ PYTHONPATH set to include current directory${NC}"

# Test core imports
echo -e "${YELLOW}🧪 Testing core imports...${NC}"
python -c "
import torch
import numpy
import pandas
print(f'✅ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')
print('✅ NumPy and Pandas imported successfully')

try:
    from src.model import SindyModel
    print('✅ SindyModel imported successfully')
except ImportError as e:
    print(f'⚠️  SindyModel import issue: {e}')

try:
    from baseline.model import LSTMModel, MLP
    print('✅ Baseline models imported successfully')
except ImportError as e:
    print(f'⚠️  Baseline models import issue: {e}')

print('\\n🎉 Environment is ready for development!')
"

echo -e "${GREEN}🚀 Environment activated successfully!${NC}"
echo -e "${YELLOW}Available commands:${NC}"
echo "  python src/main.py --help     # Train SINDy model"
echo "  jupyter notebook              # Start notebooks"
echo "  pytest tests/                 # Run tests"
echo "  make help                     # See all available commands"
echo ""
echo -e "${YELLOW}To deactivate: ${NC}deactivate"