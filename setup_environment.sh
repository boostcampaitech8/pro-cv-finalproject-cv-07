#!/bin/bash
# TFT 프로젝트 환경 설정 스크립트

echo "========================================="
echo "TFT Stock Prediction - Environment Setup"
echo "========================================="

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. Python 버전 확인
echo -e "\n${YELLOW}[1/5] Checking Python version...${NC}"
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if [[ $(echo $python_version | cut -d. -f1) -lt 3 ]] || \
   [[ $(echo $python_version | cut -d. -f1) -eq 3 && $(echo $python_version | cut -d. -f2) -lt 8 ]]; then
    echo -e "${RED}Error: Python 3.8+ required${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version OK${NC}"

# 2. Pip 업그레이드
echo -e "\n${YELLOW}[2/5] Upgrading pip...${NC}"
pip install --upgrade pip
echo -e "${GREEN}✓ Pip upgraded${NC}"

# 3. PyTorch 설치 (CUDA 확인)
echo -e "\n${YELLOW}[3/5] Installing PyTorch...${NC}"

# CUDA 사용 가능 여부 확인
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected. Installing PyTorch with CUDA support..."
    # CUDA 11.8 버전 (가장 호환성 좋음)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "CUDA not detected. Installing CPU version..."
    pip install torch torchvision torchaudio
fi
echo -e "${GREEN}✓ PyTorch installed${NC}"

# 4. 나머지 패키지 설치
echo -e "\n${YELLOW}[4/5] Installing other dependencies...${NC}"
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# 5. 설치 확인
echo -e "\n${YELLOW}[5/5] Verifying installation...${NC}"

# PyTorch 확인
python << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF

# 필수 패키지 확인
python << EOF
import sys
packages = ['numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn', 'tqdm', 'tyro']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f"✓ {pkg}")
    except ImportError:
        print(f"✗ {pkg}")
        missing.append(pkg)

if missing:
    print(f"\n⚠️  Missing packages: {', '.join(missing)}")
    sys.exit(1)
else:
    print("\n✓ All packages installed successfully!")
EOF

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}=========================================${NC}"
    echo -e "${GREEN}Environment setup completed successfully!${NC}"
    echo -e "${GREEN}=========================================${NC}"
    
    echo -e "\n${YELLOW}Next steps:${NC}"
    echo "1. Activate your virtual environment (if not already activated)"
    echo "2. Navigate to your project directory"
    echo "3. Run: python scripts/train_tft.py --help"
else
    echo -e "\n${RED}=========================================${NC}"
    echo -e "${RED}Environment setup failed!${NC}"
    echo -e "${RED}=========================================${NC}"
    exit 1
fi
