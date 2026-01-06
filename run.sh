#!/bin/bash

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Sound Hunter Project Setup ===${NC}"

# Function to get Python version
get_python_version() {
    "$1" --version 2>&1 | awk '{print $2}'
}

# Default Python command
PYTHON_CMD="python3"

# Check if Python 3 is installed and get version
if ! command -v python3 &> /dev/null; then
    PYTHON_VERSION=""
else
    PYTHON_VERSION=$(get_python_version python3)
fi

# Check if version is 3.10 or higher
if [[ -n "$PYTHON_VERSION" && "$(printf '%s\n' "$PYTHON_VERSION" "3.10" | sort -V | head -n1)" == "3.10" ]]; then
    echo -e "${YELLOW}Found suitable Python version: $PYTHON_VERSION${NC}"
else
    echo -e "${YELLOW}Python 3.10+ not found. Attempting to install...${NC}"
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # Mac OSX - try Homebrew
        if command -v brew &> /dev/null; then
            echo -e "${YELLOW}Installing Python 3.10 via Homebrew...${NC}"
            brew install python@3.10
            PYTHON_CMD="/opt/homebrew/bin/python3.10"  # Apple Silicon; adjust to /usr/local/bin/python3.10 for Intel
        else
            echo -e "${RED}Homebrew not found. Please install Python 3.10 manually or via another method.${NC}"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux - try apt
        if command -v apt-get &> /dev/null; then
            echo -e "${YELLOW}Installing Python 3.10 via apt...${NC}"
            sudo apt-get update
            sudo apt-get install -y python3.10 python3.10-venv
            PYTHON_CMD="python3.10"
        else
            echo -e "${RED}apt not found. Please install Python 3.10 manually.${NC}"
            exit 1
        fi
    else
        echo -e "${RED}Unsupported OS. Please install Python 3.10 manually.${NC}"
        exit 1
    fi
    
    # Verify the installed Python
    if ! command -v "$PYTHON_CMD" &> /dev/null; then
        echo -e "${RED}Failed to install Python 3.10.${NC}"
        exit 1
    fi
    
    PYTHON_VERSION=$(get_python_version "$PYTHON_CMD")
    if [[ "$(printf '%s\n' "$PYTHON_VERSION" "3.10" | sort -V | head -n1)" != "3.10" ]]; then
        echo -e "${RED}Installed Python version $PYTHON_VERSION is not 3.10+.${NC}"
        exit 1
    fi
fi

echo -e "${YELLOW}Using Python version: $PYTHON_VERSION${NC}"

# Check if venv module is available
if ! "$PYTHON_CMD" -m venv --help &> /dev/null; then
    echo -e "${RED}Error: venv module not found!${NC}"
    echo "Installing python3-venv..."
    
    # Try to install venv based on the OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y python3-venv
        elif command -v yum &> /dev/null; then
            sudo yum install -y python3-venv
        else
            echo -e "${RED}Please install python3-venv manually${NC}"
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # Mac OSX - venv should be included with Python 3
        echo -e "${RED}venv should be included with Python 3.10+ on macOS${NC}"
        echo "Try: brew install python@3.10"
        exit 1
    fi
fi

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Removing old one..."
    rm -rf venv
fi

"$PYTHON_CMD" -m venv venv

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

echo -e "\n${YELLOW}Installing required packages...${NC}"
# Install required packages
pip install -r requirements.txt

chmod +x build_components.sh
./build_components.sh

echo "Testing Tesseract pipeline..."
python test_pipeline.py

echo "Training the system..."
python train_system.py

echo "Retraining for demo purpose..."
python demo_retrain.py

# Cleanup Docker containers to free ports
echo -e "\n${YELLOW}Cleaning up Docker containers...${NC}"
docker ps -aq --filter ancestor=audio-filter:latest | xargs docker rm -f 2>/dev/null || true
docker ps -aq --filter ancestor=feature-extractor:latest | xargs docker rm -f 2>/dev/null || true
docker ps -aq --filter ancestor=pattern-detector:latest | xargs docker rm -f 2>/dev/null || true

echo -e "${GREEN}Setup and demo complete!${NC}"