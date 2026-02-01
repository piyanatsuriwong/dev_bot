#!/bin/bash
# Pibot Voice - Installation Script

set -e

echo "🤖 Pibot Voice Installation"
echo "=========================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Raspberry Pi
check_pi() {
    if [ -f /proc/device-tree/model ]; then
        model=$(cat /proc/device-tree/model)
        echo -e "${GREEN}✓${NC} Running on: $model"
    else
        echo -e "${YELLOW}⚠${NC} Not running on Raspberry Pi"
    fi
}

# Install system dependencies
install_system_deps() {
    echo ""
    echo "📦 Installing system dependencies..."
    
    sudo apt-get update
    sudo apt-get install -y \
        python3-pip \
        python3-venv \
        python3-pyaudio \
        portaudio19-dev \
        ffmpeg \
        mpv \
        libatlas-base-dev \
        libopenblas-dev
    
    echo -e "${GREEN}✓${NC} System dependencies installed"
}

# Create virtual environment
setup_venv() {
    echo ""
    echo "🐍 Setting up Python virtual environment..."
    
    VENV_DIR="$(dirname "$0")/../venv"
    
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
        echo -e "${GREEN}✓${NC} Virtual environment created"
    else
        echo -e "${YELLOW}⚠${NC} Virtual environment already exists"
    fi
    
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
}

# Install Python packages
install_python_deps() {
    echo ""
    echo "📦 Installing Python packages..."
    
    REQUIREMENTS="$(dirname "$0")/../requirements.txt"
    
    pip install -r "$REQUIREMENTS"
    
    echo -e "${GREEN}✓${NC} Python packages installed"
}

# Install Whisper (optional, takes time)
install_whisper() {
    echo ""
    read -p "Install local Whisper? (slow, ~5min) [y/N]: " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "📦 Installing Whisper (this may take a while)..."
        pip install openai-whisper
        echo -e "${GREEN}✓${NC} Whisper installed"
    else
        echo -e "${YELLOW}⚠${NC} Skipping Whisper (will use API mode)"
    fi
}

# Test installation
test_installation() {
    echo ""
    echo "🧪 Testing installation..."
    
    # Test TTS
    python3 -c "import edge_tts; print('  ✓ edge-tts')" 2>/dev/null || echo "  ✗ edge-tts"
    
    # Test PyAudio
    python3 -c "import pyaudio; print('  ✓ pyaudio')" 2>/dev/null || echo "  ✗ pyaudio"
    
    # Test Whisper
    python3 -c "import whisper; print('  ✓ whisper (local)')" 2>/dev/null || echo "  ⚠ whisper (API mode)"
    
    # Test aiohttp
    python3 -c "import aiohttp; print('  ✓ aiohttp')" 2>/dev/null || echo "  ✗ aiohttp"
    
    echo ""
    echo -e "${GREEN}✓${NC} Installation complete!"
}

# Print usage
print_usage() {
    echo ""
    echo "📖 Usage:"
    echo "   source venv/bin/activate"
    echo "   python3 src/main.py"
    echo ""
    echo "🧪 Test commands:"
    echo "   python3 src/main.py --list-devices"
    echo "   python3 src/main.py --test-tts 'สวัสดีครับ'"
    echo ""
}

# Main
main() {
    cd "$(dirname "$0")/.."
    
    check_pi
    install_system_deps
    setup_venv
    install_python_deps
    install_whisper
    test_installation
    print_usage
}

main "$@"
