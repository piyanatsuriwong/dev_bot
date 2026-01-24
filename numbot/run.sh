#!/bin/bash
# NumBot Runner Script
# Usage: ./run.sh [mode]
#   mode: hand, ai, demo (default: hand)

cd "$(dirname "$0")"

MODE=${1:-hand}

echo "Starting NumBot in $MODE mode..."
./venv/bin/python3 main_roboeyes.py --mode "$MODE"
