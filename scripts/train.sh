#!/bin/bash
set -e

CONFIG="$1"
if [ -z "$CONFIG" ]; then
    echo "Usage: $0 <config.json>"
    echo "Available configs:"
    ls configs/*.json 2>/dev/null || echo "  (none)"
    exit 1
fi

echo "=== K-Mamba Pipeline ==="
echo "Configuration: $CONFIG"
echo ""

# Phase 1: Model creation
echo "[1/3] Creating model..."
./model "$CONFIG"
if [ $? -ne 0 ]; then
    echo "Model creation failed."
    exit 1
fi

# Phase 2: Confirmation
echo ""
read -p "[2/3] Launch training? (y/n) " confirm
if [ "$confirm" != "y" ]; then
    echo "Training cancelled."
    exit 0
fi

# Phase 3: Training
echo ""
echo "[3/3] Starting training..."
./train "$CONFIG" "$@"
echo ""
echo "=== Pipeline complete ==="
