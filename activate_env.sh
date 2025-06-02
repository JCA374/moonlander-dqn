#!/bin/bash

echo ""
echo "========================================"
echo "   MoonLander DQN Environment"
echo "========================================"
echo ""

if [ ! -f "venv/bin/activate" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please run: python3 setup_env.py"
    exit 1
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Environment activated successfully!"
echo ""
echo "Available commands:"
echo "  python train.py --episodes 200 --restart"
echo "  python play_best_model.py"
echo "  python test_rewards.py"
echo ""
echo "To deactivate: deactivate"
echo ""

# Start an interactive bash session with the environment activated
exec bash --rcfile <(echo '. ~/.bashrc; PS1="(moonlander) $PS1"')