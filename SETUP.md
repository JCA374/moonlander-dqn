# 🚀 MoonLander DQN - Quick Setup Guide

## Environment Setup

### Option 1: Automated Setup (Recommended)
```bash
python3 setup_env.py
```

### Option 2: Manual Setup
```bash
# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate    # Linux/Mac
# OR
venv\Scripts\activate.bat   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Activation

### Windows
```cmd
activate_env.bat
```

### Linux/Mac
```bash
./activate_env.sh
```

## Quick Start Training

Once environment is activated:

```bash
# Start fresh training (recommended after improvements)
python train.py --episodes 200 --restart

# Continue existing training
python train.py --episodes 500

# Test trained model
python play_best_model.py
```

## Environment Variables (Optional)

For optimal performance on Intel CPUs:
```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TF_ENABLE_ONEDNN_OPTS=1
```

## Troubleshooting

**Issue**: `python: command not found`
**Solution**: Use `python3` instead of `python`

**Issue**: Permission denied on scripts
**Solution**: `chmod +x setup_env.py activate_env.sh`

**Issue**: TensorFlow warnings
**Solution**: These are normal and can be ignored. The CPU optimizations are working correctly.

## Training Tips

- Start with `--episodes 200 --restart` to test improvements
- Monitor for landing behavior vs hovering
- Expected: 80%+ success rate within 500 episodes
- Training should be 2-3x faster with the new optimizations