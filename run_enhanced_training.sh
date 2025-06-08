#!/bin/bash

# Enhanced training script for Intel i7-8565U with 32GB RAM
# This script runs the enhanced training with optimal settings

echo "🚀 Starting Enhanced MoonLander Training"
echo "System: Intel i7-8565U (4 cores, 8 threads), 32GB RAM"
echo "=================================================="

# Create backup of current results if they exist
if [ -d "results/current_run" ]; then
    timestamp=$(date +%Y%m%d_%H%M%S)
    echo "📦 Backing up current results to results/backup_$timestamp"
    mkdir -p results/backups
    mv results/current_run results/backups/run_$timestamp
fi

# Run enhanced training with aggressive settings
python enhanced_train.py \
    --episodes 3000 \
    --cpu-threads 8 \
    --batch-size 128 \
    --memory-size 100000 \
    --learning-rate 0.0003 \
    --epsilon-decay 0.997 \
    --tau 0.001 \
    --update-target-freq 15 \
    --save-interval 50 \
    --double-dqn \
    --dueling \
    --prioritized-replay \
    --performance-monitor

echo "✅ Training complete!"