#!/usr/bin/env python3
"""Quick training test - Intel i7-8565U optimized"""
import subprocess
import sys

print("Starting quick training test (200 episodes)...")
cmd = [
    sys.executable, "train_optimized.py",
    "--episodes", "200",
    "--batch-size", "128", 
    "--memory-size", "200000",
    "--learning-rate", "0.001",
    "--epsilon-decay", "0.998",
    "--epsilon-min", "0.02",
    "--update-target-freq", "10",
    "--train-freq", "2",
    "--eval-freq", "25"
]

subprocess.run(cmd)
