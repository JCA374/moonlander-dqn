#!/usr/bin/env python3
"""
Python script to run unified training on Windows
Works cross-platform without needing bash or batch files
"""
import os
import sys
import subprocess
import shutil
from datetime import datetime


def create_backup():
    """Create backup of current results if they exist."""
    if os.path.exists("results/current_run"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"results/backups/run_{timestamp}"

        print(f"📦 Creating backup of current results...")
        os.makedirs("results/backups", exist_ok=True)
        shutil.move("results/current_run", backup_dir)
        print(f"✅ Backup created: {backup_dir}")
        return True
    return False


def run_training():
    """Run the unified training with optimal settings."""
    print("="*60)
    print("🚀 Starting Unified MoonLander Training")
    print("System: Intel i7-8565U (4 cores, 8 threads), 32GB RAM")
    print("="*60)

    # Create backup if needed
    create_backup()

    # Training arguments - optimized for 32GB RAM and i7-8565U
    args = [
        sys.executable,  # Use current Python interpreter
        "train_unified.py",
        "--episodes", "3000",
        "--cpu-threads", "8",
        "--batch-size", "256",          # Increased for 32GB RAM
        "--memory-size", "500000",      # 10x larger buffer for 32GB RAM
        "--learning-rate", "0.0003",
        "--epsilon-decay", "0.997",
        "--update-target-freq", "15",
        "--save-interval", "50",
        "--batch-norm",
        "--gradient-clip", "1.0",
        "--dropout", "0.1",
        "--performance-monitor",
        "--eval-freq", "50",
        "--eval-episodes", "5",
        "--prefetch-batches", "6",      # Prefetch more batches with more RAM
        "--mixed-precision"             # Enable for better performance
    ]

    print("\n🎯 Starting training with unified settings...")
    print("Command:", " ".join(args[1:]))
    print("-"*60)

    try:
        # Run the training
        result = subprocess.run(args, check=True)

        print("\n" + "="*60)
        print("✅ Training completed successfully!")
        print("="*60)

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with error code: {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    # Check if train_unified.py exists
    if not os.path.exists("train_unified.py"):
        print("❌ Error: train_unified.py not found!")
        print("Please make sure you've saved the unified training script.")
        sys.exit(1)

    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  Warning: Not running in a virtual environment")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(0)

    # Run the training
    run_training()


if __name__ == "__main__":
    main()