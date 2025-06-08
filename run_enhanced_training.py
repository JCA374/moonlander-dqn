#!/usr/bin/env python3
"""
Python script to run enhanced training on Windows
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
    """Run the enhanced training with optimal settings."""
    print("="*60)
    print("🚀 Starting Enhanced MoonLander Training")
    print("System: Intel i7-8565U (4 cores, 8 threads), 32GB RAM")
    print("="*60)

    # Create backup if needed
    create_backup()

    # Training arguments
    args = [
        sys.executable,  # Use current Python interpreter
        "enhanced_train.py",
        "--episodes", "3000",
        "--cpu-threads", "8",
        "--batch-size", "128",
        "--memory-size", "100000",
        "--learning-rate", "0.0003",
        "--epsilon-decay", "0.997",
        "--tau", "0.001",
        "--update-target-freq", "15",
        "--save-interval", "50",
        "--double-dqn",
        "--prioritized-replay",
        "--performance-monitor"
    ]

    print("\n🎯 Starting training with enhanced settings...")
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
    # Check if enhanced_train.py exists
    if not os.path.exists("enhanced_train.py"):
        print("❌ Error: enhanced_train.py not found!")
        print("Please make sure you've saved the enhanced training script.")
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
