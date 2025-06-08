#!/usr/bin/env python3
"""
Script to continue training from your best previous model
instead of starting from scratch.
"""
import os
import sys
import subprocess
import shutil
from datetime import datetime


def find_best_model():
    """Find the best model from all previous runs."""
    print("\n🔍 Searching for best models...")

    models = []

    # Search in all locations
    search_dirs = ["results/current_run", "results/backups"]

    for base_dir in search_dirs:
        if not os.path.exists(base_dir):
            continue

        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file == "best_model.weights.h5":
                    path = os.path.join(root, file)
                    size = os.path.getsize(path) / (1024 * 1024)  # MB
                    mtime = os.path.getmtime(path)

                    # Check parent directory for run info
                    run_dir = os.path.dirname(os.path.dirname(path))
                    args_file = os.path.join(run_dir, "args.json")

                    models.append({
                        'path': path,
                        'size_mb': size,
                        'mtime': mtime,
                        'run_dir': run_dir,
                        'has_args': os.path.exists(args_file)
                    })

    if not models:
        return None

    # Sort by modification time (most recent first)
    models.sort(key=lambda x: x['mtime'], reverse=True)

    print(f"\nFound {len(models)} models:")
    for i, model in enumerate(models[:5]):
        print(f"\n{i+1}. {model['path']}")
        print(f"   Size: {model['size_mb']:.2f} MB")
        print(
            f"   Modified: {datetime.fromtimestamp(model['mtime']).strftime('%Y-%m-%d %H:%M')}")

    # Let user choose
    if len(models) > 1:
        choice = input(
            f"\nSelect model [1-{min(5, len(models))}] or press Enter for most recent: ")
        if choice.isdigit() and 1 <= int(choice) <= min(5, len(models)):
            return models[int(choice)-1]['path']

    return models[0]['path']


def run_continued_training():
    """Run training continuing from best model."""
    print("="*60)
    print("🚀 CONTINUE TRAINING FROM BEST MODEL")
    print("="*60)

    # Find best model
    best_model_path = find_best_model()

    if not best_model_path:
        print("\n❌ No previous models found!")
        print("You'll need to train from scratch first.")
        return

    print(f"\n✅ Selected model: {best_model_path}")

    # Backup current run if exists
    if os.path.exists("results/current_run"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"results/backups/run_{timestamp}"
        print(f"\n📦 Backing up current run to: {backup_dir}")
        os.makedirs("results/backups", exist_ok=True)
        shutil.move("results/current_run", backup_dir)

    # Conservative settings for continued training
    args = [
        sys.executable,
        "train_unified.py",
        "--model-path", best_model_path,
        "--episodes", "1000",  # Moderate episode count
        "--batch-size", "128",  # Conservative batch size
        "--memory-size", "200000",
        "--learning-rate", "0.0001",  # Lower learning rate for fine-tuning
        "--epsilon", "0.1",  # Start with low exploration since model is trained
        "--epsilon-min", "0.01",
        "--epsilon-decay", "0.995",
        "--update-target-freq", "50",  # Less frequent updates for stability
        "--save-interval", "25",
        "--eval-freq", "25",
        "--eval-episodes", "5",
        "--cpu-threads", "6",
        "--performance-monitor",
        # Start without aggressive optimizations
        # Can add these if training is stable:
        # "--batch-norm",
        # "--mixed-precision",
        # "--enable-xla",
    ]

    print("\n🎯 Training settings:")
    print("- Starting from pre-trained model")
    print("- Lower learning rate (0.0001) for fine-tuning")
    print("- Low epsilon (0.1) since model already knows basics")
    print("- Conservative batch size (128)")
    print("- 1000 additional episodes")
    print("-"*60)

    response = input("\nProceed with continued training? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    try:
        result = subprocess.run(args, check=True)
        print("\n✅ Continued training completed successfully!")

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")


def run_test_play():
    """Test play the best model to check its performance."""
    print("\n🎮 TESTING BEST MODEL")
    print("="*60)

    best_model_path = find_best_model()

    if not best_model_path:
        print("No model found to test.")
        return

    args = [
        sys.executable,
        "enhanced_play_best_model.py",
        "--model-path", best_model_path,
        "--episodes", "5",
        "--detailed"
    ]

    print(f"\nTesting model: {best_model_path}")
    print("Running 5 test episodes...")

    try:
        subprocess.run(args, check=True)
    except Exception as e:
        print(f"Test failed: {e}")


def main():
    """Main menu for continued training."""
    print("""
    🚀 MOONLANDER CONTINUED TRAINING
    ================================
    """)

    while True:
        print("\nOptions:")
        print("1. Continue training from best model")
        print("2. Test play best model")
        print("3. Run diagnostic (check diagnose_training.py)")
        print("4. Exit")

        choice = input("\nSelect option [1-4]: ")

        if choice == '1':
            run_continued_training()
        elif choice == '2':
            run_test_play()
        elif choice == '3':
            if os.path.exists("diagnose_training.py"):
                subprocess.run([sys.executable, "diagnose_training.py"])
            else:
                print("Run the diagnostic script first!")
        elif choice == '4':
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
