"""
Fresh start with corrected architecture and aggressive landing incentives
"""
import os
import shutil
from datetime import datetime

# First, backup everything
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if os.path.exists("results"):
    backup_dir = f"results_failed_{timestamp}"
    shutil.move("results", backup_dir)
    print(f"Moved failed results to {backup_dir}")

# Create fresh directories
os.makedirs("results/current_run/models", exist_ok=True)
os.makedirs("results/current_run/logs", exist_ok=True)
os.makedirs("results/current_run/plots", exist_ok=True)

print("\nFresh directories created!")
print("\nNow run the aggressive training command below:")
