import os
import time
import json
from datetime import datetime


def monitor_training():
    """Monitor training progress overnight."""
    log_file = "overnight_monitor.log"

    while True:
        try:
            # Check for best model
            model_path = "results/current_run/models/best_model.weights.h5"
            if os.path.exists(model_path):
                mod_time = os.path.getmtime(model_path)
                time_str = datetime.fromtimestamp(
                    mod_time).strftime('%Y-%m-%d %H:%M:%S')

                with open(log_file, 'a') as f:
                    f.write(
                        f"{datetime.now()}: Best model updated at {time_str}\n")

            # Sleep for 5 minutes
            time.sleep(300)

        except KeyboardInterrupt:
            break
        except Exception as e:
            with open(log_file, 'a') as f:
                f.write(f"{datetime.now()}: Error - {str(e)}\n")


if __name__ == "__main__":
    monitor_training()
