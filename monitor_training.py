#!/usr/bin/env python3
"""
Training monitor for Intel i7-8565U optimized training.
Shows real-time performance metrics and progress.
"""
import os
import time
import json
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Note: Install psutil for system monitoring: pip install psutil")

from datetime import datetime

def monitor_training():
    """Monitor training progress in real-time."""
    print("MoonLander Training Monitor - Intel i7-8565U")
    print("=" * 60)
    
    log_file = "training_monitor.log"
    
    while True:
        try:
            # System metrics (if psutil available)
            if HAS_PSUTIL:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_used_gb = memory.used / (1024**3)
                system_info = f"CPU: {cpu_percent:5.1f}% | RAM: {memory_percent:5.1f}% ({memory_used_gb:.1f}GB)"
            else:
                system_info = "System monitoring unavailable (install psutil)"
            
            # Check for model updates
            model_path = "results/current_run/models/best_model.weights.h5"
            model_status = "Found" if os.path.exists(model_path) else "Not found"
            
            if os.path.exists(model_path):
                mod_time = os.path.getmtime(model_path)
                last_update = datetime.fromtimestamp(mod_time).strftime('%H:%M:%S')
            else:
                last_update = "N/A"
            
            # Display metrics
            current_time = datetime.now().strftime('%H:%M:%S')
            print(f"\r[{current_time}] {system_info} | Model: {model_status} | Last update: {last_update}", end="")
            
            # Log to file
            with open(log_file, 'a', encoding='utf-8') as f:
                if HAS_PSUTIL:
                    f.write(f"{current_time}, {cpu_percent:.1f}, {memory_percent:.1f}, {model_status}\n")
                else:
                    f.write(f"{current_time}, N/A, N/A, {model_status}\n")
            
            time.sleep(10)  # Update every 10 seconds
            
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"\nMonitor error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_training()
