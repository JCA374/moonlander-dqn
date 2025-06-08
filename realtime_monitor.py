#!/usr/bin/env python3
"""
Real-time training monitor with enhanced statistics including hovering detection.
Shows CPU usage, memory usage, success rates, and outcome breakdowns.
"""
import os
import time
import json
import sys
from datetime import datetime
from collections import deque
import re

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Note: Install psutil for system monitoring: pip install psutil")

try:
    from colorama import init, Fore, Back, Style
    init()
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False
    

class TrainingMonitor:
    """Real-time training monitor with detailed statistics."""
    
    def __init__(self):
        self.log_file = "training_monitor.log"
        self.last_episode = 0
        self.outcomes = {
            "success": 0,
            "crash": 0,
            "hover_timeout": 0,
            "timeout": 0,
            "other": 0
        }
        self.recent_rewards = deque(maxlen=100)
        self.start_time = time.time()
        
    def parse_log_line(self, line):
        """Parse a log line for relevant information."""
        # Episode completion with outcome
        if "Episode" in line and "/" in line:
            match = re.search(r'Episode (\d+)/(\d+)', line)
            if match:
                self.last_episode = int(match.group(1))
                
        # Success detection
        if "SUCCESSFUL LANDING" in line:
            self.outcomes["success"] += 1
            match = re.search(r'reward: ([-\d.]+)', line)
            if match:
                self.recent_rewards.append(float(match.group(1)))
                
        # Parse outcome statistics
        if "Crash:" in line:
            match = re.search(r'Success: ([\d.]+)%, Crash: ([\d.]+)%, Hover: ([\d.]+)%', line)
            if match:
                return {
                    'success_rate': float(match.group(1)),
                    'crash_rate': float(match.group(2)),
                    'hover_rate': float(match.group(3))
                }
                
        return None
        
    def get_system_stats(self):
        """Get current system statistics."""
        stats = {}
        
        if HAS_PSUTIL:
            # CPU stats
            stats['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            stats['cpu_per_core'] = psutil.cpu_percent(interval=0.1, percpu=True)
            
            # Memory stats
            memory = psutil.virtual_memory()
            stats['memory_percent'] = memory.percent
            stats['memory_used_gb'] = memory.used / (1024**3)
            stats['memory_available_gb'] = memory.available / (1024**3)
            
            # Process specific stats
            try:
                # Find python processes running training
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    if 'python' in proc.info['name'].lower():
                        cmdline = ' '.join(proc.info['cmdline'] or [])
                        if 'train' in cmdline and 'enhanced' in cmdline:
                            p = psutil.Process(proc.info['pid'])
                            stats['process_cpu'] = p.cpu_percent(interval=0.1)
                            stats['process_memory_mb'] = p.memory_info().rss / (1024**2)
                            break
            except:
                pass
                
        return stats
        
    def display_stats(self, stats, rates):
        """Display formatted statistics."""
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # Header
        if HAS_COLOR:
            print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}🚀 MOONLANDER ENHANCED TRAINING MONITOR{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        else:
            print("="*80)
            print("🚀 MOONLANDER ENHANCED TRAINING MONITOR")
            print("="*80)
            
        # Time and progress
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        
        print(f"\n⏱️  Training Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"📊 Current Episode: {self.last_episode}")
        
        if self.last_episode > 0:
            eps_per_hour = self.last_episode / (elapsed / 3600)
            print(f"⚡ Speed: {eps_per_hour:.1f} episodes/hour")
            
        # System resources
        print(f"\n{'-'*40}")
        print("💻 SYSTEM RESOURCES")
        print(f"{'-'*40}")
        
        if stats:
            # CPU usage with color coding
            cpu_color = ""
            if HAS_COLOR:
                if stats['cpu_percent'] > 80:
                    cpu_color = Fore.GREEN
                elif stats['cpu_percent'] > 50:
                    cpu_color = Fore.YELLOW
                else:
                    cpu_color = Fore.RED
                    
            print(f"CPU Total: {cpu_color}{stats['cpu_percent']:5.1f}%{Style.RESET_ALL if HAS_COLOR else ''}")
            
            # Per-core usage
            if 'cpu_per_core' in stats:
                core_str = "CPU Cores: "
                for i, usage in enumerate(stats['cpu_per_core']):
                    if HAS_COLOR:
                        if usage > 70:
                            core_str += f"{Fore.GREEN}{usage:4.0f}%{Style.RESET_ALL} "
                        elif usage > 30:
                            core_str += f"{Fore.YELLOW}{usage:4.0f}%{Style.RESET_ALL} "
                        else:
                            core_str += f"{Fore.RED}{usage:4.0f}%{Style.RESET_ALL} "
                    else:
                        core_str += f"{usage:4.0f}% "
                print(core_str)
                
            # Memory usage
            mem_color = ""
            if HAS_COLOR:
                if stats['memory_percent'] > 80:
                    mem_color = Fore.RED
                elif stats['memory_percent'] > 60:
                    mem_color = Fore.YELLOW
                else:
                    mem_color = Fore.GREEN
                    
            print(f"Memory: {mem_color}{stats['memory_percent']:5.1f}%{Style.RESET_ALL if HAS_COLOR else ''} "
                  f"({stats['memory_used_gb']:.1f}GB / {stats['memory_used_gb'] + stats['memory_available_gb']:.1f}GB)")
                  
            if 'process_cpu' in stats:
                print(f"Training Process: {stats['process_cpu']:.1f}% CPU, "
                      f"{stats['process_memory_mb']:.0f}MB RAM")
        else:
            print("System monitoring unavailable (install psutil)")
            
        # Training statistics
        print(f"\n{'-'*40}")
        print("🎯 TRAINING STATISTICS")
        print(f"{'-'*40}")
        
        if rates:
            # Success rate with color
            success_color = ""
            if HAS_COLOR:
                if rates['success_rate'] > 60:
                    success_color = Fore.GREEN
                elif rates['success_rate'] > 40:
                    success_color = Fore.YELLOW
                else:
                    success_color = Fore.RED
                    
            print(f"Success Rate: {success_color}{rates['success_rate']:.1f}%{Style.RESET_ALL if HAS_COLOR else ''}")
            print(f"Crash Rate: {rates['crash_rate']:.1f}%")
            print(f"Hover Timeout Rate: {rates['hover_rate']:.1f}%")
            
        # Recent performance
        if len(self.recent_rewards) > 0:
            avg_reward = sum(self.recent_rewards) / len(self.recent_rewards)
            max_reward = max(self.recent_rewards)
            min_reward = min(self.recent_rewards)
            
            print(f"\n{'-'*40}")
            print("📈 RECENT PERFORMANCE (last 100 episodes)")
            print(f"{'-'*40}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Best Reward: {max_reward:.2f}")
            print(f"Worst Reward: {min_reward:.2f}")
            
        # Model status
        print(f"\n{'-'*40}")
        print("💾 MODEL STATUS")
        print(f"{'-'*40}")
        
        model_path = "results/current_run/models/best_model.weights.h5"
        if os.path.exists(model_path):
            mod_time = os.path.getmtime(model_path)
            last_update = datetime.fromtimestamp(mod_time).strftime('%H:%M:%S')
            file_size = os.path.getsize(model_path) / (1024*1024)
            print(f"Best Model: Found ({file_size:.1f}MB)")
            print(f"Last Updated: {last_update}")
        else:
            print("Best Model: Not found yet")
            
        # Tips based on performance
        print(f"\n{'-'*40}")
        print("💡 PERFORMANCE TIPS")
        print(f"{'-'*40}")
        
        if stats and stats['cpu_percent'] < 40:
            print("⚠️  Low CPU usage - consider increasing batch size or training frequency")
        elif stats and stats['cpu_percent'] > 90:
            print("⚠️  Very high CPU usage - system might be throttling")
            
        if rates and rates['crash_rate'] > 30:
            print("⚠️  High crash rate - model might need more exploration or reward tuning")
            
        if rates and rates['hover_rate'] > 10:
            print("⚠️  Significant hovering - anti-hover rewards might need adjustment")
            
        if rates and rates['success_rate'] > 70:
            print("✅ Excellent performance! Consider reducing epsilon for exploitation")
            
        print(f"\n{Fore.CYAN if HAS_COLOR else ''}{'='*80}{Style.RESET_ALL if HAS_COLOR else ''}")
        print("Press Ctrl+C to stop monitoring")
        
    def monitor_loop(self):
        """Main monitoring loop."""
        print("Starting training monitor...")
        print("Waiting for training data...")
        
        last_rates = None
        
        while True:
            try:
                # Get system stats
                stats = self.get_system_stats()
                
                # Read latest training logs (simplified approach)
                # In a real implementation, you might tail the actual training log file
                
                # Display current stats
                self.display_stats(stats, last_rates)
                
                # Update every 5 seconds
                time.sleep(5)
                
            except KeyboardInterrupt:
                print("\n\nMonitoring stopped.")
                break
            except Exception as e:
                print(f"\nMonitor error: {e}")
                time.sleep(5)


def main():
    """Main entry point."""
    monitor = TrainingMonitor()
    monitor.monitor_loop()


if __name__ == "__main__":
    main()
    