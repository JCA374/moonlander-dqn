#!/usr/bin/env python3
"""
Virtual Environment Setup Script for MoonLander DQN Project
Creates and configures a Python virtual environment with all dependencies.
"""

import os
import sys
import subprocess
import platform

def run_command(command, shell=True):
    """Run a command and return success status."""
    try:
        result = subprocess.run(command, shell=shell, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🚀 MoonLander DQN Environment Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        sys.exit(1)
    
    print(f"✓ Python {sys.version.split()[0]} detected")
    
    # Determine OS and set commands
    is_windows = platform.system() == "Windows"
    
    if is_windows:
        python_cmd = "python"
        venv_activate = "venv\\Scripts\\activate.bat"
        pip_cmd = "venv\\Scripts\\pip"
    else:
        python_cmd = "python3"
        venv_activate = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    print(f"✓ Detected OS: {platform.system()}")
    
    # Create virtual environment
    print("\n📦 Creating virtual environment...")
    if not run_command(f"{python_cmd} -m venv venv"):
        print("❌ Failed to create virtual environment!")
        sys.exit(1)
    
    # Upgrade pip
    print("\n🔧 Upgrading pip...")
    if not run_command(f"{pip_cmd} install --upgrade pip"):
        print("⚠️  Warning: Failed to upgrade pip")
    
    # Install requirements
    print("\n📚 Installing dependencies...")
    if os.path.exists("requirements.txt"):
        if not run_command(f"{pip_cmd} install -r requirements.txt"):
            print("❌ Failed to install requirements!")
            sys.exit(1)
    else:
        print("⚠️  requirements.txt not found, installing core packages...")
        packages = [
            "numpy>=1.21.0",
            "tensorflow>=2.10.0", 
            "gymnasium>=0.29.0",
            "pygame>=2.1.0",
            "matplotlib>=3.5.0",
            "psutil>=5.9.0"
        ]
        
        for package in packages:
            if not run_command(f"{pip_cmd} install {package}"):
                print(f"⚠️  Warning: Failed to install {package}")
    
    # Create activation helper scripts
    print("\n📝 Creating activation scripts...")
    
    if is_windows:
        activate_script = """@echo off
echo Activating MoonLander DQN Environment...
call venv\\Scripts\\activate.bat
echo Environment activated! You can now run:
echo   python train.py
echo   python play_best_model.py
"""
        with open("activate_env.bat", "w") as f:
            f.write(activate_script)
        print("✓ Created activate_env.bat")
    
    # Unix/Linux activation script
    activate_script = """#!/bin/bash
echo "Activating MoonLander DQN Environment..."
source venv/bin/activate
echo "Environment activated! You can now run:"
echo "  python train.py"
echo "  python play_best_model.py"
bash
"""
    with open("activate_env.sh", "w") as f:
        f.write(activate_script)
    
    # Make it executable on Unix systems
    if not is_windows:
        run_command("chmod +x activate_env.sh")
    
    print("✓ Created activate_env.sh")
    
    # Success message
    print("\n🎉 Environment setup complete!")
    print("=" * 50)
    print("To activate the environment:")
    if is_windows:
        print("  activate_env.bat")
        print("  OR: venv\\Scripts\\activate.bat")
    else:
        print("  ./activate_env.sh")
        print("  OR: source venv/bin/activate")
    
    print("\nTo start training:")
    print("  python train.py --episodes 200 --restart")
    
    print("\nTo test trained models:")
    print("  python play_best_model.py")

if __name__ == "__main__":
    main()