@echo off
echo.
echo ========================================
echo   MoonLander DQN Environment
echo ========================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please run: python setup_env.py
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Environment activated successfully!
echo.
echo Available commands:
echo   python train.py --episodes 200 --restart
echo   python play_best_model.py
echo   python test_rewards.py
echo.
echo To deactivate: deactivate
echo.