# 🚀 MoonLander - Deep Reinforcement Learning

A comprehensive reinforcement learning project that trains an intelligent agent to master lunar landing using OpenAI Gymnasium's LunarLander-v3 environment. The agent learns to control thrust and steering to achieve safe, efficient landings on the moon's surface.

## 🎯 Project Overview

This project implements a sophisticated Deep Q-Network (DQN) agent optimized for CPU performance. The AI learns to:
- Navigate complex physics simulations
- Minimize fuel consumption
- Achieve precise landing accuracy
- Handle various environmental conditions

### Key Features
- **CPU-Optimized Training**: Enhanced TensorFlow configuration for maximum CPU performance
- **Advanced DQN Implementation**: Experience replay, target networks, and epsilon-greedy exploration
- **Comprehensive Monitoring**: Real-time training metrics and visualization
- **Multiple Play Modes**: Test trained models with different scenarios
- **Automated Saving**: Best model checkpointing during training

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM recommended
- Modern multi-core CPU for optimal training performance

### Setup Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/moonlander-dqn.git
cd moonlander-dqn
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Training a New Agent
```bash
python train.py
```

### Playing with a Trained Model
```bash
python play_best_model.py
```

### Testing Landing-Focused Training
```bash
python play_landing_focused.py
```

## 📊 Training Configuration

The training script supports various parameters:
- **Episodes**: Default 2000 training episodes
- **Batch Size**: 64 for optimal memory usage
- **Learning Rate**: 0.001 with Adam optimizer
- **Memory Buffer**: 100,000 experiences
- **Network Architecture**: Multi-layer neural network with batch normalization

## 📁 Project Structure

```
moonlander-dqn/
├── train.py                 # Main training script
├── play_best_model.py      # Play with trained models
├── play_landing_focused.py # Landing-specific gameplay
├── test_rewards.py         # Reward system testing
├── landing_fix.py          # Landing optimization utilities
├── requirements.txt        # Python dependencies
├── results/                # Training outputs and models
│   ├── current_run/       # Latest training session
│   └── landing_focused/   # Landing-specific models
├── old/                   # Archived training runs and experiments
└── .gitignore            # Git ignore configuration
```

## 🎮 How It Works

The agent uses a Deep Q-Network to learn optimal actions in the LunarLander environment:

1. **State Input**: 8-dimensional vector (position, velocity, angle, contact sensors)
2. **Action Output**: 4 discrete actions (no-op, left engine, main engine, right engine)
3. **Reward System**: Positive for safe landing, negative for crashes and fuel usage
4. **Learning Algorithm**: DQN with experience replay and target network updates

## 📈 Performance Metrics

The training process tracks:
- **Episode Rewards**: Total reward per episode
- **Success Rate**: Percentage of successful landings
- **Training Loss**: Q-network learning progress
- **Exploration Rate**: Epsilon decay over time

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI Gymnasium for the LunarLander environment
- TensorFlow team for the deep learning framework
- Reinforcement learning community for DQN research

### Training a New Model

```bash
# Train a new agent for 500 episodes
python train.py --mode train --episodes 500 --restart

# Train with rendering enabled (slower but visualizes the process)
python train.py --mode train --episodes 500 --render
```

### Playing with a Trained Model

```bash
# Watch the trained agent perform
python play_best_model.py

# Adjust playback speed (0.0 for max speed)
python play_best_model.py --delay 0.05

# Play multiple episodes
python play_best_model.py --episodes 10
```

## Command-line Arguments

### For Training (train.py)

- `--mode`: Mode to run (train or play) (default: train)
- `--output-dir`: Directory to save results (default: 'results')
- `--model-path`: Path to a specific model to load (optional)
- `--episodes`: Number of episodes for training (default: 500)
- `--restart`: Start training with a new model (optional)
- `--batch-size`: Batch size for training (default: 64)
- `--memory-size`: Size of replay memory (default: 10000)
- `--gamma`: Discount factor (default: 0.99)
- `--epsilon`: Initial exploration rate (default: 1.0)
- `--epsilon-min`: Minimum exploration rate (default: 0.1)
- `--epsilon-decay`: Exploration rate decay (default: 0.995)
- `--learning-rate`: Learning rate (default: 0.0005)
- `--update-target-freq`: Target network update frequency (default: 20)
- `--render`: Render the environment during training (optional)
- `--delay`: Delay between steps when rendering (default: 0.0 seconds)

### For Visualization (play_best_model.py)

- `--model-path`: Path to a specific model to load (optional, will find best model if not specified)
- `--episodes`: Number of episodes to play (default: 5)
- `--delay`: Delay between steps for visualization (default: 0.0 seconds)
- `--seed`: Random seed (default: 42)

## Neural Network Architecture

The project uses a balanced Deep Q-Network architecture optimized for stable learning:

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 64)                576       
_________________________________________________________________
dense_1 (Dense)              (None, 64)                4160      
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 260       
=================================================================
Total params: 4,996
Trainable params: 4,996
Non-trainable params: 0
_________________________________________________________________
```

### Key Features

1. **Stabilized Deep Q-Learning**:
   - Advanced reward scaling for LunarLander-v3's extreme rewards
   - Soft target network updates for stability
   - Multi-pass mini-batch training
   - Q-value regularization to prevent overestimation

2. **Optimized Hyperparameters**:
   - Conservative learning rate (0.0005) for stable learning
   - Balanced exploration with epsilon decay of 0.995
   - Appropriate discount factor (0.99) for this environment
   - Target network updates with soft blending (tau=0.1)

3. **Advanced Training Techniques**:
   - Special reward scaling:
     - Large rewards (>100) scaled to 10.0
     - Large negative rewards (<-100) scaled to -10.0
     - Medium rewards scaled proportionally by 0.1
   - Gradual Q-value updates with 5%-95% blending
   - Small time delays for CPU synchronization

4. **CPU Performance Optimizations**:
   - Efficient model architecture (64-64)
   - Multiple small batch updates instead of single large ones
   - Streamlined experience replay

## Environment

The project uses OpenAI Gymnasium's LunarLander-v3 environment which provides:

- **State Space**: 8-dimensional continuous state (position, velocity, angle, etc.)
- **Action Space**: 4 discrete actions (do nothing, fire left engine, fire main engine, fire right engine)
- **Rewards**:
  - Landing safely on the pad: +100 to +140
  - Crashing: -100
  - Fuel usage: small negative rewards
  - Moving toward landing pad: small positive rewards

## Learning Process

The agent uses several specialized techniques for stable learning:

1. **Experience Collection**:
   - Collects state, action, reward, next_state, done tuples
   - Applies specialized reward scaling to prevent oscillation
   - Maintains a replay memory of 10,000 recent experiences

2. **Training Process**:
   - Samples random batches of 64 experiences
   - Uses soft updates for Q-values (5% current, 95% new)
   - Processes each batch twice with 16-sample mini-batches 
   - Updates target network every 20 steps with soft blending

3. **Evaluation**:
   - Regularly evaluates agent during training
   - Saves best-performing model automatically
   - Provides detailed metrics on success rate

## Results Storage

Training results are stored in the `results/current_run` directory:
- Models are saved in `results/current_run/models/`
- Learning curves are saved in `results/current_run/plots/`
- The best performing model is saved as `best_model.weights.h5`

## Credits

- OpenAI Gymnasium for the LunarLander-v3 environment
- This implementation uses TensorFlow for neural network training